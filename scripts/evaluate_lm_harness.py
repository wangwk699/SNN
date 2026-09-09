from __future__ import annotations

import os
import copy
import time
from pathlib import Path
from types import MethodType

from accelerate import Accelerator

import torch
import logging
from contextlib import contextmanager

from _common import apply_deployment_overrides, parser, setup

from snn2.artifacts import lm_eval_spec_dirname, prefix_enabled_dirname, safe_name, read_json, write_json
from snn2.conversion import validate_conversion_metadata
from snn2.config import (
    final_ann_evaluation_prefix_enabled,
    evaluation_prefix_enabled,
    final_ann_evaluation_prefix_artifact_stage,
    final_snn_evaluation_prefix_artifact_stage,
    rotated_pre_finetuning_prefix_enabled,
)
from snn2.evaluation import (
    EvaluationModelProxy,
    activation_neuron_operators_per_temporal_forward,
    global_final_norm_evaluation_metadata,
    build_evaluation_controller,
    append_evaluation_num_samples_if_needed,
    deployment_policy_metadata,
    evaluation_calibration_metadata,
    evaluation_forward_metadata,
    evaluation_ann_common_clip_enabled,
)
from snn2.logging_utils import StageRun
from snn2.model_integration import install_model_integration
from snn2.sites import SITE_COUNT, SITE_TOPOLOGY_VERSION
from snn2.modeling import (
    load_model,
    load_tokenizer,
    model_source_for_stage,
    prefix_ids_for_stage,
    prefix_key_values_for_stage,
    rotation_state,
)
from snn2.training import validate_recorded_training_artifact_provenance
from snn2.lm_eval_distributed import (DistributedPreinitializedHFLM, distributed_max_seconds,
    gather_sum_execution_counter, indices_for_rank)
from snn2.lm_eval_protocol import (LM_EVAL_PINNED_REVISION, build_test_selection,
    correct_effective_sample_counts, enabled_lm_eval_task_specs, prune_empty_selected_leaves,
    extract_metric_value, result_contains_metric, seconds_to_hms, selection_by_leaf)


def execution_counter_delta(before, after):
    """Return a task-local counter delta without changing the proxy's cumulative state."""
    return {key: int(after.get(key, 0)) - int(before.get(key, 0))
            for key in set(before) | set(after)}


def _run_simple_evaluate(
    simple_evaluate,
    *,
    harness_model,
    spec,
    task_manager,
    batch_size: int,
    experiment_seed: int,
    apply_chat_template: bool,
    is_tulu_lm_eval: bool,
):
    """Run one lm-eval task with Tulu3 sample logging disabled at the source."""
    return simple_evaluate(
        model=harness_model,
        tasks=[spec["name"]],
        task_manager=task_manager,
        num_fewshot=int(spec["num_fewshot"]),
        batch_size=batch_size,
        limit=None,
        random_seed=experiment_seed,
        numpy_random_seed=experiment_seed,
        torch_random_seed=experiment_seed,
        fewshot_random_seed=experiment_seed,
        apply_chat_template=apply_chat_template,
        log_samples=False if is_tulu_lm_eval else True,
    )


def _write_json_atomic(path, payload):
    """Replace a repeatedly updated JSON file without exposing a partial write."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    write_json(temporary, payload)
    os.replace(temporary, path)
    return path


def _write_tulu_summary(output_root, *, task_times, task_metrics):
    summary = {
        "task_times": dict(task_times),
        "task_metrics": dict(task_metrics),
    }
    _write_json_atomic(Path(output_root) / "evaluation_summary.json", summary)
    return summary


def _write_task_result_files(output_dir, *, result, test_selection):
    """Persist one task immediately; task result objects never span task boundaries."""
    output_dir = Path(output_dir)
    write_json(output_dir / "results.json", result)
    write_json(output_dir / "test_selection.json", test_selection)
    return output_dir


class _ZeroShotLmEvalWarningFilter(logging.Filter):
    """Suppress known lm-eval warnings that are harmless for explicit 0-shot tasks."""

    def filter(self, record):
        message = record.getMessage()

        # ConfigurableTask initialization calls fewshot_docs() even for 0-shot
        # tasks. AGIEval tasks without train/validation splits therefore emit
        # this warning although no few-shot examples are actually used.
        if (
            record.name == "lm_eval.api.task"
            and "has_training_docs and has_validation_docs are False" in message
            and "using test_docs as fewshot_docs but this is not recommended" in message
        ):
            return False

        # The project explicitly passes num_fewshot=0 to simple_evaluate().
        # lm-eval reports the None -> 0 override as a warning.
        if (
            record.name == "lm_eval.evaluator"
            and message.startswith("Overwriting default num_fewshot of ")
            and " from None to 0" in message
        ):
            return False

        return True


@contextmanager
def _suppress_zero_shot_lm_eval_warnings(enabled):
    """Temporarily suppress only the two known lm-eval 0-shot warnings."""
    if not enabled:
        yield
        return

    warning_filter = _ZeroShotLmEvalWarningFilter()
    loggers = [
        logging.getLogger("lm_eval.api.task"),
        logging.getLogger("lm_eval.evaluator"),
    ]

    for logger in loggers:
        logger.addFilter(warning_filter)

    try:
        yield
    finally:
        for logger in loggers:
            logger.removeFilter(warning_filter)


def _leaf_tasks(task_tree):
    for name, value in task_tree.items():
        if isinstance(value, dict):
            yield from _leaf_tasks(value)
        else:
            yield name, value

@contextmanager
def _prefer_local_dataset_cache():
    """
    Make datasets.load_dataset() cache-first:

    1. Try strictly local cache first.
    2. If the local cache is insufficient, retry normally and allow download.

    This changes only dataset loading policy.
    """
    import datasets
    from datasets import DownloadConfig

    original_load_dataset = datasets.load_dataset

    def cache_first_load_dataset(*args, **kwargs):
        local_kwargs = dict(kwargs)

        existing_download_config = local_kwargs.get("download_config")

        if existing_download_config is None:
            local_download_config = DownloadConfig(
                local_files_only=True
            )
        else:
            local_download_config = copy.copy(
                existing_download_config
            )
            local_download_config.local_files_only = True

        local_kwargs["download_config"] = local_download_config

        try:
            # First attempt: strictly local cache, no Hub download.
            return original_load_dataset(
                *args,
                **local_kwargs,
            )
        except Exception:
            # Local cache is unavailable/incomplete.
            # Retry with the original arguments, which allows Hub access
            # and preserves Hugging Face's normal caching behavior.
            return original_load_dataset(
                *args,
                **kwargs,
            )

    datasets.load_dataset = cache_first_load_dataset

    try:
        yield
    finally:
        datasets.load_dataset = original_load_dataset

def _selected_lm_eval_tasks(name, spec):
    """Build an evaluation-only doc view; few-shot datasets remain untouched."""
    from lm_eval.tasks import TaskManager
    with _prefer_local_dataset_cache():
        manager = TaskManager()
        task_tree = manager.load_task_or_group([name])
    leaves = list(_leaf_tasks(task_tree))
    selection = build_test_selection(
        {leaf_name: len(task.eval_docs) for leaf_name, task in leaves},
        task=name, test_samples=spec["test_samples"], test_seed=int(spec["test_seed"]),
    )
    selected = selection_by_leaf(selection)
    task_tree = prune_empty_selected_leaves(task_tree, selected)
    leaves = list(_leaf_tasks(task_tree))
    for leaf_name, task in leaves:
        indices = sorted(selected[leaf_name])
        def doc_iterator(self, *, rank=0, limit=None, world_size=1, _indices=indices):
            # Limit is intentionally ignored: this wrapper is the deterministic test view.
            for index in indices_for_rank(_indices, rank=int(rank), world_size=int(world_size)):
                yield index, self.eval_docs[index]
        task.doc_iterator = MethodType(doc_iterator, task)
    # simple_evaluate resolves by name again. Return this already wrapped tree so group
    # aggregation is retained without mutating a dataset or its few-shot pool.
    manager.load_task_or_group = lambda task_list=None: task_tree
    return manager, selection


def main():
    eval_parser = parser(
        "Evaluate a Base model, ANN checkpoint, or converted SNN "
        "with lm-evaluation-harness",
        neuron=True,
        allow_ann=True,
    )

    model_variant_group = eval_parser.add_mutually_exclusive_group()
    model_variant_group.add_argument(
        "--base",
        action="store_true",
        help="Evaluate the original pretrained Base model",
    )
    model_variant_group.add_argument(
        "--rotated-pre-finetuning",
        action="store_true",
        help="Evaluate the rotated fused Base checkpoint before ANN fine-tuning",
    )

    args = eval_parser.parse_args()

    # Base evaluation 的 resolved_config 独立保存，
    # 不进入 ann_mode/lr... 的 run 目录。
    cfg, layout = setup(
        args.config,
        config_scope=(
            "base"
            if args.base
            else ("rotated_pre_finetuning" if args.rotated_pre_finetuning else "run")
        ),
    )
    apply_deployment_overrides(args, cfg)
    accelerator = Accelerator()
    rank = int(accelerator.process_index)
    world_size = int(accelerator.num_processes)
    device = accelerator.device
    env_rank = int(os.environ.get("RANK", rank))
    if env_rank != rank:
        raise RuntimeError(f"Accelerate rank mismatch: environment={env_rank}, accelerator={rank}")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # --------------------------------------------------
    # Base evaluation 只能是原始 ANN，
    # 并且必须使用 vanilla config：
    #
    # rotation = False
    # prefix   = False
    # replacement = none
    # --------------------------------------------------
    if args.base:
        if args.neuron != "ann":
            raise ValueError(
                "--base can only be used with --neuron ann"
            )

        if cfg["experiment"]["ann_mode"] != "vanilla":
            raise ValueError(
                "--base must use a vanilla configuration"
            )

    if args.rotated_pre_finetuning:
        if args.neuron != "ann":
            raise ValueError("--rotated-pre-finetuning can only be used with --neuron ann")
        if not cfg["rotation"]["enabled"]:
            raise ValueError("--rotated-pre-finetuning requires rotation.enabled=true")
        required = [
            layout.rotation_dir / "fused_base" / "config.json",
            layout.rotation_dir / "rotation_state.pt",
        ]
        if rotated_pre_finetuning_prefix_enabled(cfg):
            state_path = layout.rotated_pre_finetuning_prefix_dir / "prefix_state.json"
            required.append(state_path)
            if state_path.exists() and read_json(state_path).get("prefix_token_ids", []):
                required.append(layout.rotated_pre_finetuning_prefix_dir / "prefixed_key_values.pt")
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Rotated pre-finetuning evaluation dependencies are missing: "
                + ", ".join(missing)
                + ". Run prepare_rotation.py and, when enabled, "
                + "discover_prefix.py --stage pre_finetuning."
            )

    if args.neuron != "ann" and not args.base and not args.rotated_pre_finetuning:
        validate_conversion_metadata(cfg, layout, args.neuron)

    active_prefix_enabled = (
        False if args.base else (
            rotated_pre_finetuning_prefix_enabled(cfg)
            if args.rotated_pre_finetuning
            else (final_ann_evaluation_prefix_enabled(cfg) if args.neuron == "ann" else evaluation_prefix_enabled(cfg))
        )
    )

    model_variant = (
        "base"
        if args.base
        else ("rotated_pre_finetuning_ann" if args.rotated_pre_finetuning else args.neuron)
    )

    stage = (
        f"evaluate_lm_harness_"
        f"{model_variant}_rank{rank}"
    )

    logs_dir = (
        layout.base_logs_dir
        if args.base
        else (layout.rotated_pre_finetuning_logs_dir if args.rotated_pre_finetuning else layout.logs_dir)
    )

    with StageRun(
        stage,
        logs_dir,
        cfg["experiment"],
    ) as run:

        from lm_eval import simple_evaluate

        # --------------------------------------------------
        # Model source
        #
        # --base:
        #     直接加载原始 pretrained Base model
        #
        # normal --neuron ann:
        #     加载 fine-tuned ann/final
        # --------------------------------------------------
        if args.base:
            checkpoint_stage = "base_evaluation"
        elif args.rotated_pre_finetuning:
            checkpoint_stage = "rotated_pre_finetuning"
        else:
            checkpoint_stage = "post_finetuning"
        source = model_source_for_stage(cfg, layout, stage=checkpoint_stage)

        model = load_model(
            cfg,
            source,
            training=False,
        )

        model.to(
            device
        )

        model.eval()

        tokenizer = load_tokenizer(
            cfg,
            source,
        )

        if (
            args.neuron == "ann"
            and not args.base
            and not args.rotated_pre_finetuning
            and cfg["experiment"]["ann_mode"] in {"phase_aware", "gif_aware"}
        ):
            validate_recorded_training_artifact_provenance(cfg, layout)
        controller, steps = build_evaluation_controller(
            cfg,
            layout,
            neuron=args.neuron,
            base=args.base,
            rotated_pre_finetuning=args.rotated_pre_finetuning,
        )

        # Base + vanilla + ann：
        #
        # args.neuron == ann
        # rotation.enabled == False
        #
        # 所以不会进入这里，
        # 保证原始 Base 模型不安装 SNN2 integration。
        if (
            args.neuron != "ann"
            or cfg["rotation"]["enabled"]
        ):
            install_model_integration(
                model,
                controller,
                rotation_state(
                    cfg,
                    layout,
                ),
            )

        if args.base:
            prefix_stage = "base_evaluation"
        elif args.rotated_pre_finetuning:
            prefix_stage = "rotated_pre_finetuning"
        elif args.neuron == "ann":
            prefix_stage = "final_ann_evaluation"
        else:
            prefix_stage = "final_snn_evaluation"
        model_prefix_ids = prefix_ids_for_stage(cfg, layout, stage=prefix_stage)

        proxy = EvaluationModelProxy(
            model,
            controller,
            prefix_key_values_for_stage(cfg, layout, stage=prefix_stage),
        )

        batch_size = int(
            cfg["evaluation"].get(
                "batch_size",
                1,
            )
        )

        harness_model = DistributedPreinitializedHFLM(
            accelerator=accelerator,
            pretrained=proxy,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=int(
                cfg["data"]["max_seq_length"]
            ),
        )
        if harness_model.rank != rank or harness_model.world_size != world_size:
            raise RuntimeError(
                f"lm-eval distributed binding mismatch: rank={harness_model.rank}/{rank}, "
                f"world_size={harness_model.world_size}/{world_size}"
            )

        task_specs = enabled_lm_eval_task_specs(cfg)
        is_tulu_lm_eval = cfg["experiment"]["task"] == "tulu3"

        layers = int(
            getattr(
                model.config,
                "num_hidden_layers",
            )
        )

        per_forward_operators = activation_neuron_operators_per_temporal_forward(
            num_hidden_layers=layers, neuron=args.neuron
        )
        common_snn2_metadata = {
            # ----------------------------------
            # 明确区分原始 Base 与 fine-tuned ANN
            # ----------------------------------
            "model_variant": (
                "base"
                if args.base
                else (
                    "rotated_pre_finetuning_ann"
                    if args.rotated_pre_finetuning
                    else (
                    "finetuned_ann"
                    if args.neuron == "ann"
                    else f"snn_{args.neuron}"
                    )
                )
            ),

            "model_source": source,

            "model_revision": (
                cfg["experiment"].get(
                    "model_revision"
                )
                if args.base
                else None
            ),

            "neuron": args.neuron,

            "full_temporal_steps": steps,

            "site_count": SITE_COUNT,

            "site_topology_version": SITE_TOPOLOGY_VERSION,
            "per_temporal_forward_activation_neuron_operators": per_forward_operators,
            **global_final_norm_evaluation_metadata(neuron=args.neuron, controller=controller),
            "ann_training_common_clip_enabled": (
                False
                if args.base or args.rotated_pre_finetuning
                else evaluation_ann_common_clip_enabled(cfg)
            ),
            **deployment_policy_metadata(controller),

            "batch_size": batch_size,
            "evaluation_parallelism": (
                "lm_eval_document_data_parallel" if world_size > 1 else "single_process"
            ),
            "evaluation_world_size": world_size,
            "evaluation_batch_size_per_rank": batch_size,
            "evaluation_model_replication": "one_full_model_per_process",

            "prefix_token_ids": (
                model_prefix_ids
            ),
            "prefix_stage": prefix_stage,
            "prefix_enabled": active_prefix_enabled,
            "prefix_root": (
                None
                if args.base or args.rotated_pre_finetuning or not active_prefix_enabled
                else str(
                    (
                        layout.ann_training_prefix_dir
                        if (
                            final_ann_evaluation_prefix_artifact_stage(cfg)
                            if args.neuron == "ann"
                            else final_snn_evaluation_prefix_artifact_stage(cfg)
                        ) == "pre_finetuning"
                        else layout.post_finetuning_prefix_dir
                    )
                )
            ),
            "prefix_source_stage": (
                None if args.base or args.rotated_pre_finetuning
                else (
                    final_ann_evaluation_prefix_artifact_stage(cfg)
                    if args.neuron == "ann"
                    else final_snn_evaluation_prefix_artifact_stage(cfg)
                )
            ),
            **evaluation_calibration_metadata(
                cfg,
                layout,
                neuron=args.neuron,
                base=args.base,
                rotated_pre_finetuning=args.rotated_pre_finetuning,
            ),
            **evaluation_forward_metadata(
                cfg,
                layout,
                neuron=args.neuron,
                controller=controller,
                base=args.base,
                rotated_pre_finetuning=args.rotated_pre_finetuning,
            ),

        }

        # --------------------------------------------------
        # Output directory
        #
        # Base:
        #   <model>/base/seed42/evaluation/<task>/<spec>/
        #
        # Fine-tuned ANN:
        #   <model>/<ann_mode>/<lr>/<run_variant>/
        #       [surrogate_slope_<value>_warmup_ratio_<value>/]seed42/
        #       ann/evaluation/<task>/<spec>/
        #
        # SNN:
        #   <model>/<ann_mode>/<lr>/<run_variant>/
        #       [surrogate_slope_<value>_warmup_ratio_<value>/]seed42/
        #       snn/<neuron>/evaluation/<task>/<spec>/
        # --------------------------------------------------
        if args.base:
            model_output_dir = (
                layout.base_dir
            )

        elif args.rotated_pre_finetuning:
            model_output_dir = layout.rotated_pre_finetuning_dir

        elif args.neuron == "ann":
            model_output_dir = (
                layout.ann_dir
            )

        else:
            model_output_dir = (
                layout.snn_dir(
                    args.neuron
                )
            )

        output_root = model_output_dir / "evaluation"
        if not args.base:
            output_root = output_root / prefix_enabled_dirname(active_prefix_enabled)
        output_root = append_evaluation_num_samples_if_needed(
            output_root, cfg, base=args.base,
            rotated_pre_finetuning=args.rotated_pre_finetuning, neuron=args.neuron,
        )
        task_results_root = output_root / "task_results" if is_tulu_lm_eval else output_root
        task_times = {}
        task_metrics = {}

        for spec in task_specs:
            # cot is checked by enabled_lm_eval_task_specs; it is intentionally not
            # passed to simple_evaluate because lm-eval 0.4.8 has no such argument.
            task_started = time.perf_counter() if is_tulu_lm_eval else None
            with _suppress_zero_shot_lm_eval_warnings(enabled=int(spec["num_fewshot"]) == 0):
                task_manager, test_selection = _selected_lm_eval_tasks(spec["name"], spec)
                setup_seconds = (time.perf_counter() - task_started if task_started is not None else None)
                before_counter = dict(proxy.execution_counter)
                evaluation_started = (time.perf_counter() if is_tulu_lm_eval else None)
                task_result = _run_simple_evaluate(
                    simple_evaluate,
                    harness_model=harness_model,
                    spec=spec,
                    task_manager=task_manager,
                    batch_size=batch_size,
                    experiment_seed=int(cfg["experiment"]["seed"]),
                    apply_chat_template=bool(cfg["evaluation"].get("apply_chat_template", True)),
                    is_tulu_lm_eval=is_tulu_lm_eval,
                )
            local_timing = (
                {
                    "selection_setup_seconds": setup_seconds,
                    "lm_eval_seconds": time.perf_counter() - evaluation_started,
                    "total_seconds": time.perf_counter() - task_started,
                    "selected_documents": int(test_selection["selected_count"]),
                }
                if is_tulu_lm_eval else None
            )
            after_counter = dict(proxy.execution_counter)
            local_counter = (execution_counter_delta(before_counter, after_counter)
                             if is_tulu_lm_eval else after_counter)
            task_counter = gather_sum_execution_counter(local_counter, world_size=world_size)
            timing = (
                {
                    "selection_setup_seconds": distributed_max_seconds(
                        local_timing["selection_setup_seconds"], device=device, world_size=world_size),
                    "lm_eval_seconds": distributed_max_seconds(
                        local_timing["lm_eval_seconds"], device=device, world_size=world_size),
                    "total_seconds": distributed_max_seconds(
                        local_timing["total_seconds"], device=device, world_size=world_size),
                    "selected_documents": local_timing["selected_documents"],
                }
                if is_tulu_lm_eval else None
            )
            if accelerator.is_main_process:
                if task_result is None:
                    raise RuntimeError(f"lm-eval returned no result on main rank for {spec['name']!r}")
                correct_effective_sample_counts(task_result, test_selection)
                if not result_contains_metric(task_result, spec["metric"]):
                    raise ValueError(f"lm-eval result for {spec['name']!r} does not contain configured metric {spec['metric']!r}")
                name = spec["name"]
                actual = int(test_selection["selected_count"])
                task_temporal_forwards = task_counter.get("temporal_sample_step_forwards", 0)
                task_temporal_slots = task_counter.get("batched_temporal_sample_slots", 0)
                result = {
                    "tasks": {name: task_result},
                    "snn2_metadata": {**common_snn2_metadata,
                        "execution_counter": task_counter,
                        **({"evaluation_timing": timing} if is_tulu_lm_eval else {}),
                        "activation_site_temporal_operator_calls": task_temporal_forwards * per_forward_operators,
                        "batched_activation_site_temporal_slots": task_temporal_slots * per_forward_operators,
                        "lm_eval_task_spec": spec, "lm_eval_revision": LM_EVAL_PINNED_REVISION,
                        "cot_semantic_source": "project_audited_pinned_lm_eval_0_4_8",
                        "test_sampling": test_selection["sampling"], "actual_test_samples": actual,
                        "fewshot_random_seed": int(cfg["experiment"]["seed"]), "test_seed": spec["test_seed"]},
                }
                output_dir = task_results_root / safe_name(name) / lm_eval_spec_dirname(spec)
                _write_task_result_files(
                    output_dir, result=result, test_selection=test_selection
                )
                if is_tulu_lm_eval:
                    task_times[name] = seconds_to_hms(timing["lm_eval_seconds"])
                    task_metrics[name] = extract_metric_value(
                        task_result, spec["metric"], task_name=name
                    )
                    _write_tulu_summary(
                        output_root,
                        task_times=task_times,
                        task_metrics=task_metrics,
                    )
                run.event("evaluation_saved", output_dir=str(output_dir),
                          model_variant=common_snn2_metadata["model_variant"], tasks=[name],
                          **({"lm_eval_seconds": timing["lm_eval_seconds"],
                              "total_seconds": timing["total_seconds"],
                              "selected_documents": timing["selected_documents"]}
                             if is_tulu_lm_eval else {}))
                del result
            elif task_result is not None:
                raise RuntimeError(f"lm-eval unexpectedly returned a result on worker rank {rank} for {spec['name']!r}")
            del task_result
            del test_selection
            del task_manager
            accelerator.wait_for_everyone()

        if accelerator.is_main_process and is_tulu_lm_eval:
            summary = _write_tulu_summary(
                output_root,
                task_times=task_times,
                task_metrics=task_metrics,
            )
            run.event("lm_eval_summary_saved", output_dir=str(output_root), **summary)


if __name__ == "__main__":
    main()
