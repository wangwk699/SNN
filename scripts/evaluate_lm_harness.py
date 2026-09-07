from __future__ import annotations

import os
from types import MethodType

import torch

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
from snn2.lm_eval_protocol import (LM_EVAL_PINNED_REVISION, build_test_selection,
    correct_effective_sample_counts, enabled_lm_eval_task_specs, prune_empty_selected_leaves,
    selection_by_leaf)


def execution_counter_delta(before, after):
    """Return a task-local counter delta without changing the proxy's cumulative state."""
    return {key: int(after.get(key, 0)) - int(before.get(key, 0))
            for key in set(before) | set(after)}


def _leaf_tasks(task_tree):
    for name, value in task_tree.items():
        if isinstance(value, dict):
            yield from _leaf_tasks(value)
        else:
            yield name, value


def _selected_lm_eval_tasks(name, spec):
    """Build an evaluation-only doc view; few-shot datasets remain untouched."""
    from lm_eval.tasks import TaskManager
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
            for position, index in enumerate(_indices):
                if position % int(world_size) == int(rank):
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

    rank = int(
        os.environ.get(
            "RANK",
            "0",
        )
    )
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
        from lm_eval.models.huggingface import HFLM

        if torch.cuda.is_available():
            local_rank = int(
                os.environ.get(
                    "LOCAL_RANK",
                    "0",
                )
            )

            device = torch.device(
                "cuda",
                local_rank,
            )

            torch.cuda.set_device(
                device
            )

        else:
            device = torch.device(
                "cpu"
            )

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

        harness_model = HFLM(
            pretrained=proxy,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=int(
                cfg["data"]["max_seq_length"]
            ),
        )

        task_specs = enabled_lm_eval_task_specs(cfg)
        task_results: dict[str, dict] = {}
        for spec in task_specs:
            # cot is checked by enabled_lm_eval_task_specs; it is intentionally not
            # passed to simple_evaluate because lm-eval 0.4.8 has no such argument.
            task_manager, test_selection = _selected_lm_eval_tasks(spec["name"], spec)
            before_counter = dict(proxy.execution_counter)
            task_result = simple_evaluate(
                model=harness_model, tasks=[spec["name"]], task_manager=task_manager,
                num_fewshot=int(spec["num_fewshot"]), batch_size=batch_size, limit=None,
                random_seed=int(cfg["experiment"]["seed"]),
                numpy_random_seed=int(cfg["experiment"]["seed"]),
                torch_random_seed=int(cfg["experiment"]["seed"]),
                fewshot_random_seed=int(cfg["experiment"]["seed"]),
                apply_chat_template=bool(cfg["evaluation"].get("apply_chat_template", True)),
            )
            correct_effective_sample_counts(task_result, test_selection)
            after_counter = dict(proxy.execution_counter)
            task_counter = (execution_counter_delta(before_counter, after_counter)
                            if cfg["experiment"]["task"] == "tulu3" else after_counter)
            task_results[spec["name"]] = (task_result, test_selection, task_counter)

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
        if rank == 0:
            for spec in task_specs:
                name = spec["name"]
                task_result, selection, task_counter = task_results[name]
                actual = int(selection["selected_count"])
                task_temporal_forwards = task_counter.get("temporal_sample_step_forwards", 0)
                task_temporal_slots = task_counter.get("batched_temporal_sample_slots", 0)
                result = {
                    "tasks": {name: task_result},
                    "snn2_metadata": {**common_snn2_metadata,
                        "execution_counter": task_counter,
                        "activation_site_temporal_operator_calls": task_temporal_forwards * per_forward_operators,
                        "batched_activation_site_temporal_slots": task_temporal_slots * per_forward_operators,
                        "lm_eval_task_spec": spec, "lm_eval_revision": LM_EVAL_PINNED_REVISION,
                        "cot_semantic_source": "project_audited_pinned_lm_eval_0_4_8",
                        "test_sampling": selection["sampling"], "actual_test_samples": actual,
                        "fewshot_random_seed": int(cfg["experiment"]["seed"]), "test_seed": spec["test_seed"]},
                }
                output_dir = output_root / safe_name(name) / lm_eval_spec_dirname(spec)
                write_json(output_dir / "results.json", result)
                write_json(output_dir / "test_selection.json", selection)
                run.event("evaluation_saved", output_dir=str(output_dir),
                          model_variant=common_snn2_metadata["model_variant"], tasks=[name])


if __name__ == "__main__":
    main()
