from __future__ import annotations

import os

import torch

from _common import parser, setup

from snn2.artifacts import prefix_enabled_dirname, read_json, write_json
from snn2.config import (
    evaluation_prefix_enabled,
    final_evaluation_prefix_artifact_stage,
    rotated_pre_finetuning_prefix_enabled,
)
from snn2.conversion import validate_conversion_metadata
from snn2.evaluation import (
    EvaluationModelProxy,
    activation_neuron_operators_per_temporal_forward,
    build_evaluation_controller,
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
        False
        if args.base
        else (
            rotated_pre_finetuning_prefix_enabled(cfg)
            if args.rotated_pre_finetuning
            else evaluation_prefix_enabled(cfg)
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

        model_prefix_ids = prefix_ids_for_stage(
            cfg, layout, stage=(
                "base_evaluation" if args.base else (
                    "rotated_pre_finetuning" if args.rotated_pre_finetuning else "final_evaluation"
                )
            )
        )

        # Base + vanilla 时这里自然为 []
        proxy = EvaluationModelProxy(
            model,
            controller,
            prefix_key_values_for_stage(cfg, layout, stage=(
                "base_evaluation" if args.base else (
                    "rotated_pre_finetuning" if args.rotated_pre_finetuning else "final_evaluation"
                )
            )),
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

        task_specs = list(
            cfg["evaluation"].get(
                "lm_eval_task_specs",
                [],
            )
        )

        if not task_specs:
            raise ValueError(
                "evaluation.lm_eval_task_specs "
                "must contain at least one task"
            )

        results = {
            "tasks": {},
            "task_specs": task_specs,
        }

        for spec in task_specs:
            name = spec["name"]

            task_result = simple_evaluate(
                model=harness_model,
                tasks=[name],
                num_fewshot=int(
                    spec["num_fewshot"]
                ),
                batch_size=batch_size,
                limit=cfg["evaluation"].get(
                    "limit"
                ),
                random_seed=int(
                    cfg["experiment"]["seed"]
                ),
                numpy_random_seed=int(
                    cfg["experiment"]["seed"]
                ),
                torch_random_seed=int(
                    cfg["experiment"]["seed"]
                ),
                fewshot_random_seed=int(
                    cfg["experiment"]["seed"]
                ),
                apply_chat_template=bool(
                    cfg["evaluation"].get(
                        "apply_chat_template",
                        True,
                    )
                ),
            )

            results["tasks"][
                name
            ] = task_result

        layers = int(
            getattr(
                model.config,
                "num_hidden_layers",
            )
        )

        execution_counter = dict(
            proxy.execution_counter
        )

        temporal_sample_step_forwards = (
            execution_counter.get(
                "temporal_sample_step_forwards",
                0,
            )
        )

        batched_temporal_sample_slots = (
            execution_counter.get(
                "batched_temporal_sample_slots",
                0,
            )
        )

        per_forward_operators = activation_neuron_operators_per_temporal_forward(
            num_hidden_layers=layers, neuron=args.neuron
        )
        results["snn2_metadata"] = {
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
            "global_final_norm_phase_neuron_present": args.neuron == "phase",
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
            "prefix_stage": (
                "base_evaluation" if args.base else (
                    "rotated_pre_finetuning" if args.rotated_pre_finetuning else "final_evaluation"
                )
            ),
            "prefix_enabled": active_prefix_enabled,
            "prefix_source_stage": (
                None if args.base or args.rotated_pre_finetuning
                else final_evaluation_prefix_artifact_stage(cfg)
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

            # 保存全部原始 execution counter
            "execution_counter": (
                execution_counter
            ),

            # Batch-size-independent logical
            # activation-site operator equivalents
            "activation_site_temporal_operator_calls": (
                temporal_sample_step_forwards * per_forward_operators
            ),

            # Actual batched sample-slot execution
            "batched_activation_site_temporal_slots": (
                batched_temporal_sample_slots * per_forward_operators
            ),
        }

        # --------------------------------------------------
        # Output directory
        #
        # Base:
        #   <model>/base/seed42/evaluation/lm_harness/
        #
        # Fine-tuned ANN:
        #   <model>/<ann_mode>/<lr>/seed42/
        #       ann/evaluation/lm_harness/
        #
        # SNN:
        #   <model>/<ann_mode>/<lr>/seed42/
        #       snn/<neuron>/evaluation/lm_harness/
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

        output_dir = (
            model_output_dir
            / "evaluation"
            / "lm_harness"
        )

        if not args.base:
            output_dir = output_dir / prefix_enabled_dirname(active_prefix_enabled)

        if rank == 0:
            write_json(
                output_dir / "results.json",
                results,
            )

            run.event(
                "evaluation_saved",
                output_dir=str(
                    output_dir
                ),
                model_variant=(
                    results[
                        "snn2_metadata"
                    ][
                        "model_variant"
                    ]
                ),
                tasks=[
                    spec["name"]
                    for spec in task_specs
                ],
            )


if __name__ == "__main__":
    main()
