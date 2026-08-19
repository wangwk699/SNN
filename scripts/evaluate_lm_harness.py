from __future__ import annotations

import os

import torch

from _common import parser, setup

from snn2.artifacts import write_json
from snn2.controller import SiteController
from snn2.evaluation import EvaluationModelProxy
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


def main():
    eval_parser = parser(
        "Evaluate a Base model, ANN checkpoint, or converted SNN "
        "with lm-evaluation-harness",
        neuron=True,
        allow_ann=True,
    )

    eval_parser.add_argument(
        "--base",
        action="store_true",
        help=(
            "Evaluate the original pretrained Base model instead "
            "of the fine-tuned ANN checkpoint"
        ),
    )

    args = eval_parser.parse_args()

    # Base evaluation 的 resolved_config 独立保存，
    # 不进入 ann_mode/lr... 的 run 目录。
    cfg, layout = setup(
        args.config,
        config_scope=(
            "base"
            if args.base
            else "run"
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

    rank = int(
        os.environ.get(
            "RANK",
            "0",
        )
    )

    model_variant = (
        "base"
        if args.base
        else args.neuron
    )

    stage = (
        f"evaluate_lm_harness_"
        f"{model_variant}_rank{rank}"
    )

    logs_dir = (
        layout.base_logs_dir
        if args.base
        else layout.logs_dir
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
            source = cfg["experiment"]["model_name"]
        else:
            source = model_source_for_stage(cfg, layout, stage="post_finetuning")

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

        controller = SiteController(
            mode="identity",
            site_root=layout.post_finetuning_site_dir if not args.base else None,
        )

        steps = (
            1
            if args.neuron == "ann"
            else controller.set_deployment(
                args.neuron
            )
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
            cfg, layout, stage="base_evaluation" if args.base else "post_finetuning"
        )

        # Base + vanilla 时这里自然为 []
        proxy = EvaluationModelProxy(
            model,
            controller,
            prefix_key_values_for_stage(cfg, layout, stage="base_evaluation" if args.base else "post_finetuning"),
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

        results["snn2_metadata"] = {
            # ----------------------------------
            # 明确区分原始 Base 与 fine-tuned ANN
            # ----------------------------------
            "model_variant": (
                "base"
                if args.base
                else (
                    "finetuned_ann"
                    if args.neuron == "ann"
                    else f"snn_{args.neuron}"
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

            "batch_size": batch_size,

            "prefix_token_ids": (
                model_prefix_ids
            ),
            "prefix_stage": "base_evaluation" if args.base else "post_finetuning",
            "post_finetuning_recalibration": False if args.base else True,
            "calibration_root": None if args.base else str(layout.post_finetuning_site_dir),

            # 保存全部原始 execution counter
            "execution_counter": (
                execution_counter
            ),

            # Batch-size-independent logical
            # activation-site operator equivalents
            "activation_site_temporal_operator_calls": (
                temporal_sample_step_forwards
                * layers
                * SITE_COUNT
            ),

            # Actual batched sample-slot execution
            "batched_activation_site_temporal_slots": (
                batched_temporal_sample_slots
                * layers
                * SITE_COUNT
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