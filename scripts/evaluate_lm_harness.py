from __future__ import annotations

import os

import torch

from _common import parser, setup

from snn2.artifacts import write_json
from snn2.controller import SiteController
from snn2.evaluation import EvaluationModelProxy
from snn2.logging_utils import StageRun
from snn2.model_integration import install_model_integration
from snn2.modeling import load_model, load_tokenizer, model_source, prefix_ids, rotation_state


def main():
    args = parser(
        "Evaluate an ANN checkpoint or converted SNN with lm-evaluation-harness",
        neuron=True,
        allow_ann=True,
    ).parse_args()
    cfg, layout = setup(args.config)
    rank = int(os.environ.get("RANK", "0"))
    stage = f"evaluate_lm_harness_{args.neuron}_rank{rank}"
    with StageRun(stage, layout.logs_dir, cfg["experiment"]) as run:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM

        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        source = model_source(cfg, layout, ann=True)
        model = load_model(cfg, source, training=False)
        model.to(device)
        model.eval()
        tokenizer = load_tokenizer(cfg, source)
        controller = SiteController(mode="identity", site_root=layout.site_dir)
        steps = 1 if args.neuron == "ann" else controller.set_deployment(args.neuron)
        if args.neuron != "ann" or cfg["rotation"]["enabled"]:
            install_model_integration(model, controller, rotation_state(cfg, layout))
        proxy = EvaluationModelProxy(model, controller, prefix_ids(cfg, layout))
        harness_model = HFLM(
            pretrained=proxy,
            tokenizer=tokenizer,
            batch_size=int(cfg["evaluation"].get("batch_size", 1)),
            max_length=int(cfg["data"]["max_seq_length"]),
        )
        task_specs = list(cfg["evaluation"].get("lm_eval_task_specs", []))
        if not task_specs:
            raise ValueError("evaluation.lm_eval_task_specs must contain at least one task")
        results = {"tasks": {}, "task_specs": task_specs}
        for spec in task_specs:
            name = spec["name"]
            task_result = simple_evaluate(
                model=harness_model,
                tasks=[name],
                num_fewshot=int(spec["num_fewshot"]),
                batch_size=int(cfg["evaluation"].get("batch_size", 1)),
                limit=cfg["evaluation"].get("limit"),
                random_seed=int(cfg["experiment"]["seed"]),
                numpy_random_seed=int(cfg["experiment"]["seed"]),
                torch_random_seed=int(cfg["experiment"]["seed"]),
                fewshot_random_seed=int(cfg["experiment"]["seed"]),
                apply_chat_template=bool(
                    cfg["evaluation"].get("apply_chat_template", True)
                ),
            )
            results["tasks"][name] = task_result
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
            "neuron": args.neuron,
            "full_temporal_steps": steps,
            "batch_size": int(
                cfg["evaluation"].get(
                    "batch_size",
                    1,
                )
            ),
            "prefix_token_ids": prefix_ids(
                cfg,
                layout,
            ),

            # 保存全部原始 execution counter
            "execution_counter": execution_counter,

            # ----------------------------------------
            # Batch-size-independent logical
            # activation-site operator equivalents
            # ----------------------------------------
            "activation_site_temporal_operator_calls": (
                temporal_sample_step_forwards
                * layers
                * 9
            ),

            # ----------------------------------------
            # Actual batched sample-slot execution
            # ----------------------------------------
            "batched_activation_site_temporal_slots": (
                batched_temporal_sample_slots
                * layers
                * 9
            ),
        }
        output_dir = (
            layout.ann_dir if args.neuron == "ann" else layout.snn_dir(args.neuron)
        ) / "evaluation" / "lm_harness"
        if rank == 0:
            write_json(output_dir / "results.json", results)
            run.event(
                "evaluation_saved",
                output_dir=str(output_dir),
                tasks=[spec["name"] for spec in task_specs],
            )


if __name__ == "__main__":
    main()
