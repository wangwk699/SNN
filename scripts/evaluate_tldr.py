from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch

from _common import parser, setup
from tqdm.auto import tqdm
from snn2.artifacts import write_json
from snn2.controller import SiteController
from snn2.data import _as_text, load_selected_raw
from snn2.evaluation import greedy_generate
from snn2.logging_utils import StageRun
from snn2.model_integration import install_model_integration
from snn2.modeling import load_model, load_tokenizer, model_source, prefix_ids, rotation_state


def _prompt_and_reference(row):
    prompt = _as_text(
        row.get("prompt", row.get("pompt", row.get("article", row.get("text", ""))))
    )
    reference = _as_text(
        row.get("completion", row.get("summary", row.get("label", row.get("response", ""))))
    )
    return prompt, reference


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    eval_parser = parser(
        "Evaluate a Base model, ANN checkpoint, or converted SNN on Reddit TL;DR",
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

    # 先读取 config，之后才能访问 cfg
    cfg, layout = setup(
        args.config,
        config_scope=(
            "base"
            if args.base
            else "run"
        ),
    )

    if cfg["experiment"]["task"] != "tldr":
        raise ValueError(
            "evaluate_tldr.py only accepts a TL;DR configuration"
        )

    # cfg 已经定义以后，再检查 Base evaluation 的合法性
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

    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            "1",
        )
    )

    model_variant = (
        "base"
        if args.base
        else args.neuron
    )

    stage = (
        f"evaluate_tldr_"
        f"{model_variant}_rank{rank}"
    )

    logs_dir = (
        layout.base_logs_dir
        if args.base
        else layout.logs_dir
    )

    with StageRun(stage, logs_dir, cfg["experiment"]) as run:
        from accelerate import Accelerator
        from accelerate.utils import gather_object
        import evaluate

        accelerator = Accelerator()

        if args.base:
            source = cfg["experiment"]["model_name"]
        else:
            source = model_source(cfg, layout, ann=True)

        model = load_model(cfg, source, training=False)
        model.to(accelerator.device)
        model.eval()
        tokenizer = load_tokenizer(cfg, source)
        tokenizer.padding_side = "left"
        controller = SiteController(mode="identity", site_root=layout.site_dir)
        steps = 1 if args.neuron == "ann" else controller.set_deployment(args.neuron)
        if args.neuron != "ann" or cfg["rotation"]["enabled"]:
            install_model_integration(model, controller, rotation_state(cfg, layout))
        prefixes = prefix_ids(cfg, layout)

        evaluation = load_selected_raw(cfg, layout).evaluation
        if evaluation is None:
            raise FileNotFoundError(
                "TL;DR evaluation manifest/test split is missing"
            )

        total_test_samples = len(evaluation)

        configured_test_samples = cfg["evaluation"].get(
            "tldr_test_samples"
        )

        tldr_test_seed = int(
            cfg["evaluation"].get(
                "tldr_test_seed",
                42,
            )
        )

        # --------------------------------------------------
        # Select TL;DR evaluation samples.
        #
        # tldr_test_samples = null
        #     -> full test split
        #
        # tldr_test_samples = N
        #     -> deterministic random subset of N samples,
        #        sampled without replacement.
        # --------------------------------------------------
        if configured_test_samples is None:
            selected_indices = list(
                range(total_test_samples)
            )

            is_full_test = True

        else:
            requested_test_samples = int(
                configured_test_samples
            )

            if requested_test_samples <= 0:
                raise ValueError(
                    "evaluation.tldr_test_samples must be "
                    "a positive integer or null"
                )

            if requested_test_samples >= total_test_samples:
                selected_indices = list(
                    range(total_test_samples)
                )

                is_full_test = True

            else:
                rng = random.Random(
                    tldr_test_seed
                )

                selected_indices = rng.sample(
                    range(total_test_samples),
                    k=requested_test_samples,
                )

                # 保持最终处理顺序与原始 test split 一致。
                # 不影响随机抽到的是哪些样本。
                selected_indices.sort()

                is_full_test = False


        selected_test_samples = len(
            selected_indices
        )

        if is_full_test:
            test_samples_dirname = (
                f"test_samples_{selected_test_samples}_full"
            )
            sampling_method = "full_split"
        else:
            test_samples_dirname = (
                f"test_samples_{selected_test_samples}"
            )
            sampling_method = (
                "seeded_random_without_replacement"
            )


        max_new = int(
            cfg["evaluation"].get(
                "max_new_tokens",
                32,
            )
        )

        input_length = int(
            cfg["evaluation"].get(
                "tldr_input_length",
                512,
            )
        )

        local_rows = []
        execution_counter: dict[str, int] = {}

        batch_size = int(cfg["evaluation"]["batch_size"])
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        local_indices = selected_indices[rank::world_size]

        progress = tqdm(
            total=len(local_indices),
            desc=f"Evaluating TL;DR ({args.neuron})",
            dynamic_ncols=True,
            disable=not accelerator.is_local_main_process,
        )

        for start in range(0, len(local_indices), batch_size):
            batch_indices = local_indices[start : start + batch_size]

            batch_input_ids = []
            batch_references = []

            # 先准备这一整个 batch 的样本
            for index in batch_indices:
                row = evaluation[index]
                prompt, reference = _prompt_and_reference(row)

                prompt_ids = tokenizer.encode(
                    prompt,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max(input_length - len(prefixes), 1),
                )

                input_ids = prefixes + prompt_ids

                batch_input_ids.append(input_ids)
                batch_references.append(reference)

            # 对不同长度的 prompt 做 padding
            padded = tokenizer.pad(
                {"input_ids": batch_input_ids},
                padding=True,
                return_tensors="pt",
            )

            tensor = padded["input_ids"].to(accelerator.device)
            mask = padded["attention_mask"].to(accelerator.device)

            # 一次 forward/generation 处理整个 batch
            output = greedy_generate(
                model,
                controller,
                tensor,
                mask,
                max_new_tokens=max_new,
                eos_token_id=tokenizer.eos_token_id,
                counter=execution_counter,
            )

            # 再逐条 decode batch 中的结果
            for batch_position, index in enumerate(batch_indices):
                decoded = tokenizer.decode(
                    output[batch_position].tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

                prediction = (
                    decoded.split("TL;DR:", 1)[1].strip()
                    if "TL;DR:" in decoded
                    else decoded.strip()
                )

                local_rows.append(
                    {
                        "index": index,
                        "prediction": prediction,
                        "reference": batch_references[batch_position],
                    }
                )

            # 进度仍然按照“样本数”显示，而不是 batch 数
            progress.update(len(batch_indices))

        progress.close()

        gathered = gather_object(local_rows)
        gathered_counters = gather_object([execution_counter])

        if accelerator.is_main_process:
            rows = sorted(
                gathered,
                key=lambda item: item["index"],
            )

            rouge = evaluate.load("rouge")

            metrics = rouge.compute(
                predictions=[
                    row["prediction"]
                    for row in rows
                ],
                references=[
                    row["reference"]
                    for row in rows
                ],
            )

            # -----------------------------
            # 汇总所有 evaluation processes
            # 的执行计数
            # -----------------------------
            model_forward_calls = sum(
                item.get(
                    "model_forward_calls",
                    0,
                )
                for item in gathered_counters
            )

            temporal_model_step_forwards = sum(
                item.get(
                    "temporal_model_step_forwards",
                    0,
                )
                for item in gathered_counters
            )

            sample_forward_equivalents = sum(
                item.get(
                    "sample_forward_equivalents",
                    0,
                )
                for item in gathered_counters
            )

            temporal_sample_step_forwards = sum(
                item.get(
                    "temporal_sample_step_forwards",
                    0,
                )
                for item in gathered_counters
            )

            batched_sample_slots = sum(
                item.get(
                    "batched_sample_slots",
                    0,
                )
                for item in gathered_counters
            )

            batched_temporal_sample_slots = sum(
                item.get(
                    "batched_temporal_sample_slots",
                    0,
                )
                for item in gathered_counters
            )

            layers = int(
                getattr(
                    model.config,
                    "num_hidden_layers",
                )
            )

            # ----------------------------------------
            # Batch-size-independent logical operator
            # equivalent count
            #
            # temporal sample steps
            # × Transformer layers
            # × 9 replacement sites
            # ----------------------------------------
            activation_site_temporal_operator_calls = (
                temporal_sample_step_forwards
                * layers
                * 9
            )

            # ----------------------------------------
            # Actual batched execution slot count
            #
            # 这里包含 batch 中已经 EOS、
            # 但因其它样本仍在生成而继续占据的 tensor slot。
            # ----------------------------------------
            batched_activation_site_temporal_slots = (
                batched_temporal_sample_slots
                * layers
                * 9
            )

            metrics.update(
                {
                    "samples": len(rows),

                    # TL;DR test sampling information
                    "total_test_samples": (
                        total_test_samples
                    ),
                    "tldr_test_samples": (
                        selected_test_samples
                    ),
                    "tldr_test_seed": (
                        None
                        if is_full_test
                        else tldr_test_seed
                    ),
                    "tldr_test_sampling": (
                        sampling_method
                    ),
                    "full_test_split": (
                        is_full_test
                    ),

                    "batch_size": batch_size,
                    "world_size": world_size,
                    "neuron": args.neuron,
                    "full_temporal_steps": steps,
                    "decode": "greedy",
                    "input_length": input_length,
                    "max_new_tokens": max_new,

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
                        cfg["experiment"].get("model_revision")
                        if args.base
                        else None
                    ),

                    # 实际 batched model 调用次数
                    "model_forward_calls": (
                        model_forward_calls
                    ),

                    # 实际 temporal model-step 调用次数
                    "temporal_model_step_forwards": (
                        temporal_model_step_forwards
                    ),

                    # batch-size-independent
                    # sample-level logical forwards
                    "sample_forward_equivalents": (
                        sample_forward_equivalents
                    ),

                    # batch-size-independent
                    # sample-level temporal forwards
                    "temporal_sample_step_forwards": (
                        temporal_sample_step_forwards
                    ),

                    # 实际占据 batch tensor 的 sample slots
                    "batched_sample_slots": (
                        batched_sample_slots
                    ),

                    "batched_temporal_sample_slots": (
                        batched_temporal_sample_slots
                    ),

                    # 后续论文中的主要 SNN 等价计算量
                    "activation_site_temporal_operator_calls": (
                        activation_site_temporal_operator_calls
                    ),

                    # 实际 batching 下的 slot-level 计算量
                    "batched_activation_site_temporal_slots": (
                        batched_activation_site_temporal_slots
                    ),
                }
            )

            if args.base:
                model_output_dir = layout.base_dir
            elif args.neuron == "ann":
                model_output_dir = layout.ann_dir
            else:
                model_output_dir = layout.snn_dir(args.neuron)       

            output_dir = (
                model_output_dir
                / "evaluation"
                / "tldr"
                / test_samples_dirname
            )

            _write_jsonl(
                output_dir / "predictions.jsonl",
                rows,
            )

            write_json(
                output_dir / "selection.json",
                {
                    "total_test_samples": (
                        total_test_samples
                    ),
                    "selected_test_samples": (
                        selected_test_samples
                    ),
                    "sampling": sampling_method,
                    "seed": (
                        None
                        if is_full_test
                        else tldr_test_seed
                    ),
                    "indices": selected_indices,
                },
            )

            write_json(
                output_dir / "metrics.json",
                metrics,
            )

            run.event(
                "evaluation_saved",
                output_dir=str(output_dir),
                **metrics,
            )

        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
