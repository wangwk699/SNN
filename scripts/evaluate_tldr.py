from __future__ import annotations

import json
import os
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
    args = parser(
        "Evaluate an ANN checkpoint or converted SNN on Reddit TL;DR",
        neuron=True,
        allow_ann=True,
    ).parse_args()
    cfg, layout = setup(args.config)
    if cfg["experiment"]["task"] != "tldr":
        raise ValueError("evaluate_tldr.py only accepts a TL;DR configuration")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    stage = f"evaluate_tldr_{args.neuron}_rank{rank}"
    with StageRun(stage, layout.logs_dir, cfg["experiment"]) as run:
        from accelerate import Accelerator
        from accelerate.utils import gather_object
        import evaluate

        accelerator = Accelerator()
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
            raise FileNotFoundError("TL;DR evaluation manifest/test split is missing")
        max_samples = cfg["evaluation"].get("max_samples")
        stop = len(evaluation)
        if max_samples is not None:
            stop = min(stop, int(max_samples))
        max_new = int(cfg["evaluation"].get("max_new_tokens", 32))
        input_length = int(cfg["evaluation"].get("tldr_input_length", 512))

        local_rows = []
        execution_counter: dict[str, int] = {}

        batch_size = int(cfg["evaluation"]["batch_size"])
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        local_indices = range(rank, stop, world_size)

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
            rows = sorted(gathered, key=lambda item: item["index"])
            rouge = evaluate.load("rouge")
            metrics = rouge.compute(
                predictions=[row["prediction"] for row in rows],
                references=[row["reference"] for row in rows],
            )
            metrics.update(
                {
                    "samples": len(rows),
                    "neuron": args.neuron,
                    "full_temporal_steps": steps,
                    "decode": "greedy",
                    "input_length": input_length,
                    "max_new_tokens": max_new,
                    "roste_alignment": "test[:6528], decode full sequence, split on TL;DR:",
                    "model_forward_calls": sum(
                        item.get("model_forward_calls", 0) for item in gathered_counters
                    ),
                    "temporal_model_step_forwards": sum(
                        item.get("temporal_model_step_forwards", 0)
                        for item in gathered_counters
                    ),
                }
            )
            layers = int(getattr(model.config, "num_hidden_layers"))
            metrics["activation_site_temporal_operator_calls"] = (
                metrics["temporal_model_step_forwards"] * layers * 9
            )
            output_dir = (
                layout.ann_dir if args.neuron == "ann" else layout.snn_dir(args.neuron)
            ) / "evaluation" / "tldr"
            _write_jsonl(output_dir / "predictions.jsonl", rows)
            write_json(output_dir / "metrics.json", metrics)
            run.event("evaluation_saved", output_dir=str(output_dir), **metrics)
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
