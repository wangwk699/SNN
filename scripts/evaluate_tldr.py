from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch

from _common import parser, setup
from tqdm.auto import tqdm
from snn2.artifacts import prefix_enabled_dirname, read_json, write_json
from snn2.controller import SiteController
from snn2.conversion import validate_conversion_metadata
from snn2.config import (
    evaluation_prefix_enabled,
    final_evaluation_prefix_artifact_stage,
    rotated_pre_finetuning_prefix_enabled,
)
from snn2.data import (
    encode_tldr_generation_prompt,
    load_selected_raw,
    tldr_prompt_and_reference,
)
from snn2.evaluation import (
    activation_neuron_operators_per_temporal_forward,
    greedy_generate,
    deployment_policy_metadata,
    evaluation_calibration_metadata,
    evaluation_ann_common_clip_enabled,
    resolve_tldr_evaluation_layout,
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
from snn2.prefix_cache import install_prefix_kv_forward


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _validate_rotated_pre_finetuning_dependencies(cfg, layout) -> None:
    if not bool(cfg["rotation"]["enabled"]):
        raise ValueError("--rotated-pre-finetuning requires rotation.enabled=true")

    fused_config = layout.rotation_dir / "fused_base" / "config.json"
    rotation_state_path = layout.rotation_dir / "rotation_state.pt"
    missing = [
        path for path in (fused_config, rotation_state_path) if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Rotated pre-finetuning evaluation requires rotation artifacts. "
            f"Run `python scripts/prepare_rotation.py --config ...` first. Missing: {', '.join(str(path) for path in missing)}"
        )

    if not rotated_pre_finetuning_prefix_enabled(cfg):
        return

    prefix_state_path = layout.rotated_pre_finetuning_prefix_dir / "prefix_state.json"
    if not prefix_state_path.exists():
        raise FileNotFoundError(
            "Rotated pre-finetuning Prefix is enabled but its state is missing. "
            "Run `python scripts/discover_prefix.py --config ... "
            "--stage pre_finetuning`."
        )

    state = read_json(prefix_state_path)
    if state.get("prefix_token_ids", []) and not (
        layout.rotated_pre_finetuning_prefix_dir / "prefixed_key_values.pt"
    ).exists():
        raise FileNotFoundError(
            "Rotated pre-finetuning Prefix is non-empty but its fixed KV cache is missing. "
            "Run `python scripts/discover_prefix.py --config ... "
            "--stage pre_finetuning`."
        )


def main():
    eval_parser = parser(
        "Evaluate a Base model, ANN checkpoint, or converted SNN on Reddit TL;DR",
        neuron=True,
        allow_ann=True,
    )

    model_variant_group = eval_parser.add_mutually_exclusive_group()
    model_variant_group.add_argument(
        "--base",
        action="store_true",
        help=(
            "Evaluate the original pretrained Base model instead "
            "of the fine-tuned ANN checkpoint"
        ),
    )
    model_variant_group.add_argument(
        "--rotated-pre-finetuning",
        action="store_true",
        help="Evaluate the rotated fused Base checkpoint before ANN fine-tuning",
    )

    args = eval_parser.parse_args()

    # 先读取 config，之后才能访问 cfg
    cfg, layout = setup(
        args.config,
        config_scope=(
            "base"
            if args.base
            else (
                "rotated_pre_finetuning"
                if args.rotated_pre_finetuning
                else "run"
            )
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

    if args.neuron != "ann" and not args.base and not args.rotated_pre_finetuning:
        validate_conversion_metadata(cfg, layout, args.neuron)

    if args.rotated_pre_finetuning:
        if args.neuron != "ann":
            raise ValueError("--rotated-pre-finetuning can only be used with --neuron ann")
        _validate_rotated_pre_finetuning_dependencies(cfg, layout)

    rotated_prefix_enabled = (
        rotated_pre_finetuning_prefix_enabled(cfg)
        if args.rotated_pre_finetuning
        else None
    )
    final_evaluation_prefix_enabled = (
        evaluation_prefix_enabled(cfg)
        if not args.base and not args.rotated_pre_finetuning
        else None
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
        else (
            "rotated_pre_finetuning_ann"
            if args.rotated_pre_finetuning
            else args.neuron
        )
    )

    stage = (
        f"evaluate_tldr_"
        f"{model_variant}_rank{rank}"
    )

    logs_dir = (
        layout.base_logs_dir
        if args.base
        else (
            layout.rotated_pre_finetuning_logs_dir
            if args.rotated_pre_finetuning
            else layout.logs_dir
        )
    )

    with StageRun(stage, logs_dir, cfg["experiment"]) as run:
        from accelerate import Accelerator
        from accelerate.utils import gather_object
        import evaluate

        accelerator = Accelerator()

        if args.base:
            checkpoint_stage = "base_evaluation"
        elif args.rotated_pre_finetuning:
            checkpoint_stage = "rotated_pre_finetuning"
        else:
            checkpoint_stage = "post_finetuning"
        source = model_source_for_stage(cfg, layout, stage=checkpoint_stage)

        model = load_model(cfg, source, training=False)
        model.to(accelerator.device)
        model.eval()
        tokenizer = load_tokenizer(cfg, source)
        tokenizer.padding_side = "left"
        prefix_stage = (
            "base_evaluation"
            if args.base
            else (
                "rotated_pre_finetuning"
                if args.rotated_pre_finetuning
                else "final_evaluation"
            )
        )
        controller = SiteController(
            mode="identity",
            site_root=(
                layout.conversion_site_dir
                if not args.base and not args.rotated_pre_finetuning
                else None
            ),
        )
        steps = 1 if args.neuron == "ann" else controller.set_deployment(args.neuron)
        if args.neuron != "ann" or cfg["rotation"]["enabled"]:
            install_model_integration(model, controller, rotation_state(cfg, layout))
        prefixes = prefix_ids_for_stage(cfg, layout, stage=prefix_stage)
        install_prefix_kv_forward(
            model,
            prefix_key_values_for_stage(cfg, layout, stage=prefix_stage),
            controller=controller,
        )

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

        selection_layout = resolve_tldr_evaluation_layout(
            total_test_samples,
            configured_test_samples,
        )
        selected_test_samples = int(selection_layout["selected_test_samples"])
        is_full_test = bool(selection_layout["is_full_test"])
        test_samples_dirname = str(selection_layout["dirname"])
        if is_full_test:
            selected_indices = list(range(total_test_samples))
            sampling_method = "full_split"
        else:
            rng = random.Random(tldr_test_seed)
            selected_indices = rng.sample(range(total_test_samples), k=selected_test_samples)
            selected_indices.sort()
            sampling_method = "seeded_random_without_replacement"


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
                prompt, reference = tldr_prompt_and_reference(row)

                prompt_ids = encode_tldr_generation_prompt(row, tokenizer, cfg)

                batch_input_ids.append(prompt_ids)
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
            # × SITE_COUNT activation replacement sites
            # ----------------------------------------
            per_forward_operators = activation_neuron_operators_per_temporal_forward(
                num_hidden_layers=layers, neuron=args.neuron
            )
            activation_site_temporal_operator_calls = (
                temporal_sample_step_forwards * per_forward_operators
            )

            # ----------------------------------------
            # Actual batched execution slot count
            #
            # 这里包含 batch 中已经 EOS、
            # 但因其它样本仍在生成而继续占据的 tensor slot。
            # ----------------------------------------
            batched_activation_site_temporal_slots = (
                batched_temporal_sample_slots * per_forward_operators
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
                    "decode": "greedy",
                    "input_length": input_length,
                    "max_new_tokens": max_new,
                    "checkpoint_stage": checkpoint_stage,
                    "prefix_stage": prefix_stage,
                    "prefix_root": (
                        None
                        if args.base or (args.rotated_pre_finetuning and not rotated_prefix_enabled) or (not args.rotated_pre_finetuning and not final_evaluation_prefix_enabled)
                        else str(
                            layout.rotated_pre_finetuning_prefix_dir
                            if args.rotated_pre_finetuning
                            else layout.conversion_prefix_dir
                        )
                    ),
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
                    "rotation_enabled": bool(cfg["rotation"]["enabled"]),

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

            if not args.base:
                metrics["prefix_enabled"] = (
                    rotated_prefix_enabled
                    if args.rotated_pre_finetuning
                    else final_evaluation_prefix_enabled
                )

            if args.base:
                model_output_dir = layout.base_dir
            elif args.rotated_pre_finetuning:
                model_output_dir = layout.rotated_pre_finetuning_dir
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
            if not args.base:
                output_dir = output_dir / prefix_enabled_dirname(
                    bool(
                        rotated_prefix_enabled
                        if args.rotated_pre_finetuning
                        else final_evaluation_prefix_enabled
                    )
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
