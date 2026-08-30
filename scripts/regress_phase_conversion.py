from __future__ import annotations

import gc
import json
from pathlib import Path

import torch

from _common import parser, setup
from snn2.artifacts import write_json
from snn2.data import encode_tldr_generation_prompt, load_selected_raw
from snn2.evaluation import position_ids_from_attention_mask
from snn2.model_integration import install_model_integration, temporal_forward
from snn2.modeling import (
    load_model,
    load_tokenizer,
    prefix_key_values_for_stage,
    rotation_state,
)
from snn2.phase_conversion_regression import (
    PhaseConversionRegressionRecorder,
    logits_metrics,
    run_phase_neuron_micro_regression,
    run_temporal_primitive_regression,
    summarize_first_divergence,
    validate_phase_conversion_artifacts,
)
from snn2.prefix_cache import install_prefix_kv_forward
from snn2.controller import SiteController
from snn2.config import conversion_reuses_ann_training_artifacts


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _release_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _controller(cfg, layout, graph: str, *, bypass_final_norm_phase: bool):
    if graph == "identity":
        controller = SiteController(mode="identity", site_root=layout.ann_training_site_dir)
    elif graph == "phase_static":
        controller = SiteController(
            mode="phase",
            site_root=layout.ann_training_site_dir,
            common_clip_enabled=False,
            phase_T=int(cfg["phase"]["T"]),
            phase_surrogate_slope=float(cfg["phase"]["surrogate_slope"]),
        )
    elif graph == "phase_temporal":
        controller = SiteController(
            mode="identity", site_root=layout.ann_training_site_dir,
            phase_T=int(cfg["phase"]["T"]), mtn_T=int(cfg["mtn"]["T"]),
            mtn_K=int(cfg["mtn"]["K"]),
            mtn_threshold_factor=float(cfg["mtn"]["threshold_factor"]),
        )
        controller.set_deployment(
            "phase",
            clip_bundle_policy="forbid_all",
        )
        controller.regression_bypass_final_norm_phase = bool(bypass_final_norm_phase)
    else:
        raise ValueError(graph)
    return controller


def _load_graph(cfg, layout, graph: str, device: torch.device, *, trace: bool, bypass: bool):
    model = load_model(cfg, str(layout.ann_checkpoint_dir), training=False)
    model.to(device)
    model.eval()
    controller = _controller(cfg, layout, graph, bypass_final_norm_phase=bypass)
    recorder = None
    if trace:
        recorder = PhaseConversionRegressionRecorder(
            graph,
            controller.temporal_steps if graph == "phase_temporal" else None,
        )
        controller.set_regression_recorder(recorder)
    install_model_integration(model, controller, rotation_state(cfg, layout))
    install_prefix_kv_forward(
        model,
        prefix_key_values_for_stage(cfg, layout, stage="ann_training"),
        controller=controller,
    )
    return model, controller, recorder


@torch.no_grad()
def _forward(model, controller, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    position_ids = position_ids_from_attention_mask(attention_mask)
    if controller.mode.startswith("deploy_"):
        return temporal_forward(
            model,
            controller,
            input_ids,
            attention_mask,
            position_ids=position_ids,
        )
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    ).logits


def _run_fixed_graph(
    cfg,
    layout,
    graph: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    *,
    trace: bool,
    bypass: bool = False,
):
    model, controller, recorder = _load_graph(
        cfg, layout, graph, device, trace=trace, bypass=bypass
    )
    logits = _forward(
        model, controller, input_ids.to(device), attention_mask.to(device)
    ).detach().float().cpu()
    _release_model(model)
    return logits, recorder


def _locked_static_oracle(cfg, layout, ids, mask, device, decode_steps: int):
    model, controller, _ = _load_graph(
        cfg, layout, "phase_static", device, trace=False, bypass=False
    )
    history = ids.to(device)
    history_mask = mask.to(device)
    logits_rows = []
    tokens = []
    with torch.no_grad():
        for _ in range(decode_steps):
            last = _forward(model, controller, history, history_mask)[:, -1].float().cpu()
            token = int(last[0].argmax())
            logits_rows.append(last)
            tokens.append(token)
            next_token = torch.tensor([[token]], device=device, dtype=history.dtype)
            history = torch.cat((history, next_token), dim=-1)
            history_mask = torch.cat((history_mask, torch.ones_like(next_token)), dim=-1)
    _release_model(model)
    return logits_rows, tokens


def _locked_temporal_compare(cfg, layout, ids, mask, device, oracle_logits, tokens):
    model, controller, _ = _load_graph(
        cfg, layout, "phase_temporal", device, trace=False, bypass=False
    )
    history = ids.to(device)
    history_mask = mask.to(device)
    rows = []
    with torch.no_grad():
        for step, (reference, token) in enumerate(zip(oracle_logits, tokens)):
            last = _forward(model, controller, history, history_mask)[:, -1].float().cpu()
            metric = logits_metrics(reference.unsqueeze(1), last.unsqueeze(1), name=f"locked_step_{step:03d}")
            rows.append(
                {
                    "step": step,
                    "context_length": int(history.shape[-1]),
                    "logits_relative_l2": metric["relative_l2_error"],
                    "mean_abs": metric["mean_abs_error"],
                    "max_abs": metric["max_abs_error"],
                    "top1_equal": metric["last_token_top1_equal"],
                    "static_top1_id": metric["last_token_top1_id_ref"],
                    "snn_top1_id": metric["last_token_top1_id_test"],
                }
            )
            next_token = torch.tensor([[token]], device=device, dtype=history.dtype)
            history = torch.cat((history, next_token), dim=-1)
            history_mask = torch.cat((history_mask, torch.ones_like(next_token)), dim=-1)
    _release_model(model)
    return rows


def main() -> None:
    regression_parser = parser("Regress static Phase ANN against temporal Phase SNN")
    regression_parser.add_argument("--sample-index", type=int, default=0)
    regression_parser.add_argument("--max-input-tokens", type=int, default=64)
    regression_parser.add_argument("--decode-steps", type=int, default=16)
    regression_parser.add_argument("--dump-first-failure-tensor", action="store_true")
    regression_parser.add_argument("--skip-locked-decode", action="store_true")
    args = regression_parser.parse_args()
    if args.max_input_tokens <= 0 or args.decode_steps < 0:
        raise ValueError("max-input-tokens must be positive and decode-steps non-negative")

    cfg, layout = setup(args.config)
    output_dir = layout.root / "analysis" / "phase_conversion_regression"
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_phase_conversion_artifacts(cfg, layout)
    write_json(output_dir / "artifact_validation.json", validation)

    tokenizer = load_tokenizer(cfg, str(layout.ann_checkpoint_dir))
    tokenizer.padding_side = "left"
    bundle = load_selected_raw(cfg, layout)
    if bundle.evaluation is None:
        raise FileNotFoundError("TL;DR evaluation manifest/test split is missing")
    if not 0 <= args.sample_index < len(bundle.evaluation):
        raise IndexError(args.sample_index)
    row = bundle.evaluation[args.sample_index]
    prompt_ids = encode_tldr_generation_prompt(row, tokenizer, cfg)
    prompt_ids = prompt_ids[-args.max_input_tokens :]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    record_ids = bundle.manifests["evaluation"].get("record_ids", [])
    record_id = record_ids[args.sample_index] if args.sample_index < len(record_ids) else None
    metadata = {
        "sample_index": args.sample_index,
        "dataset_record_id": record_id,
        "input_ids": prompt_ids,
        "attention_mask": attention_mask.tolist(),
        "prefix_token_ids": validation["prefix_token_ids"],
        "max_input_tokens": args.max_input_tokens,
        "decode_steps": args.decode_steps,
        "batch_size": 1,
        "use_cache": False,
        "dropout": 0,
        "source_ann_checkpoint": str(layout.ann_checkpoint_dir.resolve()),
    }
    write_json(output_dir / "regression_metadata.json", metadata)

    num_layers = int(validation["conversion"]["expected_num_hidden_layers"])
    micro = run_phase_neuron_micro_regression(layout.ann_training_site_dir, num_layers, phase_T=int(cfg["phase"]["T"]))
    primitives = run_temporal_primitive_regression(
        steps=int(validation["conversion"]["full_temporal_steps"])
    )
    write_json(output_dir / "micro_phase_neuron.json", micro)
    write_json(output_dir / "temporal_primitives.json", primitives)
    if not micro["passed"] or not primitives["passed"]:
        raise RuntimeError("Micro Phase or FP32 temporal primitive regression failed")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    identity, _ = _run_fixed_graph(
        cfg, layout, "identity", input_ids, attention_mask, device, trace=False
    )
    phase_static, static_recorder = _run_fixed_graph(
        cfg, layout, "phase_static", input_ids, attention_mask, device, trace=True
    )
    phase_temporal, temporal_recorder = _run_fixed_graph(
        cfg, layout, "phase_temporal", input_ids, attention_mask, device, trace=True
    )
    phase_temporal_bypass, bypass_recorder = _run_fixed_graph(
        cfg,
        layout,
        "phase_temporal",
        input_ids,
        attention_mask,
        device,
        trace=True,
        bypass=True,
    )
    summary = {
        "identity_vs_static_phase": logits_metrics(identity, phase_static, name="I_vs_P"),
        "static_phase_vs_temporal_phase": logits_metrics(phase_static, phase_temporal, name="P_vs_S"),
        "identity_vs_temporal_phase": logits_metrics(identity, phase_temporal, name="I_vs_S"),
        "static_phase_vs_temporal_bypass": logits_metrics(phase_static, phase_temporal_bypass, name="P_vs_S0"),
        "temporal_phase_vs_temporal_bypass": logits_metrics(phase_temporal, phase_temporal_bypass, name="S_vs_S0"),
    }
    summary["conversion_passed"] = bool(
        summary["static_phase_vs_temporal_phase"]["relative_l2_error"] <= 1e-2
        and summary["static_phase_vs_temporal_phase"]["last_token_top1_equal"]
    )
    write_json(output_dir / "fixed_forward_summary.json", summary)

    checkpoint_rows = static_recorder.compare(temporal_recorder)
    bypass_rows = static_recorder.compare(bypass_recorder)
    _write_jsonl(output_dir / "checkpoint_metrics.jsonl", checkpoint_rows)
    first_divergence = summarize_first_divergence(checkpoint_rows)
    write_json(output_dir / "first_divergence.json", first_divergence)
    write_json(
        output_dir / "final_norm_ablation.json",
        {
            "topology_mismatch_candidate": True,
            "static_phase_vs_temporal_phase": summary["static_phase_vs_temporal_phase"],
            "static_phase_vs_temporal_bypass": summary["static_phase_vs_temporal_bypass"],
            "temporal_phase_vs_temporal_bypass": summary["temporal_phase_vs_temporal_bypass"],
            "bypass_first_divergence": summarize_first_divergence(bypass_rows),
        },
    )
    if args.dump_first_failure_tensor and first_divergence["first_relative_l2_gt_1e-2"]:
        name = first_divergence["first_relative_l2_gt_1e-2"]["name"]
        torch.save(static_recorder.tensors[name], output_dir / "first_failure_reference.pt")
        torch.save(temporal_recorder.tensors[name], output_dir / "first_failure_temporal_sum.pt")

    locked_rows = []
    if not args.skip_locked_decode and args.decode_steps:
        oracle_logits, tokens = _locked_static_oracle(
            cfg, layout, input_ids, attention_mask, device, args.decode_steps
        )
        locked_rows = _locked_temporal_compare(
            cfg, layout, input_ids, attention_mask, device, oracle_logits, tokens
        )
    _write_jsonl(output_dir / "locked_decode.jsonl", locked_rows)
    locked_summary = {
        "steps": len(locked_rows),
        "first_top1_disagreement_step": next(
            (item["step"] for item in locked_rows if not item["top1_equal"]), None
        ),
        "first_logits_relative_l2_gt_1e-2_step": next(
            (item["step"] for item in locked_rows if item["logits_relative_l2"] > 1e-2), None
        ),
    }
    write_json(output_dir / "locked_decode_summary.json", locked_summary)
    print(json.dumps({"output_dir": str(output_dir), **summary, "locked_decode": locked_summary}, indent=2))


if __name__ == "__main__":
    main()
