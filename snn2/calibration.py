from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .artifacts import ArtifactLayout, read_json, sha256_file, write_json
from .config import post_finetuning_prefix_enabled, training_prefix_enabled
from .data import CausalLMCollator, tokenize_dataset
from .neurons import gif_high_qmax
from .sites import SITE_COUNT, SITE_TOPOLOGY_VERSION, topology_metadata, validate_site_topology
from .stats import StatisticsStore
from .prefix_cache import install_prefix_kv_forward


def calibration_provenance(cfg: dict[str, Any], layout: ArtifactLayout, *, stage: str) -> dict[str, Any]:
    """Build complete, stage-aware calibration provenance metadata."""
    if stage not in {"ann_training", "vanilla_analysis", "post_finetuning"}:
        raise ValueError(f"Unknown calibration stage: {stage}")
    purpose = {
        "ann_training": "ann_training_calibration",
        "vanilla_analysis": "vanilla_analysis_calibration",
        "post_finetuning": "post_finetuning_conversion_calibration",
    }[stage]
    prefix_enabled = (
        training_prefix_enabled(cfg)
        if stage == "ann_training"
        else (post_finetuning_prefix_enabled(cfg) if stage == "post_finetuning" else False)
    )
    prefix_dir = (
        layout.ann_training_prefix_dir
        if stage == "ann_training" and prefix_enabled
        else (layout.post_finetuning_prefix_dir if stage == "post_finetuning" and prefix_enabled else None)
    )
    prefix_state = prefix_dir / "prefix_state.json" if prefix_dir else None
    if prefix_state and not prefix_state.exists():
        raise FileNotFoundError(f"Prefix state required for {stage} calibration: {prefix_state}")
    prefix_ids = [] if prefix_state is None else [int(value) for value in read_json(prefix_state).get("prefix_token_ids", [])]
    prefix_kv = prefix_dir / "prefixed_key_values.pt" if prefix_dir and prefix_ids else None
    if prefix_kv and not prefix_kv.exists():
        raise FileNotFoundError(f"Non-empty Prefix requires fixed KV cache: {prefix_kv}")
    rotation_path = layout.rotation_dir / "rotation_state.pt" if cfg["rotation"]["enabled"] else None
    if rotation_path and not rotation_path.exists():
        raise FileNotFoundError(f"Rotation state required for rotated calibration: {rotation_path}")
    data_manifest = layout.data_dir / "calibration_manifest.json"
    if not data_manifest.exists():
        raise FileNotFoundError(f"Calibration data manifest is missing: {data_manifest}")
    ann_config = layout.ann_checkpoint_dir / "config.json" if stage == "post_finetuning" else None
    if ann_config and not ann_config.exists():
        raise FileNotFoundError(f"Final ANN config is missing: {ann_config}")
    post = stage == "post_finetuning"
    state_profile = {
        "ann_training": "ann_training_with_common_clip",
        "vanilla_analysis": "analysis_statistics_only",
        "post_finetuning": "snn_conversion_without_common_clip",
    }[stage]
    return {
        "purpose": purpose,
        "analysis_only": stage == "vanilla_analysis",
        "eligible_for_ann_training": stage == "ann_training",
        "eligible_for_conversion": post,
        "post_finetuning_recalibration": post,
        "state_profile": state_profile,
        "common_clip_required": stage == "ann_training",
        "source_model_stage": "original_pretrained_base" if stage == "vanilla_analysis" else ("rotated_fused_base" if stage == "ann_training" else "final_ann_checkpoint"),
        "source_ann_mode": cfg["experiment"]["ann_mode"] if post else None,
        "source_ann_checkpoint": str(layout.ann_checkpoint_dir.resolve()) if post else None,
        "source_ann_config_sha256": sha256_file(ann_config) if ann_config else None,
        "calibration_data_manifest_path": str(data_manifest.resolve()),
        "calibration_data_manifest_sha256": sha256_file(data_manifest),
        "prefix_protocol_enabled": prefix_enabled,
        "prefix_enabled": prefix_enabled,
        "prefix_token_ids": prefix_ids,
        "prefix_kv_present": prefix_kv is not None,
        "prefix_state_path": str(prefix_state.resolve()) if prefix_state else None,
        "prefix_state_sha256": sha256_file(prefix_state) if prefix_state else None,
        "prefix_kv_path": str(prefix_kv.resolve()) if prefix_kv else None,
        "prefix_kv_sha256": sha256_file(prefix_kv) if prefix_kv else None,
        "rotation_enabled": rotation_path is not None,
        "rotation_state_path": str(rotation_path.resolve()) if rotation_path else None,
        "rotation_state_sha256": sha256_file(rotation_path) if rotation_path else None,
        "learning_rate": cfg["training"]["learning_rate"] if post else None,
        "seed": int(cfg["experiment"]["seed"]),
    }


def _group_reduce(vector: torch.Tensor, group_size: int, reduction: str) -> torch.Tensor:
    channels = vector.numel()
    if group_size <= 0 or group_size >= channels:
        group_size = channels
    if channels % group_size != 0:
        raise ValueError(f"channels={channels} must be divisible by group_size={group_size}")
    grouped = vector.reshape(-1, group_size)
    if reduction == "min":
        return grouped.amin(dim=-1)
    if reduction == "max":
        return grouped.amax(dim=-1)
    if reduction == "sum":
        return grouped.sum(dim=-1)
    raise ValueError(reduction)


def _qparams(
    minimum: torch.Tensor, maximum: torch.Tensor, bits: int, *, qmax: int | None = None
):
    qmin = 0
    qmax = 2**bits - 1 if qmax is None else int(qmax)
    scale = ((maximum - minimum) / (qmax - qmin)).clamp_min(1e-8)
    zero = torch.round(qmin - minimum / scale).clamp(qmin, qmax)
    representable_min = (qmin - zero) * scale
    representable_max = (qmax - zero) * scale
    return scale, zero, representable_min, representable_max


def build_site_states(
    statistics: dict[str, Any],
    cfg: dict[str, Any],
    *,
    include_clip: bool,
) -> dict[str, dict[str, Any]]:
    channels = int(statistics["channels"])
    variable_channels = bool(statistics.get("variable_channels", False))
    configured_group = int(cfg["calibration"].get("group_size", -1))
    if variable_channels and configured_group > 0:
        raise ValueError("Variable-length Softmax site 5 only supports the default single group")
    reduction_group_size = channels if configured_group <= 0 else configured_group
    runtime_group_size = -1 if configured_group <= 0 else configured_group
    minimum = _group_reduce(statistics["value_min"].double(), reduction_group_size, "min")
    maximum = _group_reduce(statistics["value_max"].double(), reduction_group_size, "max")
    absolute = torch.maximum(minimum.abs(), maximum.abs()).clamp_min(1e-8)
    saliency_counts = statistics.get("saliency_row_count")
    if saliency_counts is None or "saliency_sum" not in statistics:
        raise ValueError("Operator-aware GIF saliency statistics are missing for this site")
    saliency_counts = saliency_counts.long()
    observed = saliency_counts > 0
    if not torch.any(observed):
        raise ValueError("No operator-aware GIF saliency positions were observed")
    operator_saliency = torch.zeros(channels, dtype=torch.float64)
    operator_saliency[observed] = (
        statistics["saliency_sum"].double()[observed] / saliency_counts[observed]
    )

    phase_cfg = cfg["phase"]
    phase_tau = absolute.float()
    phase_state = {
        "format_version": 1,
        "T": int(phase_cfg["T"]),
        "base": float(phase_cfg["base"]),
        "surrogate_slope": float(phase_cfg["surrogate_slope"]),
        "max_spikes": int(phase_cfg.get("max_spikes", 2)),
        "group_size": runtime_group_size,
        "tau": phase_tau,
        "v0": (0.5 * phase_tau * 2 ** (-int(phase_cfg["T"]))).float(),
    }

    gif_cfg = cfg["gif"]
    base_bits = int(gif_cfg["base_bits"])
    add_bits = int(gif_cfg["add_bits"])
    high_bits = base_bits + add_bits
    high_qmax = gif_high_qmax(base_bits, add_bits)
    low_scale, low_zero, low_min, low_max = _qparams(minimum, maximum, base_bits)
    high_scale, high_zero, high_min, high_max = _qparams(
        minimum, maximum, high_bits, qmax=high_qmax
    )
    low_ratio = float(gif_cfg["low_ratio"])
    observed_indices = torch.nonzero(observed, as_tuple=False).flatten()
    low_channels = int(math.floor(low_ratio * observed_indices.numel()))
    ordering = observed_indices[
        torch.argsort(operator_saliency[observed_indices], descending=False)
    ]
    mask_low = torch.ones(channels, dtype=torch.bool) if variable_channels else torch.zeros(channels, dtype=torch.bool)
    if not variable_channels:
        mask_low[observed_indices] = False
    mask_low[ordering[:low_channels]] = True
    gif_state = {
        "format_version": 1,
        "base_bits": base_bits,
        "add_bits": int(gif_cfg["add_bits"]),
        "high_qmax": high_qmax,
        "low_ratio": low_ratio,
        "group_size": runtime_group_size,
        "low_scale": low_scale.float(),
        "low_zero": low_zero.float(),
        "high_scale": high_scale.float(),
        "high_zero": high_zero.float(),
        "mask_low": mask_low,
        "saliency_rule": "operator_aware_spikellm_extension",
        "saliency_score": operator_saliency.float(),
        "saliency_observed": observed,
        "variable_key_position_mask": variable_channels,
        "unobserved_position_policy": "low_bit" if variable_channels else "not_applicable",
        "integer_decomposition": "unsigned_q_capped_to_temporal_capacity_then_subtract_zero_once",
        "scale_initialization": "direct_min_max",
        "mse_refinement": False,
        "original_spikellm_dynamic_quantization": False,
    }

    mtn_cfg = cfg["mtn"]
    mtn_state = {
        "format_version": 1,
        "T": int(mtn_cfg["T"]),
        "K": int(mtn_cfg["K"]),
        "group_size": runtime_group_size,
        "base_scale": (2.0 * absolute).float(),
        "threshold_factor": float(mtn_cfg.get("threshold_factor", 0.75)),
    }
    states = {"phase": phase_state, "gif": gif_state, "mtn": mtn_state}
    if not include_clip:
        return states

    phase_bound = phase_tau.double() * (1.0 - 2.0 ** (-int(phase_cfg["T"])))
    mtn_bound = 2.0 * int(mtn_cfg["T"]) * absolute
    gif_lower = torch.maximum(low_min, high_min)
    gif_upper = torch.minimum(low_max, high_max)
    lower = torch.maximum(torch.maximum(-phase_bound, -mtn_bound), gif_lower)
    upper = torch.minimum(torch.minimum(phase_bound, mtn_bound), gif_upper)
    if torch.any(lower >= upper):
        bad = torch.nonzero(lower >= upper, as_tuple=False).flatten().tolist()
        raise ValueError(f"Invalid common clipping interval in groups {bad}")
    clip_state = {
        "format_version": 1,
        "group_size": runtime_group_size,
        "lower": lower.float(),
        "upper": upper.float(),
        "gif_low_range": (low_min.float(), low_max.float()),
        "gif_high_range": (high_min.float(), high_max.float()),
        "rule": "intersection(phase, mtn, intersection(gif_low, gif_high))",
    }
    states["clip"] = clip_state
    return states


def materialize_calibration_states(
    site_root: str | Path,
    cfg: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    *,
    include_clip: bool,
) -> dict[str, Any]:
    root = Path(site_root)
    manifest: dict[str, Any] = {"format_version": 1, **topology_metadata(), "sites": {}, **(metadata or {})}
    for statistics_path in sorted(root.glob("layer_*/site_*/statistics.pt")):
        directory = statistics_path.parent
        key = directory.relative_to(root).as_posix()
        statistics = torch.load(statistics_path, map_location="cpu", weights_only=False)
        states = build_site_states(statistics, cfg, include_clip=include_clip)
        for name, state in states.items():
            torch.save(state, directory / f"{name}_state.pt")
        if not include_clip:
            (directory / "clip_state.pt").unlink(missing_ok=True)
        summary = {
            "phase_T": states["phase"]["T"],
            "phase_base": states["phase"]["base"],
            "mtn_T": states["mtn"]["T"],
            "mtn_K_positive_and_negative": states["mtn"]["K"],
            "gif_base_bits": states["gif"]["base_bits"],
            "gif_add_bits": states["gif"]["add_bits"],
            "gif_low_ratio": states["gif"]["low_ratio"],
            "group_size": states["phase"]["group_size"],
            "clip_state_present": include_clip,
        }
        if include_clip:
            summary["clip_valid"] = bool(
                torch.all(states["clip"]["lower"] < states["clip"]["upper"])
            )
        write_json(directory / "calibration_summary.json", summary)
        manifest["sites"][key] = summary
    expected = int(cfg["calibration"]["expected_sites_per_layer"])
    if expected != SITE_COUNT:
        raise ValueError(
            "calibration.expected_sites_per_layer must match "
            f"the code topology: config={expected}, code={SITE_COUNT}"
        )
    site_sets = validate_site_topology(root)
    manifest["layer_site_counts"] = {layer: len(sites) for layer, sites in site_sets.items()}
    write_json(root / "calibration_state_manifest.json", manifest)
    return manifest


@torch.no_grad()
def collect_site_statistics(
    model: torch.nn.Module,
    controller: Any,
    tokenizer: Any,
    calibration_raw: Any,
    cfg: dict[str, Any],
    prefix_key_values: Any,
    site_root: str | Path,
    *,
    purpose: str,
    materialize_states: bool = True,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(site_root)
    existing_layers = [path for path in root.glob("layer_*") if path.is_dir()]
    if existing_layers:
        validate_site_topology(root)
        for manifest_name in ("statistics_manifest.json", "calibration_state_manifest.json"):
            manifest_path = root / manifest_name
            if not manifest_path.exists():
                continue
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                metadata.get("site_topology_version") != SITE_TOPOLOGY_VERSION
                or metadata.get("site_count") != SITE_COUNT
            ):
                raise RuntimeError(
                    "Existing calibration artifact uses a stale site topology; "
                    "remove or move the old sites/ directory before recalibrating."
                )
    dataset = tokenize_dataset(calibration_raw, tokenizer, cfg, prefix_ids=None)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["calibration"].get("batch_size", 1)),
        shuffle=False,
        collate_fn=CausalLMCollator(tokenizer),
    )
    controller.mode = "collect"
    controller.statistics = StatisticsStore(
        max_channels_by_site={5: int(cfg["data"]["max_seq_length"])}
    )
    install_prefix_kv_forward(model, prefix_key_values)
    model.eval()
    device = next(model.parameters()).device
    for batch in loader:
        model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            use_cache=False,
        )
    stats_manifest = controller.statistics.reduce_and_save(site_root)
    eligible_ann = purpose == "ann_training_calibration"
    eligible_conversion = purpose == "post_finetuning_conversion_calibration"
    state_profile = {
        "ann_training_calibration": "ann_training_with_common_clip",
        "vanilla_analysis_calibration": "analysis_statistics_only",
        "post_finetuning_conversion_calibration": "snn_conversion_without_common_clip",
    }[purpose]
    metadata = {
        "purpose": purpose,
        "analysis_only": purpose == "vanilla_analysis_calibration",
        "eligible_for_ann_training": eligible_ann,
        "eligible_for_conversion": eligible_conversion,
        "post_finetuning_recalibration": eligible_conversion,
        "state_profile": state_profile,
        "common_clip_required": eligible_ann,
        "source_model_stage": None,
        "source_ann_mode": None,
        "source_ann_checkpoint": None,
        "source_ann_config_sha256": None,
        "calibration_data_manifest_path": None,
        "calibration_data_manifest_sha256": None,
        "prefix_protocol_enabled": False,
        "prefix_enabled": False,
        "prefix_token_ids": [],
        "prefix_kv_present": False,
        "prefix_state_path": None,
        "prefix_state_sha256": None,
        "prefix_kv_path": None,
        "prefix_kv_sha256": None,
        "rotation_enabled": False,
        "rotation_state_path": None,
        "rotation_state_sha256": None,
        "learning_rate": None,
        "seed": int(cfg["experiment"]["seed"]),
        **(extra_metadata or {}),
    }
    stats_manifest.update(metadata)
    write_json(root / "statistics_manifest.json", stats_manifest)
    state_manifest = (
        materialize_calibration_states(
            site_root,
            cfg,
            metadata,
            include_clip=eligible_ann,
        )
        if materialize_states
        else {"format_version": 1, **topology_metadata(), **metadata, "sites": {}}
    )
    if not materialize_states:
        write_json(root / "calibration_state_manifest.json", state_manifest)
    return {"statistics": stats_manifest, "states": state_manifest}
