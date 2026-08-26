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
from .sites import (
    CLIP_ELIGIBLE_SITE_IDS,
    SOFTMAX_SITE_ID,
    SITE_COUNT,
    SITE_TOPOLOGY_VERSION,
    is_softmax_site,
    site_supports_clip,
    topology_metadata,
    validate_site_topology,
)
from .stats import StatisticsStore
from .temporal_ops import (
    CALIBRATION_MANIFEST_FORMAT_VERSION,
    GIF_ADD_BITS,
    GIF_BASE_BITS,
    GIF_HIGH_QMAX,
    GIF_INTEGER_DECOMPOSITION,
    GIF_LOCAL_STEPS,
    GIF_LOW_QMAX,
    GIF_STEP_QMAX,
    SITE_STATE_FORMAT_VERSION,
    STATISTICS_FORMAT_VERSION,
    TEMPORAL_IMPLEMENTATION_VERSION,
    CALIBRATION_GROUPING_POLICY,
    SOFTMAX_SITE5_CLIP_POLICY,
    SOFTMAX_SITE5_GIF_POLICY,
    SOFTMAX_SITE5_GROUPING_POLICY,
    temporal_policy_metadata,
)
from .prefix_cache import install_prefix_kv_forward
from .prefix_cache import prefix_length
from .phase_statistics import (
    PHASE_TAU_ACCUMULATOR_DTYPE,
    PHASE_TAU_CALIBRATION,
    PHASE_TAU_CHANNEL_POLICY,
    PHASE_TAU_EMA_FACTOR,
    PHASE_TAU_REDUCTION_POLICY,
)
from .rotation import get_model_parts


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
        "post_finetuning": "snn_conversion_without_clip",
    }[stage]
    return {
        "purpose": purpose,
        "analysis_only": stage == "vanilla_analysis",
        "eligible_for_ann_training": stage == "ann_training",
        "eligible_for_conversion": stage in {"ann_training", "post_finetuning"},
        "conversion_reuse_policy": (
            "aware_modes_only" if stage == "ann_training" else
            ("final_ann_only" if stage == "post_finetuning" else "none")
        ),
        "post_finetuning_recalibration": post,
        "state_profile": state_profile,
        "common_clip_required": stage == "ann_training",
        "common_clip_generated": stage == "ann_training",
        "common_clip_application_control": "replacement.common_clip_enabled",
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
        "calibration_group_size": int(cfg["calibration"]["group_size"]),
        "calibration_grouping_policy": CALIBRATION_GROUPING_POLICY,
        "statistics_format_version": STATISTICS_FORMAT_VERSION,
        "softmax_site5_grouping_policy": SOFTMAX_SITE5_GROUPING_POLICY,
        "softmax_site5_gif_policy": SOFTMAX_SITE5_GIF_POLICY,
        "softmax_site5_clip_policy": SOFTMAX_SITE5_CLIP_POLICY,
        "clip_eligible_site_ids": sorted(CLIP_ELIGIBLE_SITE_IDS),
        "clip_excluded_site_ids": [SOFTMAX_SITE_ID],
    }


def group_reduce_last_dim(values: torch.Tensor, group_size: int, reduction: str) -> torch.Tensor:
    width = int(values.shape[-1])
    effective = width if int(group_size) == -1 else int(group_size)
    if effective <= 0 or width % effective != 0:
        raise ValueError(
            f"last dimension {width} must be divisible by configured group_size={group_size}"
        )
    grouped = values.reshape(*values.shape[:-1], width // effective, effective)
    if reduction == "min":
        return grouped.amin(dim=-1)
    if reduction == "max":
        return grouped.amax(dim=-1)
    if reduction == "sum":
        return grouped.sum(dim=-1)
    raise ValueError(reduction)


def _qparams(
    minimum: torch.Tensor,
    maximum: torch.Tensor,
    *,
    qmin: int,
    qmax: int,
):
    qmin, qmax = int(qmin), int(qmax)
    if qmin != 0 or qmax <= qmin:
        raise ValueError(f"Invalid unsigned quantization range [{qmin}, {qmax}]")
    scale = ((maximum - minimum) / (qmax - qmin)).clamp_min(1e-8)
    zero = torch.round(qmin - minimum / scale).clamp(qmin, qmax)
    representable_min = (qmin - zero) * scale
    representable_max = (qmax - zero) * scale
    return scale, zero, representable_min, representable_max


def _validate_statistics(statistics: dict[str, Any]) -> None:
    if statistics.get("format_version") != STATISTICS_FORMAT_VERSION:
        raise ValueError(
            f"Legacy statistics schema; format_version={STATISTICS_FORMAT_VERSION} is required"
        )
    if statistics.get("phase_tau_calibration") != PHASE_TAU_CALIBRATION:
        raise ValueError("Phase statistics use an incompatible calibration policy")
    if float(statistics.get("phase_tau_ema_factor", -1.0)) != PHASE_TAU_EMA_FACTOR:
        raise ValueError("Phase statistics must use EMA factor 0.99")
    if statistics.get("phase_tau_accumulator_dtype") != PHASE_TAU_ACCUMULATOR_DTYPE:
        raise ValueError("Phase statistics must use an FP32 accumulator")
    if statistics.get("phase_tau_channel_policy") != PHASE_TAU_CHANNEL_POLICY:
        raise ValueError("Phase statistics use an incompatible channel policy")
    if statistics.get("phase_tau_reduction_policy") != PHASE_TAU_REDUCTION_POLICY:
        raise ValueError("Phase statistics use an incompatible reduction policy")


def _layout_metadata(statistics: dict[str, Any], configured_group: int) -> dict[str, Any]:
    layout = statistics["layout_kind"]
    heads = statistics.get("num_heads")
    width = statistics.get("channels_per_head")
    if layout == "last_dim":
        width = int(statistics["channels"])
        effective = width if configured_group == -1 else configured_group
        parameter_layout = "last_dim_grouped"
        heads = None
    elif layout == "attention_head":
        heads, width = int(heads), int(width)
        effective = width if configured_group == -1 else configured_group
        parameter_layout = "attention_head_grouped"
    elif layout == "attention_softmax":
        heads, width, effective = int(heads), None, -1
        parameter_layout = "attention_head_scalar"
    else:
        raise ValueError(f"Unknown statistics layout_kind={layout!r}")
    if layout != "attention_softmax" and (effective <= 0 or width % effective != 0):
        raise ValueError(
            f"site={statistics.get('site_index')} layout={layout} channel/head_dim={width} "
            f"is not divisible by configured group_size={configured_group}"
        )
    groups = 1 if layout == "attention_softmax" else width // effective
    return {
        "parameter_layout": parameter_layout,
        "configured_group_size": configured_group,
        "group_size": effective,
        "num_heads": heads,
        "channels_per_head": width,
        "groups_per_head": groups,
    }


def build_phase_state(statistics: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    _validate_statistics(statistics)
    phase_stat = statistics.get("phase_ema_abs_max")
    updates = statistics.get("phase_ema_updates")
    if not isinstance(phase_stat, torch.Tensor) or not isinstance(updates, torch.Tensor):
        raise ValueError("Phase EMA statistics are missing")
    if phase_stat.numel() != updates.numel() or not torch.any(updates > 0):
        raise ValueError("Phase EMA statistics have invalid update metadata")
    if phase_stat.dtype != torch.float32:
        raise ValueError("Phase EMA accumulator must use float32")
    configured_group = int(cfg["calibration"]["group_size"])
    layout = _layout_metadata(statistics, configured_group)
    if statistics["layout_kind"] == "attention_softmax":
        tau = phase_stat.float().reshape(int(layout["num_heads"]), 1)
    else:
        tau = group_reduce_last_dim(phase_stat.float(), configured_group, "max")
    phase_cfg = cfg["phase"]
    steps = int(phase_cfg["T"])
    return {
        "state_kind": "phase",
        "format_version": SITE_STATE_FORMAT_VERSION,
        "temporal_implementation_version": TEMPORAL_IMPLEMENTATION_VERSION,
        "T": steps,
        "base": float(phase_cfg["base"]),
        "max_spikes": int(phase_cfg.get("max_spikes", steps)),
        **layout,
        "tau": tau,
        "v0": (0.5 * tau * 2 ** (-steps)).float(),
        "tau_calibration": PHASE_TAU_CALIBRATION,
        "tau_ema_factor": PHASE_TAU_EMA_FACTOR,
        "tau_accumulator_dtype": PHASE_TAU_ACCUMULATOR_DTYPE,
        "tau_channel_policy": PHASE_TAU_CHANNEL_POLICY,
        "tau_reduction_policy": PHASE_TAU_REDUCTION_POLICY,
    }


def build_site_states(
    statistics: dict[str, Any],
    cfg: dict[str, Any],
    *,
    include_clip: bool,
) -> dict[str, dict[str, Any]]:
    _validate_statistics(statistics)
    site_index = statistics.get("site_index")
    if not isinstance(site_index, int):
        raise ValueError("Per-site statistics must contain an integer site_index")
    configured_group = int(cfg["calibration"].get("group_size", -1))
    layout = _layout_metadata(statistics, configured_group)
    is_site5 = is_softmax_site(site_index)
    if is_site5:
        minimum = statistics["value_min"].double().reshape(layout["num_heads"], 1)
        maximum = statistics["value_max"].double().reshape(layout["num_heads"], 1)
    else:
        minimum = group_reduce_last_dim(statistics["value_min"].double(), configured_group, "min")
        maximum = group_reduce_last_dim(statistics["value_max"].double(), configured_group, "max")
    absolute = torch.maximum(minimum.abs(), maximum.abs()).clamp_min(1e-8)

    phase_cfg = cfg["phase"]
    phase_state = build_phase_state(statistics, cfg)
    phase_tau = phase_state["tau"]

    gif_cfg = cfg["gif"]
    if is_site5:
        gif_state = {
            "state_kind": "gif",
            "format_version": SITE_STATE_FORMAT_VERSION,
            "temporal_implementation_version": TEMPORAL_IMPLEMENTATION_VERSION,
            "parameter_layout": "softmax_identity",
            "configured_group_size": configured_group,
            "group_size": -1,
            "group_size_source": "site5_identity_override",
            "num_heads": int(layout["num_heads"]),
            "channels_per_head": None,
            "groups_per_head": 1,
            "gif_policy": SOFTMAX_SITE5_GIF_POLICY,
            "reference_n_bits": 16,
            "reference_metric": "fix0to1",
            "quantization_applied": False,
            "temporal_steps": GIF_LOCAL_STEPS,
            "temporal_policy": "identity",
        }
    else:
        saliency_counts = statistics.get("saliency_row_count")
        if not isinstance(saliency_counts, torch.Tensor) or saliency_counts.shape != statistics["value_min"].shape:
            raise ValueError("Operator-aware GIF saliency statistics are missing or have the wrong shape")
        observed = saliency_counts.long() > 0
        if not torch.all(observed):
            raise ValueError("Every ordinary GIF channel must have operator-aware saliency")
        operator_saliency = statistics["saliency_sum"].double() / saliency_counts.double()
    base_bits = int(gif_cfg["base_bits"])
    add_bits = int(gif_cfg["add_bits"])
    gif_high_qmax(base_bits, add_bits)
    if not is_site5:
        low_scale, low_zero, low_min, low_max = _qparams(minimum, maximum, qmin=0, qmax=GIF_LOW_QMAX)
        high_scale, high_zero, high_min, high_max = _qparams(minimum, maximum, qmin=0, qmax=GIF_HIGH_QMAX)
        low_ratio = float(gif_cfg["low_ratio"])
        mask_low = torch.zeros_like(statistics["value_min"], dtype=torch.bool)
        if statistics["layout_kind"] == "attention_head":
            low_channels = int(math.floor(low_ratio * mask_low.shape[-1]))
            for head in range(mask_low.shape[0]):
                ordering = torch.argsort(operator_saliency[head], descending=False)
                mask_low[head, ordering[:low_channels]] = True
        else:
            low_channels = int(math.floor(low_ratio * mask_low.shape[-1]))
            ordering = torch.argsort(operator_saliency, descending=False)
            mask_low[ordering[:low_channels]] = True
        gif_state = {
        "state_kind": "gif",
        "format_version": SITE_STATE_FORMAT_VERSION,
        "temporal_implementation_version": TEMPORAL_IMPLEMENTATION_VERSION,
        "base_bits": base_bits,
        "add_bits": int(gif_cfg["add_bits"]),
        "low_qmin": 0,
        "low_qmax": GIF_LOW_QMAX,
        "high_qmin": 0,
        "high_qmax": GIF_HIGH_QMAX,
        "temporal_steps": GIF_LOCAL_STEPS,
        "per_step_qmin": 0,
        "per_step_qmax": GIF_STEP_QMAX,
        "low_ratio": low_ratio,
        **layout,
        "gif_policy": "ordinary_grouped_qmax30",
        "low_scale": low_scale.float(),
        "low_zero": low_zero.float(),
        "high_scale": high_scale.float(),
        "high_zero": high_zero.float(),
        "mask_low": mask_low,
        "saliency_rule": "operator_aware_spikellm_extension",
        "saliency_score": operator_saliency.float(),
        "saliency_observed": observed,
        "integer_decomposition": GIF_INTEGER_DECOMPOSITION,
        "scale_initialization": "direct_min_max",
        "mse_refinement": False,
        "original_spikellm_dynamic_quantization": False,
        }

    mtn_cfg = cfg["mtn"]
    mtn_state = {
        "state_kind": "mtn",
        "format_version": SITE_STATE_FORMAT_VERSION,
        "temporal_implementation_version": TEMPORAL_IMPLEMENTATION_VERSION,
        "T": int(mtn_cfg["T"]),
        "K": int(mtn_cfg["K"]),
        **layout,
        "base_scale": (2.0 * absolute).float(),
        "threshold_factor": float(mtn_cfg.get("threshold_factor", 0.75)),
    }
    states = {"phase": phase_state, "gif": gif_state, "mtn": mtn_state}
    if not include_clip or not site_supports_clip(site_index):
        return states

    phase_bound = phase_tau.double() * sum(
        float(phase_cfg["base"]) ** (-(step + 1))
        for step in range(int(phase_cfg["T"]))
    )
    mtn_bound = 2.0 * int(mtn_cfg["T"]) * absolute
    gif_lower = torch.maximum(low_min, high_min)
    gif_upper = torch.minimum(low_max, high_max)
    lower = torch.maximum(torch.maximum(-phase_bound, -mtn_bound), gif_lower)
    upper = torch.minimum(torch.minimum(phase_bound, mtn_bound), gif_upper)
    if torch.any(lower >= upper):
        bad = torch.nonzero(lower >= upper, as_tuple=False).flatten().tolist()
        raise ValueError(f"Invalid common clipping interval in groups {bad}")
    clip_state = {
        "state_kind": "clip",
        "format_version": SITE_STATE_FORMAT_VERSION,
        "temporal_implementation_version": TEMPORAL_IMPLEMENTATION_VERSION,
        "ordinary_gif_high_qmax": GIF_HIGH_QMAX,
        "ordinary_gif_per_step_qmax": GIF_STEP_QMAX,
        "gif_integer_decomposition": GIF_INTEGER_DECOMPOSITION,
        **layout,
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
    expected_num_hidden_layers: int,
) -> dict[str, Any]:
    if (
        not isinstance(expected_num_hidden_layers, int)
        or isinstance(expected_num_hidden_layers, bool)
        or expected_num_hidden_layers <= 0
    ):
        raise ValueError("expected_num_hidden_layers must be a positive integer")
    root = Path(site_root)
    manifest: dict[str, Any] = {
        **(metadata or {}),
        "format_version": CALIBRATION_MANIFEST_FORMAT_VERSION,
        "statistics_format_version": STATISTICS_FORMAT_VERSION,
        "calibration_group_size": int(cfg["calibration"]["group_size"]),
        "calibration_grouping_policy": CALIBRATION_GROUPING_POLICY,
        "softmax_site5_grouping_policy": SOFTMAX_SITE5_GROUPING_POLICY,
        "softmax_site5_gif_policy": SOFTMAX_SITE5_GIF_POLICY,
        "softmax_site5_clip_policy": SOFTMAX_SITE5_CLIP_POLICY,
        "clip_eligible_site_ids": sorted(CLIP_ELIGIBLE_SITE_IDS),
        "clip_excluded_site_ids": [SOFTMAX_SITE_ID],
        **topology_metadata(),
        **temporal_policy_metadata(),
        "expected_num_hidden_layers": expected_num_hidden_layers,
        "expected_layer_names": [
            f"layer_{index:03d}" for index in range(expected_num_hidden_layers)
        ],
        "sites": {},
    }
    for statistics_path in sorted(root.glob("layer_*/site_*/statistics.pt")):
        directory = statistics_path.parent
        key = directory.relative_to(root).as_posix()
        statistics = torch.load(statistics_path, map_location="cpu", weights_only=False)
        _validate_statistics(statistics)
        states = build_site_states(statistics, cfg, include_clip=include_clip)
        for name, state in states.items():
            torch.save(state, directory / f"{name}_state.pt")
        if not include_clip or is_softmax_site(int(statistics["site_index"])):
            (directory / "clip_state.pt").unlink(missing_ok=True)
        gif_state = states["gif"]
        summary = {
            "site_index": int(statistics["site_index"]),
            "layout_kind": statistics["layout_kind"],
            "parameter_layout": states["phase"]["parameter_layout"],
            "configured_group_size": states["phase"]["configured_group_size"],
            "effective_group_size": states["phase"]["group_size"],
            "num_heads": states["phase"]["num_heads"],
            "channels_per_head": states["phase"]["channels_per_head"],
            "groups_per_head": states["phase"]["groups_per_head"],
            "phase_tau_shape": list(states["phase"]["tau"].shape),
            "phase_T": states["phase"]["T"],
            "phase_base": states["phase"]["base"],
            "mtn_T": states["mtn"]["T"],
            "mtn_K_positive_and_negative": states["mtn"]["K"],
            "mtn_parameter_shape": list(states["mtn"]["base_scale"].shape),
            "gif_policy": gif_state["gif_policy"],
            "gif_parameter_shape": (
                None if is_softmax_site(int(statistics["site_index"]))
                else list(gif_state["low_scale"].shape)
            ),
            "gif_temporal_steps": gif_state["temporal_steps"],
            "clip_policy": "eligible_common_intersection" if "clip" in states else "disabled",
            "clip_state_present": "clip" in states,
            "phase_tau_calibration": states["phase"]["tau_calibration"],
            "phase_tau_ema_factor": states["phase"]["tau_ema_factor"],
            "phase_tau_accumulator_dtype": states["phase"]["tau_accumulator_dtype"],
            "state_sha256": {
                name: sha256_file(directory / f"{name}_state.pt")
                for name in states
            },
        }
        if "clip" in states:
            summary["clip_valid"] = bool(
                torch.all(states["clip"]["lower"] < states["clip"]["upper"])
            )
        write_json(directory / "calibration_summary.json", summary)
        manifest["sites"][key] = summary
    global_statistics = root / "_global" / "final_rmsnorm" / "statistics.pt"
    if not global_statistics.exists():
        raise FileNotFoundError(
            f"Final RMSNorm Phase statistics are missing: {global_statistics}"
        )
    global_directory = global_statistics.parent
    final_phase_state = build_phase_state(
        torch.load(global_statistics, map_location="cpu", weights_only=False), cfg
    )
    final_phase_path = global_directory / "phase_state.pt"
    torch.save(final_phase_state, final_phase_path)
    manifest["global_states"] = {
        "final_rmsnorm": {
            "phase_state_path": str(final_phase_path.relative_to(root)),
            "phase_state_sha256": sha256_file(final_phase_path),
            "parameter_layout": final_phase_state["parameter_layout"],
            "configured_group_size": final_phase_state["configured_group_size"],
            "effective_group_size": final_phase_state["group_size"],
            "tau_shape": list(final_phase_state["tau"].shape),
        }
    }
    expected = int(cfg["calibration"]["expected_sites_per_layer"])
    if expected != SITE_COUNT:
        raise ValueError(
            "calibration.expected_sites_per_layer must match "
            f"the code topology: config={expected}, code={SITE_COUNT}"
        )
    site_sets = validate_site_topology(
        root, expected_num_hidden_layers=expected_num_hidden_layers
    )
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
    expected_num_hidden_layers = getattr(model.config, "num_hidden_layers", None)
    if (
        not isinstance(expected_num_hidden_layers, int)
        or isinstance(expected_num_hidden_layers, bool)
        or expected_num_hidden_layers <= 0
    ):
        raise ValueError("model.config.num_hidden_layers must be a positive integer")
    existing_layers = [path for path in root.glob("layer_*") if path.is_dir()]
    if existing_layers:
        validate_site_topology(
            root, expected_num_hidden_layers=expected_num_hidden_layers
        )
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
            if (
                manifest_name == "statistics_manifest.json"
                and metadata.get("format_version") != STATISTICS_FORMAT_VERSION
            ):
                raise RuntimeError(
                    "Existing calibration artifact uses a stale statistics schema; "
                    "remove or move the old sites/ directory before recalibrating."
                )
            if (
                manifest_name == "calibration_state_manifest.json"
                and metadata.get("format_version")
                != CALIBRATION_MANIFEST_FORMAT_VERSION
            ):
                raise RuntimeError(
                    "Existing calibration artifact uses a stale calibration manifest "
                    "schema; remove or move the old sites/ directory before recalibrating."
                )
    dataset = tokenize_dataset(calibration_raw, tokenizer, cfg, prefix_ids=None)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["calibration"].get("batch_size", 1)),
        shuffle=False,
        collate_fn=CausalLMCollator(tokenizer),
    )
    if int(cfg["calibration"].get("batch_size", 1)) != 1:
        raise ValueError("Phase EMA calibration requires batch_size=1")
    controller.mode = "collect"
    actual_prefix_length = prefix_length(prefix_key_values)
    controller.statistics = StatisticsStore()
    install_prefix_kv_forward(model, prefix_key_values, controller=controller)
    model.eval()
    final_norm = get_model_parts(model).final_norm
    final_norm_handle = final_norm.register_forward_hook(
        lambda _module, _inputs, output: controller.statistics.update_global(
            "final_rmsnorm", output
        )
    )
    device = next(model.parameters()).device
    try:
        for batch in loader:
            model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                use_cache=False,
            )
    finally:
        final_norm_handle.remove()
    stats_manifest = controller.statistics.reduce_and_save(site_root)
    validate_site_topology(
        root, expected_num_hidden_layers=expected_num_hidden_layers
    )
    eligible_ann = purpose == "ann_training_calibration"
    eligible_conversion = purpose in {
        "ann_training_calibration",
        "post_finetuning_conversion_calibration",
    }
    conversion_reuse_policy = {
        "ann_training_calibration": "aware_modes_only",
        "vanilla_analysis_calibration": "none",
        "post_finetuning_conversion_calibration": "final_ann_only",
    }[purpose]
    state_profile = {
        "ann_training_calibration": "ann_training_with_common_clip",
        "vanilla_analysis_calibration": "analysis_statistics_only",
        "post_finetuning_conversion_calibration": "snn_conversion_without_clip",
    }[purpose]
    metadata = {
        "purpose": purpose,
        "analysis_only": purpose == "vanilla_analysis_calibration",
        "eligible_for_ann_training": eligible_ann,
        "eligible_for_conversion": eligible_conversion,
        "conversion_reuse_policy": conversion_reuse_policy,
        "post_finetuning_recalibration": purpose == "post_finetuning_conversion_calibration",
        "state_profile": state_profile,
        "common_clip_required": eligible_ann,
        "common_clip_generated": eligible_ann,
        "common_clip_application_control": "replacement.common_clip_enabled",
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
        "actual_prefix_length": actual_prefix_length,
        "calibration_group_size": int(cfg["calibration"]["group_size"]),
        "calibration_grouping_policy": CALIBRATION_GROUPING_POLICY,
        "statistics_format_version": STATISTICS_FORMAT_VERSION,
        "softmax_site5_grouping_policy": SOFTMAX_SITE5_GROUPING_POLICY,
        "softmax_site5_gif_policy": SOFTMAX_SITE5_GIF_POLICY,
        "softmax_site5_clip_policy": SOFTMAX_SITE5_CLIP_POLICY,
        "clip_eligible_site_ids": sorted(CLIP_ELIGIBLE_SITE_IDS),
        "clip_excluded_site_ids": [SOFTMAX_SITE_ID],
        **(extra_metadata or {}),
        "purpose": purpose,
        "analysis_only": purpose == "vanilla_analysis_calibration",
        "eligible_for_ann_training": eligible_ann,
        "eligible_for_conversion": eligible_conversion,
        "conversion_reuse_policy": conversion_reuse_policy,
        "post_finetuning_recalibration": purpose == "post_finetuning_conversion_calibration",
        "state_profile": state_profile,
        "common_clip_required": eligible_ann,
        "common_clip_generated": eligible_ann,
        "common_clip_application_control": "replacement.common_clip_enabled",
        "expected_num_hidden_layers": expected_num_hidden_layers,
        "expected_layer_names": [
            f"layer_{index:03d}" for index in range(expected_num_hidden_layers)
        ],
    }
    stats_manifest.update(metadata)
    write_json(root / "statistics_manifest.json", stats_manifest)
    state_manifest = (
        materialize_calibration_states(
            site_root,
            cfg,
            metadata,
            include_clip=eligible_ann,
            expected_num_hidden_layers=expected_num_hidden_layers,
        )
        if materialize_states
        else {
            **metadata,
            "format_version": CALIBRATION_MANIFEST_FORMAT_VERSION,
            **topology_metadata(),
            **temporal_policy_metadata(),
            "expected_num_hidden_layers": expected_num_hidden_layers,
            "expected_layer_names": [
                f"layer_{index:03d}" for index in range(expected_num_hidden_layers)
            ],
            "sites": {},
        }
    )
    if not materialize_states:
        write_json(root / "calibration_state_manifest.json", state_manifest)
    return {"statistics": stats_manifest, "states": state_manifest}
