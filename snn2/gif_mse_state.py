from __future__ import annotations

import itertools
from typing import Any

import torch

from .gif_mse_calibration import (
    normalize_mse_refinement_config,
    mse_refinement_signature,
    optimize_static_qparams,
)


def build_histogram_spec(
    statistics: dict[str, Any], direct_state: dict[str, Any], *, configured_group_size: int,
) -> dict[str, Any] | None:
    if not direct_state.get("quantization_applied", False):
        return None
    layout = statistics["layout_kind"]
    if layout not in {"last_dim", "attention_head"}:
        return None
    effective = int(direct_state["group_size"])
    minimum = statistics["value_min"].double()
    maximum = statistics["value_max"].double()
    if direct_state.get("mask_policy") == "multi_role":
        masks = {role: value.bool().cpu() for role, value in direct_state["mask_low_by_role"].items()}
    elif direct_state.get("mask_policy") == "single":
        masks = {"default": direct_state["mask_low"].bool().cpu()}
    else:
        masks = {"default": torch.ones_like(minimum, dtype=torch.bool)}
    group_shape = (*minimum.shape[:-1], minimum.shape[-1] // effective)
    result: dict[str, Any] = {
        "layout_kind": layout,
        "parameter_layout": direct_state["parameter_layout"],
        "configured_group_size": int(configured_group_size),
        "effective_group_size": effective,
        "mask_low_by_role": masks,
        "branches": ["low"] + (["high"] if "high_scale" in direct_state else []),
    }
    for branch in result["branches"]:
        selected = torch.stack(
            [mask if branch == "low" else ~mask for mask in masks.values()], dim=0
        ).any(dim=0)
        grouped_selected = selected.reshape(*group_shape, effective)
        grouped_min = minimum.reshape(*group_shape, effective)
        grouped_max = maximum.reshape(*group_shape, effective)
        lower = torch.where(grouped_selected, grouped_min, torch.inf).amin(dim=-1)
        upper = torch.where(grouped_selected, grouped_max, -torch.inf).amax(dim=-1)
        empty = ~grouped_selected.any(dim=-1)
        result[f"{branch}_lower"] = torch.where(empty, grouped_min.amin(dim=-1), lower)
        result[f"{branch}_upper"] = torch.where(empty, grouped_max.amax(dim=-1), upper)
    return result


def refine_gif_state(
    direct_state: dict[str, Any], histogram: dict[str, Any], cfg: dict[str, Any],
) -> dict[str, Any]:
    if not direct_state.get("quantization_applied", False):
        return direct_state
    mse_cfg = normalize_mse_refinement_config(cfg["gif"].get("mse_refinement"))
    signature = mse_refinement_signature(mse_cfg)
    expected = {
        "refinement_version": "static_mse_v1",
        "configured_group_size": int(cfg["calibration"]["group_size"]),
        "histogram_bins": int(mse_cfg["histogram_bins"]),
        "mse_refinement_signature": signature,
    }
    for key, value in expected.items():
        if histogram.get(key) != value:
            raise ValueError(f"GIF MSE histogram provenance mismatch for {key}")
    result = dict(direct_state)
    diagnostics: dict[str, dict[str, Any]] = {}
    for branch, qmax in (("low", 15), ("high", 30)):
        scale_key, zero_key = f"{branch}_scale", f"{branch}_zero"
        if scale_key not in direct_state:
            continue
        source = histogram.get(branch)
        if not isinstance(source, dict):
            raise ValueError(f"GIF MSE histogram is missing the {branch} branch")
        direct_scale = direct_state[scale_key].double()
        direct_zero = direct_state[zero_key].double()
        refined_scale, refined_zero = direct_scale.clone(), direct_zero.clone()
        names = (
            "sample_count", "baseline_mse", "refined_mse", "baseline_nmse",
            "refined_nmse", "optimized_lower", "optimized_upper",
            "representable_lower", "representable_upper", "clip_ratio_low",
            "clip_ratio_high", "clip_ratio_total",
        )
        fields = {name: torch.zeros_like(direct_scale) for name in names}
        applied = torch.zeros_like(direct_scale, dtype=torch.bool)
        reasons: list[str | None] = []
        for index in itertools.product(*(range(size) for size in direct_scale.shape)):
            optimized = optimize_static_qparams(
                {
                    "counts": source["counts"][index],
                    "bin_edges": source["bin_edges"][index],
                    "sample_count": int(source["sample_count"][index]),
                    "sum_sq": float(source["sum_sq"][index]),
                },
                direct_scale=float(direct_scale[index]), direct_zero=int(direct_zero[index]),
                qmin=0, qmax=qmax, config=mse_cfg,
            )
            refined_scale[index], refined_zero[index] = optimized["scale"], optimized["zero"]
            applied[index] = bool(optimized["refinement_applied"])
            reasons.append(optimized.get("fallback_reason"))
            for name in names:
                fields[name][index] = float(optimized.get(name, 0.0))
        result[f"direct_{branch}_scale"] = direct_state[scale_key].clone()
        result[f"direct_{branch}_zero"] = direct_state[zero_key].clone()
        result[scale_key], result[zero_key] = refined_scale.float(), refined_zero.float()
        diagnostics[branch] = {
            **fields, "refinement_applied": applied, "fallback_reason": reasons,
        }
    result.update({
        "scale_initialization": "direct_min_max",
        "mse_refinement": True,
        "qparam_calibration_method": "offline_static_mse",
        "mse_refinement_version": "static_mse_v1",
        "mse_objective": "elementwise_mse",
        "runtime_quantization": "static",
        "mse_refinement_signature": signature,
        "mse_diagnostics": diagnostics,
    })
    return result
