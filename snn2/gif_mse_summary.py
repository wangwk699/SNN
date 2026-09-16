from __future__ import annotations

from typing import Any

import torch


def summarize_gif_mse_state(state: dict[str, Any], *, histogram_bins: int) -> dict[str, Any]:
    refined = bool(state.get("mse_refinement", False))
    result: dict[str, Any] = {
        "gif_qparam_calibration_method": (
            "offline_static_mse" if refined else "direct_min_max"
        ),
        "gif_mse_refinement_version": state.get("mse_refinement_version"),
        "gif_histogram_bins": int(histogram_bins) if refined else None,
        "gif_refined_group_count": 0,
        "gif_fallback_group_count": 0,
    }
    diagnostics = state.get("mse_diagnostics", {})
    for branch in ("low", "high"):
        data = diagnostics.get(branch)
        if not isinstance(data, dict):
            result.update({
                f"gif_{branch}_mse_reduction_mean": None,
                f"gif_{branch}_mse_reduction_median": None,
                f"gif_{branch}_clip_ratio_mean": None,
            })
            continue
        baseline = torch.as_tensor(data["baseline_mse"], dtype=torch.float64)
        final = torch.as_tensor(data["refined_mse"], dtype=torch.float64)
        reduction = baseline - final
        clip = torch.as_tensor(data["clip_ratio_total"], dtype=torch.float64)
        applied = torch.as_tensor(data["refinement_applied"], dtype=torch.bool)
        samples = torch.as_tensor(data["sample_count"], dtype=torch.float64)
        valid = samples > 0
        result[f"gif_{branch}_mse_reduction_mean"] = (
            float(reduction[valid].mean()) if valid.any() else None
        )
        result[f"gif_{branch}_mse_reduction_median"] = (
            float(reduction[valid].median()) if valid.any() else None
        )
        result[f"gif_{branch}_clip_ratio_mean"] = (
            float(clip[valid].mean()) if valid.any() else None
        )
        result["gif_refined_group_count"] += int((applied & valid).sum())
        result["gif_fallback_group_count"] += int(((~applied) & valid).sum())
    return result
