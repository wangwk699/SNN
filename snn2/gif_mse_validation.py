from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import gif_mse_refinement_enabled, gif_mse_refinement_signature


def validate_gif_mse_state(
    state: dict[str, Any], cfg: dict[str, Any], *, path: str | Path,
) -> None:
    path = Path(path)
    enabled = gif_mse_refinement_enabled(cfg)
    quantized = bool(state.get("quantization_applied", False))
    histogram_path = path.parent / "gif_mse_histogram.pt"
    if not enabled:
        if bool(state.get("mse_refinement", False)):
            raise ValueError(f"Direct GIF state unexpectedly enables MSE refinement: {path}")
        return
    if not quantized:
        if histogram_path.exists():
            raise ValueError(f"Identity GIF site must not save an MSE histogram: {path.parent}")
        return
    expected = {
        "mse_refinement": True,
        "qparam_calibration_method": "offline_static_mse",
        "mse_refinement_version": "static_mse_v1",
        "mse_objective": "elementwise_mse",
        "runtime_quantization": "static",
        "mse_refinement_signature": gif_mse_refinement_signature(cfg),
        "configured_group_size": int(cfg["calibration"]["group_size"]),
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"MSE GIF state has invalid {key}: {path}")
    if not histogram_path.exists():
        raise FileNotFoundError(histogram_path)
    for branch, qmax in (("low", 15), ("high", 30)):
        if f"{branch}_scale" not in state:
            continue
        for prefix in ("", "direct_"):
            scale = torch.as_tensor(state[f"{prefix}{branch}_scale"])
            zero = torch.as_tensor(state[f"{prefix}{branch}_zero"])
            if not torch.isfinite(scale).all() or not torch.all(scale > 0):
                raise ValueError(f"MSE GIF {branch} scale is invalid: {path}")
            if not torch.isfinite(zero).all() or not torch.equal(zero, torch.round(zero)):
                raise ValueError(f"MSE GIF {branch} zero is not integer-valued: {path}")
            if torch.any(zero < 0) or torch.any(zero > qmax):
                raise ValueError(f"MSE GIF {branch} zero is outside [0, {qmax}]: {path}")
        diagnostic = state.get("mse_diagnostics", {}).get(branch)
        if not isinstance(diagnostic, dict):
            raise ValueError(f"MSE GIF {branch} diagnostics are missing: {path}")
        baseline = torch.as_tensor(diagnostic["baseline_mse"], dtype=torch.float64)
        refined = torch.as_tensor(diagnostic["refined_mse"], dtype=torch.float64)
        for value in diagnostic.values():
            if isinstance(value, torch.Tensor) and value.dtype != torch.bool and not torch.isfinite(value).all():
                raise ValueError(f"MSE GIF diagnostics contain non-finite values: {path}")
        if torch.any(refined > baseline + 1e-12):
            raise ValueError(f"MSE GIF refined error exceeds baseline: {path}")
