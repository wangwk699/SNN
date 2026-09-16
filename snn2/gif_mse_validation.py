from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import gif_mse_refinement_enabled, gif_mse_refinement_signature
from .gif_mse_calibration import normalize_mse_refinement_config
from .temporal_ops import GIF_SCALE_MIN
from .gif_mse_provenance import validate_histogram_provenance


def validate_gif_qparam_manifest_compatibility(
    manifest: dict[str, Any], cfg: dict[str, Any], *, context: str | Path,
) -> None:
    """Validate MSE manifest fields while retaining legacy direct artifacts."""
    context = Path(context)
    enabled = gif_mse_refinement_enabled(cfg)
    if enabled:
        expected = {
            "gif_scale_initialization": "direct_min_max",
            "gif_mse_scale_refinement": True,
            "gif_qparam_calibration_method": "offline_static_mse",
            "gif_mse_refinement_signature": gif_mse_refinement_signature(cfg),
            "gif_mse_refinement_config": normalize_mse_refinement_config(cfg["gif"].get("mse_refinement")),
        }
        for key, value in expected.items():
            if key not in manifest or manifest.get(key) != value:
                raise ValueError(f"MSE calibration manifest has invalid {key}: {context}")
        return
    direct_expected = {
        "gif_scale_initialization": "direct_min_max",
        "gif_mse_scale_refinement": False,
        "gif_qparam_calibration_method": "direct_min_max",
        "gif_mse_refinement_signature": None,
    }
    for key, value in direct_expected.items():
        if key in manifest and manifest.get(key) != value:
            raise ValueError(f"Direct GIF calibration manifest conflicts at {key}: {context}")


def validate_gif_mse_state(
    state: dict[str, Any], cfg: dict[str, Any], *, path: str | Path,
    manifest: dict[str, Any] | None = None,
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
    if manifest is not None:
        histogram = torch.load(histogram_path, map_location="cpu", weights_only=False)
        validate_histogram_provenance(histogram, manifest, cfg, site_directory=path.parent)
    for branch, qmax in (("low", 15), ("high", 30)):
        if f"{branch}_scale" not in state:
            continue
        for prefix in ("", "direct_"):
            scale = torch.as_tensor(state[f"{prefix}{branch}_scale"])
            zero = torch.as_tensor(state[f"{prefix}{branch}_zero"])
            if not torch.isfinite(scale).all() or not torch.all(scale >= GIF_SCALE_MIN):
                raise ValueError(f"MSE GIF {branch} scale is below runtime GIF_SCALE_MIN: {path}")
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
