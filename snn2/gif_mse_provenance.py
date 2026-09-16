from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .config import gif_mse_refinement_signature, previous_layers_snn_enabled


def validate_histogram_provenance(
    histogram: dict[str, Any], manifest: dict[str, Any], cfg: dict[str, Any],
    *, site_directory: str | Path,
) -> None:
    sequential = previous_layers_snn_enabled(cfg, "gif")
    expected = {
        "configured_group_size": int(cfg["calibration"]["group_size"]),
        "num_samples": int(cfg["calibration"]["num_samples"]),
        "previous_layers_snn": sequential,
        "trajectory_source": "sequential_temporal_gif" if sequential else "ann_common",
        "mse_refinement_signature": gif_mse_refinement_signature(cfg),
        "calibration_data_manifest_sha256": manifest.get("calibration_data_manifest_sha256"),
        "prefix_state_sha256": manifest.get("prefix_state_sha256"),
        "prefix_kv_sha256": manifest.get("prefix_kv_sha256"),
        "rotation_state_sha256": manifest.get("rotation_state_sha256"),
    }
    mismatched = {
        key: (value, histogram.get(key))
        for key, value in expected.items()
        if histogram.get(key) != value
    }
    if mismatched:
        raise ValueError(f"GIF MSE histogram provenance mismatch: {mismatched}")
    if histogram.get("format_version") != 2:
        raise ValueError("GIF MSE histogram must use format_version 2")
    expected_statistics_name = "gif_statistics.pt" if sequential else "statistics.pt"
    if histogram.get("source_statistics_file") != expected_statistics_name:
        raise ValueError(f"GIF MSE histogram source statistics file mismatch: expected {expected_statistics_name!r}")
    source_path = Path(site_directory) / expected_statistics_name
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if histogram.get("source_statistics_sha256") != sha256_file(source_path):
        raise ValueError(f"GIF MSE histogram source statistics hash mismatch: {source_path}")
