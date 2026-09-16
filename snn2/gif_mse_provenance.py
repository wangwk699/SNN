from __future__ import annotations

from typing import Any

from .config import gif_mse_refinement_signature, previous_layers_snn_enabled


def validate_histogram_provenance(
    histogram: dict[str, Any], manifest: dict[str, Any], cfg: dict[str, Any],
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
