from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from .config import gif_mse_refinement_signature
from .artifacts import sha256_file
from .gif_mse_calibration import GIFMSEHistogramStore
from .gif_mse_state import build_histogram_spec


def create_histogram_store(
    site_root: str | Path, cfg: dict[str, Any], *, statistics_name: str,
    layer_index: int | None = None,
) -> GIFMSEHistogramStore:
    from .calibration import build_gif_state

    root = Path(site_root)
    pattern = (
        f"layer_{int(layer_index):03d}/site_*/{statistics_name}"
        if layer_index is not None else f"layer_*/site_*/{statistics_name}"
    )
    direct_cfg = deepcopy(cfg)
    direct_cfg["gif"]["mse_scale_refinement"] = False
    specs: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob(pattern)):
        statistics = torch.load(path, map_location="cpu", weights_only=False)
        direct_state = build_gif_state(statistics, direct_cfg)
        spec = build_histogram_spec(
            statistics, direct_state,
            configured_group_size=int(cfg["calibration"]["group_size"]),
        )
        if spec is not None:
            specs[path.parent.relative_to(root).as_posix()] = spec
    return GIFMSEHistogramStore(
        specs, histogram_bins=int(cfg["gif"]["mse_refinement"]["histogram_bins"]),
    )


def histogram_provenance(
    cfg: dict[str, Any], metadata: dict[str, Any], *, trajectory_source: str,
) -> dict[str, Any]:
    return {
        "num_samples": int(cfg["calibration"]["num_samples"]),
        "previous_layers_snn": trajectory_source == "sequential_temporal_gif",
        "trajectory_source": trajectory_source,
        "calibration_data_manifest_sha256": metadata.get("calibration_data_manifest_sha256"),
        "prefix_state_sha256": metadata.get("prefix_state_sha256"),
        "prefix_kv_sha256": metadata.get("prefix_kv_sha256"),
        "rotation_state_sha256": metadata.get("rotation_state_sha256"),
        "mse_refinement_signature": gif_mse_refinement_signature(cfg),
    }


def save_histogram_store(
    store: GIFMSEHistogramStore, site_root: str | Path, metadata: dict[str, Any],
    *, statistics_name: str,
) -> None:
    root = Path(site_root)
    for key in sorted(store.specs):
        path = root / key / "gif_mse_histogram.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        source_path = root / key / statistics_name
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        site_metadata = {
            **metadata,
            "source_statistics_file": statistics_name,
            "source_statistics_sha256": sha256_file(source_path),
        }
        torch.save(store.state_for(key, site_metadata), path)
