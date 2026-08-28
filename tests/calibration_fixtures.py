from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch

from snn2.artifacts import sha256_file, write_json
from snn2.sites import SITE_IDS, SITE_NAMES, topology_metadata
from snn2.temporal_ops import (
    CALIBRATION_GROUPING_POLICY,
    STATISTICS_FORMAT_VERSION,
    STATISTICS_MANIFEST_FORMAT_VERSION,
)


def write_stage_a_statistics(
    root: Path,
    cfg: dict[str, Any],
    statistics: Callable[[int | None], dict[str, Any]],
    *,
    layers: int = 1,
    purpose: str = "post_finetuning_conversion_calibration",
) -> None:
    sites: dict[str, dict[str, Any]] = {}
    for layer in range(layers):
        for site in SITE_IDS:
            directory = root / f"layer_{layer:03d}" / f"site_{site:02d}_{SITE_NAMES[site]}"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "statistics.pt"
            torch.save(statistics(site), path)
            write_json(directory / "statistics_summary.json", {"site_index": site})
            sites[directory.relative_to(root).as_posix()] = {"statistics_sha256": sha256_file(path)}
    global_dir = root / "_global" / "final_rmsnorm"
    global_dir.mkdir(parents=True, exist_ok=True)
    global_path = global_dir / "statistics.pt"
    torch.save(statistics(None), global_path)
    write_json(global_dir / "statistics_summary.json", {})
    manifest = {
        "format_version": STATISTICS_FORMAT_VERSION,
        "statistics_format_version": STATISTICS_FORMAT_VERSION,
        "statistics_manifest_format_version": STATISTICS_MANIFEST_FORMAT_VERSION,
        "calibration_group_size": int(cfg["calibration"]["group_size"]),
        "calibration_num_samples": int(cfg["calibration"]["num_samples"]),
        "calibration_seed": int(cfg["calibration"]["seed"]),
        "purpose": purpose,
        "source_model_stage": None,
        "source_ann_mode": None,
        "source_ann_checkpoint": None,
        "source_ann_config_sha256": None,
        "prefix_enabled": False,
        "prefix_state_sha256": None,
        "prefix_kv_sha256": None,
        "rotation_enabled": False,
        "rotation_state_sha256": None,
        "calibration_data_manifest_sha256": None,
        "calibration_grouping_policy": CALIBRATION_GROUPING_POLICY,
        "expected_num_hidden_layers": layers,
        "expected_layer_names": [f"layer_{index:03d}" for index in range(layers)],
        "sites": sites,
        "global_states": {"final_rmsnorm": {"statistics_sha256": sha256_file(global_path)}},
        **topology_metadata(),
    }
    write_json(root / "statistics_manifest.json", manifest)
