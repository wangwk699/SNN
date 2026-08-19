from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .artifacts import ArtifactLayout, read_json, sha256_file, write_json
from .sites import SITE_COUNT, SITE_TOPOLOGY_VERSION, topology_metadata, validate_site_topology
from .controller import SiteController


def validate_calibration(site_root: str | Path) -> dict[str, Any]:
    root = Path(site_root)
    site_sets = validate_site_topology(root)
    sites = sorted(path for path in root.glob("layer_*/site_*") if path.is_dir())
    for directory in sites:
        for name in ("statistics.pt", "phase_state.pt", "gif_state.pt", "mtn_state.pt", "clip_state.pt"):
            if not (directory / name).exists():
                raise FileNotFoundError(directory / name)
        clip = torch.load(directory / "clip_state.pt", map_location="cpu", weights_only=False)
        if torch.any(clip["lower"] >= clip["upper"]):
            raise ValueError(f"Invalid clipping interval: {directory}")
    manifest_path = root / "calibration_state_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("site_topology_version") != SITE_TOPOLOGY_VERSION or manifest.get("site_count") != SITE_COUNT:
            raise RuntimeError(
                "Calibration manifest topology does not match the current code: "
                f"expected version={SITE_TOPOLOGY_VERSION}, sites={SITE_COUNT}"
            )
    return {
        "layers": len(site_sets),
        "sites": len(sites),
        "site_counts": {layer: len(names) for layer, names in site_sets.items()},
        **topology_metadata(),
    }


def create_conversion(cfg: dict[str, Any], layout: ArtifactLayout, neuron: str) -> dict[str, Any]:
    validation = validate_calibration(layout.post_finetuning_site_dir)
    ann_checkpoint = layout.ann_checkpoint_dir
    ann_config = ann_checkpoint / "config.json"
    if not ann_config.exists():
        raise FileNotFoundError(
            "The final fine-tuned ANN checkpoint is required before conversion: "
            f"{ann_config}"
        )
    calibration_manifest = layout.post_finetuning_site_dir / "calibration_state_manifest.json"
    if not calibration_manifest.exists():
        raise FileNotFoundError(
            f"Calibration state manifest is missing: {calibration_manifest}"
        )
    manifest = read_json(calibration_manifest)
    if manifest.get("purpose") != "post_finetuning_conversion_calibration" or not manifest.get("eligible_for_conversion") or not manifest.get("post_finetuning_recalibration"):
        raise ValueError("Conversion requires post_finetuning_conversion_calibration eligible for conversion")
    controller = SiteController(site_root=layout.post_finetuning_site_dir)
    steps = controller.set_deployment(neuron)
    output = layout.snn_dir(neuron)
    output.mkdir(parents=True, exist_ok=True)
    rotation_path = layout.rotation_dir / "rotation_state.pt"
    prefix_path = layout.post_finetuning_prefix_dir / "prefix_state.json"
    metadata = {
        "format_version": 1,
        "experiment": cfg["experiment"],
        "source_ann_checkpoint": str(ann_checkpoint.resolve()),
        "source_ann_config_sha256": sha256_file(ann_config),
        "calibration_root": str(layout.post_finetuning_site_dir.resolve()),
        "calibration_state_manifest_sha256": sha256_file(calibration_manifest),
        "deployment_neuron": neuron,
        "full_temporal_steps": steps,
        "gif_local_decomposition_steps": 2 ** int(cfg["gif"]["add_bits"]),
        "rotation_enabled": bool(cfg["rotation"]["enabled"]),
        "prefix_enabled": True,
        "rotation_state_sha256": (
            sha256_file(rotation_path) if bool(cfg["rotation"]["enabled"]) else None
        ),
        "prefix_state_sha256": (
            sha256_file(prefix_path) if prefix_path.exists() else None
        ),
        "post_finetuning_recalibration": True,
        "prefix_root": str(layout.post_finetuning_prefix_dir.resolve()),
        "calibration_validation": validation,
        **topology_metadata(),
    }
    write_json(output / "conversion_metadata.json", metadata)
    return metadata
