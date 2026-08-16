from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .artifacts import ArtifactLayout, sha256_file, write_json
from .controller import SiteController


def validate_calibration(site_root: str | Path) -> dict[str, Any]:
    root = Path(site_root)
    sites = sorted(path.parent for path in root.glob("layer_*/site_*/clip_state.pt"))
    if not sites:
        raise FileNotFoundError(f"No site calibration states under {root}")
    layers: dict[str, int] = {}
    for directory in sites:
        layer = directory.parent.name
        layers[layer] = layers.get(layer, 0) + 1
        for name in ("statistics.pt", "phase_state.pt", "gif_state.pt", "mtn_state.pt", "clip_state.pt"):
            if not (directory / name).exists():
                raise FileNotFoundError(directory / name)
        clip = torch.load(directory / "clip_state.pt", map_location="cpu", weights_only=False)
        if torch.any(clip["lower"] >= clip["upper"]):
            raise ValueError(f"Invalid clipping interval: {directory}")
    incomplete = {layer: count for layer, count in layers.items() if count != 9}
    if incomplete:
        raise RuntimeError(f"Expected nine sites per layer, got {incomplete}")
    return {"layers": len(layers), "sites": len(sites), "site_counts": layers}


def create_conversion(cfg: dict[str, Any], layout: ArtifactLayout, neuron: str) -> dict[str, Any]:
    validation = validate_calibration(layout.site_dir)
    ann_checkpoint = layout.ann_dir / "best"
    ann_config = ann_checkpoint / "config.json"
    if not ann_config.exists():
        raise FileNotFoundError(
            "The validation-best ANN checkpoint is required before conversion: "
            f"{ann_config}"
        )
    calibration_manifest = layout.site_dir / "calibration_state_manifest.json"
    if not calibration_manifest.exists():
        raise FileNotFoundError(
            f"Calibration state manifest is missing: {calibration_manifest}"
        )
    controller = SiteController(site_root=layout.site_dir)
    steps = controller.set_deployment(neuron)
    output = layout.snn_dir(neuron)
    output.mkdir(parents=True, exist_ok=True)
    rotation_path = layout.rotation_dir / "rotation_state.pt"
    prefix_path = layout.prefix_dir / "prefix_state.json"
    metadata = {
        "format_version": 1,
        "experiment": cfg["experiment"],
        "source_ann_checkpoint": str(ann_checkpoint.resolve()),
        "source_ann_config_sha256": sha256_file(ann_config),
        "calibration_root": str(layout.site_dir.resolve()),
        "calibration_state_manifest_sha256": sha256_file(calibration_manifest),
        "deployment_neuron": neuron,
        "full_temporal_steps": steps,
        "gif_local_decomposition_steps": 2 ** int(cfg["gif"]["add_bits"]),
        "rotation_enabled": bool(cfg["rotation"]["enabled"]),
        "prefix_enabled": bool(cfg["prefix"]["enabled"]),
        "rotation_state_sha256": (
            sha256_file(rotation_path) if bool(cfg["rotation"]["enabled"]) else None
        ),
        "prefix_state_sha256": (
            sha256_file(prefix_path) if bool(cfg["prefix"]["enabled"]) else None
        ),
        "post_finetuning_recalibration": False,
        "calibration_validation": validation,
    }
    write_json(output / "conversion_metadata.json", metadata)
    return metadata
