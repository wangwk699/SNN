from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import ArtifactLayout, read_json, sha256_file, write_json
from .config import post_finetuning_prefix_enabled
from .controller import SiteController
from .sites import SITE_COUNT, SITE_TOPOLOGY_VERSION, topology_metadata, validate_site_topology
from .state_validation import validate_site_state_bundle
from .temporal_ops import (
    CONVERSION_METADATA_FORMAT_VERSION,
    GIF_LOCAL_STEPS,
    temporal_policy_metadata,
    validate_temporal_policy,
)


def validate_calibration(site_root: str | Path) -> dict[str, Any]:
    root = Path(site_root)
    validation = validate_site_state_bundle(root, require_clip=True)
    for directory in sorted(path for path in root.glob("layer_*/site_*") if path.is_dir()):
        for name in (
            "statistics.pt",
            "phase_state.pt",
            "gif_state.pt",
            "mtn_state.pt",
            "clip_state.pt",
        ):
            if not (directory / name).exists():
                raise FileNotFoundError(directory / name)
    return {
        "layers": validation["layers"],
        "sites": validation["sites"],
        "temporal_steps": validation["temporal_steps"],
        **topology_metadata(),
    }


def validate_post_finetuning_prefix(layout: ArtifactLayout, enabled: bool = True) -> dict[str, Any]:
    if not enabled:
        return {
            "prefix_root": None,
            "prefix_token_ids": [],
            "prefix_state_sha256": None,
            "prefix_kv_sha256": None,
        }
    root = layout.post_finetuning_prefix_dir
    state_path = root / "prefix_state.json"
    if not state_path.exists():
        raise FileNotFoundError(
            f"Post-finetuning Prefix state is required before conversion: {state_path}. "
            "Re-run scripts/discover_prefix.py --stage post_finetuning."
        )
    token_ids = [int(value) for value in read_json(state_path).get("prefix_token_ids", [])]
    kv_path = root / "prefixed_key_values.pt"
    if token_ids and not kv_path.exists():
        raise FileNotFoundError(f"Non-empty post-finetuning Prefix requires fixed KV cache: {kv_path}")
    return {
        "prefix_root": str(root.resolve()),
        "prefix_token_ids": token_ids,
        "prefix_state_sha256": sha256_file(state_path),
        "prefix_kv_sha256": sha256_file(kv_path) if token_ids else None,
    }


def validate_conversion_metadata(
    cfg: dict[str, Any], layout: ArtifactLayout, neuron: str
) -> dict[str, Any]:
    """Reject stale conversion descriptors before an SNN evaluation starts."""
    path = layout.snn_conversion_dir(neuron) / "conversion_metadata.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Converted SNN metadata is missing: {path}. Run scripts/convert_snn.py first."
        )
    metadata = read_json(path)
    if metadata.get("format_version") != CONVERSION_METADATA_FORMAT_VERSION:
        raise ValueError(
            f"{path} uses a legacy conversion metadata format; re-run conversion"
        )
    validate_temporal_policy(metadata, context=str(path))
    expected_prefix = post_finetuning_prefix_enabled(cfg)
    prefix = validate_post_finetuning_prefix(layout, enabled=expected_prefix)
    bundle = validate_site_state_bundle(
        layout.post_finetuning_site_dir, require_clip=True
    )
    expected_root = str(layout.post_finetuning_site_dir.resolve())
    manifest_path = layout.post_finetuning_site_dir / "calibration_state_manifest.json"
    mismatched: dict[str, dict[str, Any]] = {}
    ann_config = layout.ann_checkpoint_dir / "config.json"
    expected = {
        "deployment_neuron": neuron,
        "full_temporal_steps": bundle["temporal_steps"].get(neuron),
        "source_ann_checkpoint": str(layout.ann_checkpoint_dir.resolve()),
        "source_ann_config_sha256": (
            sha256_file(ann_config) if ann_config.exists() else None
        ),
        "post_finetuning_recalibration": True,
        "rotation_enabled": bool(cfg["rotation"]["enabled"]),
        "prefix_enabled": expected_prefix,
        "prefix_token_ids": prefix["prefix_token_ids"],
        "prefix_state_sha256": prefix["prefix_state_sha256"],
        "prefix_kv_sha256": prefix["prefix_kv_sha256"],
        "calibration_root": expected_root,
        "calibration_state_manifest_sha256": (
            sha256_file(manifest_path) if manifest_path.exists() else None
        ),
        "common_clip_applied": True,
        "gif_local_decomposition_steps": GIF_LOCAL_STEPS,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            mismatched[key] = {"expected": value, "actual": metadata.get(key)}
    steps = metadata.get("full_temporal_steps")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        mismatched["full_temporal_steps"] = {
            "expected": bundle["temporal_steps"].get(neuron),
            "actual": steps,
        }
    if mismatched:
        raise ValueError(
            f"{path} does not match the requested v2 SNN evaluation: {mismatched}. "
            "Re-run scripts/convert_snn.py."
        )
    return metadata


def create_conversion(cfg: dict[str, Any], layout: ArtifactLayout, neuron: str) -> dict[str, Any]:
    prefix_enabled = post_finetuning_prefix_enabled(cfg)
    prefix = validate_post_finetuning_prefix(layout, enabled=prefix_enabled)
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
    if (
        manifest.get("purpose") != "post_finetuning_conversion_calibration"
        or not manifest.get("eligible_for_conversion")
        or not manifest.get("post_finetuning_recalibration")
        or manifest.get("state_profile") != "snn_conversion_with_common_clip"
        or manifest.get("common_clip_required") is not True
    ):
        raise ValueError(
            "Conversion requires post-finetuning calibration with temporal common Clip"
        )
    if manifest.get("prefix_enabled") != prefix_enabled:
        raise ValueError(
            "Conversion calibration prefix_enabled does not match "
            "post_finetuning.prefix_enabled"
        )
    controller = SiteController(site_root=layout.post_finetuning_site_dir)
    steps = controller.set_deployment(neuron)
    output = layout.snn_conversion_dir(neuron)
    output.mkdir(parents=True, exist_ok=True)
    rotation_path = layout.rotation_dir / "rotation_state.pt"
    metadata = {
        "format_version": CONVERSION_METADATA_FORMAT_VERSION,
        "experiment": cfg["experiment"],
        "source_ann_checkpoint": str(ann_checkpoint.resolve()),
        "source_ann_config_sha256": sha256_file(ann_config),
        "calibration_root": str(layout.post_finetuning_site_dir.resolve()),
        "calibration_state_manifest_sha256": sha256_file(calibration_manifest),
        "deployment_neuron": neuron,
        "full_temporal_steps": steps,
        "gif_local_decomposition_steps": GIF_LOCAL_STEPS,
        "rotation_enabled": bool(cfg["rotation"]["enabled"]),
        "prefix_enabled": prefix_enabled,
        "prefix_token_ids": prefix["prefix_token_ids"],
        "rotation_state_sha256": (
            sha256_file(rotation_path) if bool(cfg["rotation"]["enabled"]) else None
        ),
        "prefix_state_sha256": prefix["prefix_state_sha256"],
        "prefix_kv_sha256": prefix["prefix_kv_sha256"],
        "post_finetuning_recalibration": True,
        "common_clip_applied": True,
        "prefix_root": prefix["prefix_root"],
        "calibration_validation": validation,
        **temporal_policy_metadata(),
        **topology_metadata(),
    }
    write_json(output / "conversion_metadata.json", metadata)
    return metadata
