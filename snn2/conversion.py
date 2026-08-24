from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import ArtifactLayout, read_json, sha256_file, write_json
from .config import (
    conversion_calibration_stage,
    conversion_prefix_enabled,
    conversion_reuses_ann_training_artifacts,
)
from .controller import SiteController
from .sites import topology_metadata
from .state_validation import validate_site_state_bundle
from .temporal_ops import (
    CONVERSION_METADATA_FORMAT_VERSION,
    GIF_LOCAL_STEPS,
    temporal_policy_metadata,
    validate_temporal_policy,
)


def _ann_num_hidden_layers(ann_config: Path) -> int:
    if not ann_config.exists():
        raise FileNotFoundError(ann_config)
    value = read_json(ann_config).get("num_hidden_layers")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{ann_config} num_hidden_layers must be a positive integer")
    return value


def validate_calibration(
    site_root: str | Path,
    *,
    expected_num_hidden_layers: int | None = None,
    allow_clip_bundle: bool = False,
) -> dict[str, Any]:
    root = Path(site_root)
    validation = validate_site_state_bundle(
        root,
        require_clip=allow_clip_bundle,
        expected_num_hidden_layers=expected_num_hidden_layers,
    )
    required = ["statistics.pt", "phase_state.pt", "gif_state.pt", "mtn_state.pt"]
    if allow_clip_bundle:
        required.append("clip_state.pt")
    for directory in sorted(path for path in root.glob("layer_*/site_*") if path.is_dir()):
        for name in required:
            if not (directory / name).exists():
                raise FileNotFoundError(directory / name)
    clip_states = sorted(root.glob("layer_*/site_*/clip_state.pt"))
    if not allow_clip_bundle and clip_states:
        raise ValueError(
            "Post-finetuning conversion calibration must be clip-free; "
            "re-run calibrate_sites.py --stage post_finetuning"
        )
    return {
        "expected_num_hidden_layers": validation["expected_num_hidden_layers"],
        "layers": validation["layers"],
        "sites": validation["sites"],
        "temporal_steps": validation["temporal_steps"],
        **topology_metadata(),
    }


def _validate_source_manifest(manifest: dict[str, Any], *, reused: bool) -> None:
    expected = (
        {
            "purpose": "ann_training_calibration",
            "eligible_for_ann_training": True,
            "eligible_for_conversion": True,
            "conversion_reuse_policy": "aware_modes_only",
            "post_finetuning_recalibration": False,
            "state_profile": "ann_training_with_common_clip",
            "common_clip_required": True,
        }
        if reused
        else {
            "purpose": "post_finetuning_conversion_calibration",
            "eligible_for_ann_training": False,
            "eligible_for_conversion": True,
            "conversion_reuse_policy": "final_ann_only",
            "post_finetuning_recalibration": True,
            "state_profile": "snn_conversion_without_clip",
            "common_clip_required": False,
        }
    )
    mismatched = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatched:
        source = "ANN-training" if reused else "post-finetuning"
        raise ValueError(f"Invalid {source} conversion calibration manifest: {mismatched}")


def validate_conversion_prefix(
    cfg: dict[str, Any], layout: ArtifactLayout
) -> dict[str, Any]:
    enabled = conversion_prefix_enabled(cfg)
    source_stage = (
        "pre_finetuning"
        if conversion_reuses_ann_training_artifacts(cfg)
        else "post_finetuning"
    )
    root = layout.conversion_prefix_dir
    if not enabled:
        return {
            "prefix_root": None,
            "prefix_source_stage": source_stage,
            "prefix_token_ids": [],
            "prefix_state_sha256": None,
            "prefix_kv_sha256": None,
        }
    state_path = root / "prefix_state.json"
    if not state_path.exists():
        command_stage = "pre_finetuning" if source_stage == "pre_finetuning" else "post_finetuning"
        raise FileNotFoundError(
            f"Conversion Prefix state is missing: {state_path}. Run "
            f"scripts/discover_prefix.py --stage {command_stage}."
        )
    token_ids = [int(value) for value in read_json(state_path).get("prefix_token_ids", [])]
    kv_path = root / "prefixed_key_values.pt"
    if token_ids and not kv_path.exists():
        raise FileNotFoundError(f"Non-empty conversion Prefix requires KV cache: {kv_path}")
    return {
        "prefix_root": str(root.resolve()),
        "prefix_source_stage": source_stage,
        "prefix_token_ids": token_ids,
        "prefix_state_sha256": sha256_file(state_path),
        "prefix_kv_sha256": sha256_file(kv_path) if token_ids else None,
    }


def _validate_aware_training_provenance(
    layout: ArtifactLayout,
    prefix: dict[str, Any],
    calibration_manifest: Path,
) -> dict[str, Any]:
    result_path = layout.ann_dir / "training_result.json"
    if not result_path.exists():
        raise FileNotFoundError(
            f"Aware conversion requires ANN training provenance: {result_path}"
        )
    result = read_json(result_path)
    expected = {
        "ann_training_prefix_root": prefix["prefix_root"],
        "ann_training_prefix_state_sha256": prefix["prefix_state_sha256"],
        "ann_training_prefix_kv_sha256": prefix["prefix_kv_sha256"],
        "ann_training_prefix_token_ids": prefix["prefix_token_ids"],
        "ann_training_calibration_root": str(layout.ann_training_site_dir.resolve()),
        "ann_training_calibration_manifest_sha256": sha256_file(calibration_manifest),
    }
    mismatched = {
        key: {"expected": value, "actual": result.get(key)}
        for key, value in expected.items()
        if result.get(key) != value
    }
    if mismatched:
        raise ValueError(
            "Aware conversion artifacts differ from those fixed during ANN training: "
            f"{mismatched}"
        )
    return {"training_result_path": str(result_path.resolve()), **expected}


def _source_bundle(
    cfg: dict[str, Any], layout: ArtifactLayout
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    reused = conversion_reuses_ann_training_artifacts(cfg)
    prefix = validate_conversion_prefix(cfg, layout)
    ann_config = layout.ann_checkpoint_dir / "config.json"
    layers = _ann_num_hidden_layers(ann_config)
    validation = validate_calibration(
        layout.conversion_site_dir,
        expected_num_hidden_layers=layers,
        allow_clip_bundle=reused,
    )
    manifest_path = layout.conversion_site_dir / "calibration_state_manifest.json"
    manifest = read_json(manifest_path)
    _validate_source_manifest(manifest, reused=reused)
    if manifest.get("prefix_enabled") != conversion_prefix_enabled(cfg):
        raise ValueError("Conversion calibration Prefix policy disagrees with config")
    provenance = (
        _validate_aware_training_provenance(layout, prefix, manifest_path)
        if reused
        else {}
    )
    return prefix, validation, manifest_path, provenance


def validate_conversion_metadata(
    cfg: dict[str, Any], layout: ArtifactLayout, neuron: str
) -> dict[str, Any]:
    path = layout.snn_conversion_dir(neuron) / "conversion_metadata.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Converted SNN metadata is missing: {path}. Run scripts/convert_snn.py first."
        )
    metadata = read_json(path)
    if metadata.get("format_version") != CONVERSION_METADATA_FORMAT_VERSION:
        raise ValueError(f"{path} uses a legacy conversion schema; format v5 is required")
    validate_temporal_policy(metadata, context=str(path))
    prefix, bundle, manifest_path, _ = _source_bundle(cfg, layout)
    reused = conversion_reuses_ann_training_artifacts(cfg)
    ann_config = layout.ann_checkpoint_dir / "config.json"
    rotation_enabled = bool(cfg["rotation"]["enabled"])
    rotation_path = layout.rotation_dir / "rotation_state.pt"
    expected = {
        "deployment_neuron": neuron,
        "full_temporal_steps": bundle["temporal_steps"].get(neuron),
        "source_ann_checkpoint": str(layout.ann_checkpoint_dir.resolve()),
        "source_ann_config_sha256": sha256_file(ann_config),
        "calibration_root": str(layout.conversion_site_dir.resolve()),
        "calibration_state_manifest_sha256": sha256_file(manifest_path),
        "calibration_source_stage": conversion_calibration_stage(cfg),
        "prefix_source_stage": prefix["prefix_source_stage"],
        "reused_ann_training_artifacts": reused,
        "post_finetuning_recalibration": not reused,
        "prefix_enabled": conversion_prefix_enabled(cfg),
        "prefix_token_ids": prefix["prefix_token_ids"],
        "prefix_state_sha256": prefix["prefix_state_sha256"],
        "prefix_kv_sha256": prefix["prefix_kv_sha256"],
        "rotation_enabled": rotation_enabled,
        "rotation_state_sha256": sha256_file(rotation_path) if rotation_enabled else None,
        "expected_num_hidden_layers": _ann_num_hidden_layers(ann_config),
        "snn_clip_applied": False,
        "gif_local_decomposition_steps": GIF_LOCAL_STEPS,
    }
    mismatched = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatched:
        raise ValueError(f"{path} does not match the current conversion artifacts: {mismatched}")
    return metadata


def create_conversion(
    cfg: dict[str, Any], layout: ArtifactLayout, neuron: str
) -> dict[str, Any]:
    ann_checkpoint = layout.ann_checkpoint_dir
    ann_config = ann_checkpoint / "config.json"
    expected_num_hidden_layers = _ann_num_hidden_layers(ann_config)
    prefix, validation, manifest_path, training_provenance = _source_bundle(cfg, layout)
    controller = SiteController(site_root=layout.conversion_site_dir)
    steps = controller.set_deployment(neuron)
    output = layout.snn_conversion_dir(neuron)
    output.mkdir(parents=True, exist_ok=True)
    rotation_enabled = bool(cfg["rotation"]["enabled"])
    rotation_path = layout.rotation_dir / "rotation_state.pt"
    if rotation_enabled and not rotation_path.exists():
        raise FileNotFoundError(rotation_path)
    reused = conversion_reuses_ann_training_artifacts(cfg)
    metadata = {
        "format_version": CONVERSION_METADATA_FORMAT_VERSION,
        "experiment": cfg["experiment"],
        "source_ann_checkpoint": str(ann_checkpoint.resolve()),
        "source_ann_config_sha256": sha256_file(ann_config),
        "calibration_root": str(layout.conversion_site_dir.resolve()),
        "calibration_state_manifest_sha256": sha256_file(manifest_path),
        "calibration_source_stage": conversion_calibration_stage(cfg),
        "prefix_source_stage": prefix["prefix_source_stage"],
        "reused_ann_training_artifacts": reused,
        "deployment_neuron": neuron,
        "full_temporal_steps": steps,
        "gif_local_decomposition_steps": GIF_LOCAL_STEPS,
        "rotation_enabled": rotation_enabled,
        "expected_num_hidden_layers": expected_num_hidden_layers,
        "prefix_enabled": conversion_prefix_enabled(cfg),
        "prefix_token_ids": prefix["prefix_token_ids"],
        "rotation_state_sha256": sha256_file(rotation_path) if rotation_enabled else None,
        "prefix_state_sha256": prefix["prefix_state_sha256"],
        "prefix_kv_sha256": prefix["prefix_kv_sha256"],
        "post_finetuning_recalibration": not reused,
        "snn_clip_applied": False,
        "prefix_root": prefix["prefix_root"],
        "calibration_validation": validation,
        "training_artifact_provenance": training_provenance,
        **temporal_policy_metadata(),
        **topology_metadata(),
    }
    write_json(output / "conversion_metadata.json", metadata)
    return metadata
