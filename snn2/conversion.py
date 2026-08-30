from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import ArtifactLayout, read_json, sha256_file, write_json
from .config import (
    conversion_calibration_stage,
    conversion_prefix_enabled,
    conversion_reuses_ann_training_artifacts,
    training_common_clip_enabled,
)
from .controller import SiteController
from .data import validate_prefix_discovery_state
from .sites import topology_metadata
from .state_validation import validate_site_state_bundle
from .temporal_ops import (
    CONVERSION_METADATA_FORMAT_VERSION,
    CALIBRATION_GROUPING_POLICY,
    GIF_LOCAL_STEPS,
    SOFTMAX_SITE5_CLIP_POLICY,
    SOFTMAX_SITE5_GIF_POLICY,
    SOFTMAX_SITE5_GROUPING_POLICY,
    STATISTICS_FORMAT_VERSION,
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
    clip_policy: str,
) -> dict[str, Any]:
    root = Path(site_root)
    validation = validate_site_state_bundle(
        root,
        clip_policy=clip_policy,
        expected_num_hidden_layers=expected_num_hidden_layers,
    )
    required = ["statistics.pt", "phase_state.pt", "gif_state.pt", "mtn_state.pt"]
    for directory in sorted(path for path in root.glob("layer_*/site_*") if path.is_dir()):
        for name in required:
            if not (directory / name).exists():
                raise FileNotFoundError(directory / name)
    clip_states = sorted(root.glob("layer_*/site_*/clip_state.pt"))
    if clip_policy == "forbid_all" and clip_states:
        raise ValueError(
            "Post-finetuning conversion calibration must be clip-free; "
            "re-run calibrate_sites.py --stage post_finetuning --calibration-phase A"
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
            "state_profile": "stage_a_common_states",
            "common_clip_required": False,
            "common_clip_generated": False,
            "common_clip_application_control": "replacement.common_clip_enabled",
        }
        if reused
        else {
            "purpose": "post_finetuning_conversion_calibration",
            "eligible_for_ann_training": False,
            "eligible_for_conversion": True,
            "conversion_reuse_policy": "final_ann_only",
            "post_finetuning_recalibration": True,
            "state_profile": "stage_a_common_states",
            "common_clip_required": False,
            "common_clip_generated": False,
            "common_clip_application_control": "replacement.common_clip_enabled",
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
            "prefix_num_samples": None,
            "prefix_discovery_manifest_sha256": None,
        }
    prefix_info = validate_prefix_discovery_state(cfg, layout, root)
    state_path = prefix_info["state_path"]
    kv_path = prefix_info["kv_path"]
    return {
        "prefix_root": str(root.resolve()),
        "prefix_source_stage": source_stage,
        "prefix_token_ids": prefix_info["token_ids"],
        "prefix_state_sha256": sha256_file(state_path),
        "prefix_kv_sha256": sha256_file(kv_path) if kv_path else None,
        "prefix_num_samples": int(prefix_info["state"]["discovery_num_samples"]),
        "prefix_discovery_manifest_sha256": prefix_info["state"][
            "discovery_manifest_sha256"
        ],
    }


def _validate_aware_training_provenance(
    cfg: dict[str, Any],
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
        "ann_training_prefix_num_samples": prefix["prefix_num_samples"],
        "ann_training_prefix_discovery_manifest_sha256": prefix[
            "prefix_discovery_manifest_sha256"
        ],
        "ann_training_prefix_token_ids": prefix["prefix_token_ids"],
        "ann_training_calibration_root": str(layout.ann_training_site_dir.resolve()),
        "ann_training_calibration_manifest_sha256": sha256_file(calibration_manifest),
        "ann_training_calibration_group_size": int(cfg["calibration"]["group_size"]),
        "ann_training_calibration_grouping_policy": CALIBRATION_GROUPING_POLICY,
        "statistics_format_version": STATISTICS_FORMAT_VERSION,
    }
    profile_path = Path(result.get("ann_training_clip_profile_root", "")) / "clip_profile_manifest.json"
    if not profile_path.exists() or result.get("ann_training_clip_profile_manifest_sha256") != sha256_file(profile_path):
        raise ValueError("Aware ANN training Stage B Clip profile provenance is missing or changed")
    for key in ("ann_training_phase_T", "ann_training_mtn_T", "ann_training_calibration_num_samples"):
        if not isinstance(result.get(key), int):
            raise ValueError(f"Aware ANN training provenance is missing {key}")
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
    return {
        "training_result_path": str(result_path.resolve()),
        **expected,
        "ann_training_phase_T": int(result["ann_training_phase_T"]),
        "ann_training_mtn_T": int(result["ann_training_mtn_T"]),
        "ann_training_calibration_num_samples": int(result["ann_training_calibration_num_samples"]),
        "ann_training_clip_profile_root": result["ann_training_clip_profile_root"],
        "ann_training_clip_profile_manifest_sha256": result["ann_training_clip_profile_manifest_sha256"],
    }


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
        clip_policy="forbid_all",
    )
    manifest_path = layout.conversion_site_dir / "calibration_state_manifest.json"
    manifest = read_json(manifest_path)
    _validate_source_manifest(manifest, reused=reused)
    expected_grouping = {
        "calibration_group_size": int(cfg["calibration"]["group_size"]),
        "calibration_num_samples": int(cfg["calibration"]["num_samples"]),
        "calibration_grouping_policy": CALIBRATION_GROUPING_POLICY,
        "statistics_format_version": STATISTICS_FORMAT_VERSION,
        "softmax_site5_grouping_policy": SOFTMAX_SITE5_GROUPING_POLICY,
        "softmax_site5_gif_policy": SOFTMAX_SITE5_GIF_POLICY,
        "softmax_site5_clip_policy": SOFTMAX_SITE5_CLIP_POLICY,
    }
    mismatch = {key: (value, manifest.get(key)) for key, value in expected_grouping.items() if manifest.get(key) != value}
    if mismatch:
        raise ValueError(f"Conversion calibration grouping provenance mismatch: {mismatch}")
    if manifest.get("prefix_enabled") != conversion_prefix_enabled(cfg):
        raise ValueError("Conversion calibration Prefix policy disagrees with config")
    provenance = (
        _validate_aware_training_provenance(cfg, layout, prefix, manifest_path)
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
        raise ValueError(
            f"{path} uses a legacy conversion schema; "
            f"format v{CONVERSION_METADATA_FORMAT_VERSION} is required"
        )
    validate_temporal_policy(metadata, context=str(path))
    prefix, bundle, manifest_path, training_provenance = _source_bundle(cfg, layout)
    reused = conversion_reuses_ann_training_artifacts(cfg)
    ann_config = layout.ann_checkpoint_dir / "config.json"
    rotation_enabled = bool(cfg["rotation"]["enabled"])
    rotation_path = layout.rotation_dir / "rotation_state.pt"
    expected = {
        "deployment_neuron": neuron,
        "full_temporal_steps": ({"phase": int(cfg["phase"]["T"]), "mtn": int(cfg["mtn"]["T"]), "gif": GIF_LOCAL_STEPS}[neuron]),
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
        "source_ann_common_clip_enabled": training_common_clip_enabled(cfg),
        "ordinary_gif_local_decomposition_steps": GIF_LOCAL_STEPS,
        "calibration_group_size": int(cfg["calibration"]["group_size"]),
        "calibration_num_samples": int(cfg["calibration"]["num_samples"]),
        "source_ann_training_phase_T": training_provenance.get("ann_training_phase_T"),
        "source_ann_training_mtn_T": training_provenance.get("ann_training_mtn_T"),
        "deployment_phase_T": int(cfg["phase"]["T"]) if neuron == "phase" else None,
        "deployment_mtn_T": int(cfg["mtn"]["T"]) if neuron == "mtn" else None,
        "deployment_mtn_K": int(cfg["mtn"]["K"]) if neuron == "mtn" else None,
        "calibration_grouping_policy": CALIBRATION_GROUPING_POLICY,
        "statistics_format_version": STATISTICS_FORMAT_VERSION,
        "softmax_site5_grouping_policy": SOFTMAX_SITE5_GROUPING_POLICY,
        "softmax_site5_gif_policy": SOFTMAX_SITE5_GIF_POLICY,
        "softmax_site5_clip_policy": SOFTMAX_SITE5_CLIP_POLICY,
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
    reused = conversion_reuses_ann_training_artifacts(cfg)
    prefix, validation, manifest_path, training_provenance = _source_bundle(cfg, layout)
    controller = SiteController(
        site_root=layout.conversion_site_dir,
        phase_T=int(cfg["phase"]["T"]),
        mtn_T=int(cfg["mtn"]["T"]),
        mtn_K=int(cfg["mtn"]["K"]),
        mtn_threshold_factor=float(cfg["mtn"]["threshold_factor"]),
    )
    steps = controller.set_deployment(
        neuron,
        clip_bundle_policy="forbid_all",
    )
    output = layout.snn_conversion_dir(neuron)
    output.mkdir(parents=True, exist_ok=True)
    rotation_enabled = bool(cfg["rotation"]["enabled"])
    rotation_path = layout.rotation_dir / "rotation_state.pt"
    if rotation_enabled and not rotation_path.exists():
        raise FileNotFoundError(rotation_path)
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
        "source_ann_training_phase_T": training_provenance.get("ann_training_phase_T"),
        "source_ann_training_mtn_T": training_provenance.get("ann_training_mtn_T"),
        "deployment_phase_T": int(cfg["phase"]["T"]) if neuron == "phase" else None,
        "deployment_mtn_T": int(cfg["mtn"]["T"]) if neuron == "mtn" else None,
        "deployment_mtn_K": int(cfg["mtn"]["K"]) if neuron == "mtn" else None,
        "ordinary_gif_local_decomposition_steps": GIF_LOCAL_STEPS,
        "rotation_enabled": rotation_enabled,
        "expected_num_hidden_layers": expected_num_hidden_layers,
        "prefix_enabled": conversion_prefix_enabled(cfg),
        "prefix_token_ids": prefix["prefix_token_ids"],
        "rotation_state_sha256": sha256_file(rotation_path) if rotation_enabled else None,
        "prefix_state_sha256": prefix["prefix_state_sha256"],
        "prefix_kv_sha256": prefix["prefix_kv_sha256"],
        "post_finetuning_recalibration": not reused,
        "snn_clip_applied": False,
        "source_ann_common_clip_enabled": training_common_clip_enabled(cfg),
        "calibration_group_size": int(cfg["calibration"]["group_size"]),
        "calibration_num_samples": int(cfg["calibration"]["num_samples"]),
        "calibration_grouping_policy": CALIBRATION_GROUPING_POLICY,
        "statistics_format_version": STATISTICS_FORMAT_VERSION,
        "softmax_site5_grouping_policy": SOFTMAX_SITE5_GROUPING_POLICY,
        "softmax_site5_gif_policy": SOFTMAX_SITE5_GIF_POLICY,
        "softmax_site5_clip_policy": SOFTMAX_SITE5_CLIP_POLICY,
        "prefix_root": prefix["prefix_root"],
        "calibration_validation": validation,
        "training_artifact_provenance": training_provenance,
        **temporal_policy_metadata(),
        **topology_metadata(),
    }
    write_json(output / "conversion_metadata.json", metadata)
    return metadata
