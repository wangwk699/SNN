from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from snn2.artifacts import sha256_file
from snn2.calibration import materialize_calibration_states
from snn2.conversion import create_conversion, validate_conversion_metadata
from snn2.controller import SiteController
from snn2.sites import SITE_IDS, SITE_NAMES
from snn2.temporal_ops import (
    CALIBRATION_GROUPING_POLICY,
    CONVERSION_METADATA_FORMAT_VERSION,
    GIF_LOCAL_STEPS,
    SOFTMAX_SITE5_CLIP_POLICY,
    SOFTMAX_SITE5_GIF_POLICY,
    SOFTMAX_SITE5_GROUPING_POLICY,
    STATISTICS_FORMAT_VERSION,
    temporal_policy_metadata,
)
from snn2.phase_statistics import (
    PHASE_TAU_ACCUMULATOR_DTYPE, PHASE_TAU_CALIBRATION,
    PHASE_TAU_CHANNEL_POLICY, PHASE_TAU_EMA_FACTOR, PHASE_TAU_REDUCTION_POLICY,
)


class _Layout:
    def __init__(self, root):
        self.root = root
        self.post_finetuning_site_dir = root / "sites"
        self.post_finetuning_prefix_dir = root / "prefix"
        self.pre_finetuning_prefix_dir = root / "pre_prefix"
        self.ann_training_site_dir = root / "ann_sites"
        self.ann_dir = root / "ann"
        self.conversion_prefix_dir = self.post_finetuning_prefix_dir
        self.conversion_site_dir = self.post_finetuning_site_dir
        self.ann_checkpoint_dir = root / "ann" / "final"
        self.rotation_dir = root / "rotation"

    def snn_conversion_dir(self, neuron):
        return self.root / "snn" / neuron / "conversion"


def _statistics(site_index=1):
    if site_index in {2, 3, 4}:
        shape, layout, heads, width, channels = (1, 4), "attention_head", 1, 4, 4
    elif site_index == 5:
        shape, layout, heads, width, channels = (1,), "attention_softmax", 1, None, 1
    else:
        shape, layout, heads, width, channels = (4,), "last_dim", None, None, 4
    saliency_shape = shape
    roles = (
        ("q", "k", "v") if site_index == 1 else
        (("gate", "up") if site_index == 7 else
         (("default",) if site_index in {3, 4, 6, 10} else ()))
    )
    return {
        "format_version": STATISTICS_FORMAT_VERSION, "site_index": site_index,
        "layout_kind": layout, "num_heads": heads, "channels_per_head": width,
        "channels": channels, "value_min": torch.full(shape, -1.0),
        "value_max": torch.full(shape, 1.0),
        "saliency_row_count_by_role": {role: torch.ones(saliency_shape, dtype=torch.long) for role in roles},
        "saliency_sum_by_role": {role: torch.zeros(saliency_shape, dtype=torch.float64 if site_index in {3, 4} else torch.float32) for role in roles},
        "saliency_rule_by_role": {role: ("spikellm_qk_k_fp64" if site_index == 3 else ("spikellm_pv_v_fp64" if site_index == 4 else "spikellm_linear_fp32")) for role in roles},
        "saliency_accumulator_dtype_by_role": {role: ("float64" if site_index in {3, 4} else "float32") for role in roles},
        "phase_ema_abs_max": torch.ones(shape), "phase_ema_updates": torch.ones(shape, dtype=torch.long),
        "phase_tau_calibration": PHASE_TAU_CALIBRATION,
        "phase_tau_ema_factor": PHASE_TAU_EMA_FACTOR,
        "phase_tau_accumulator_dtype": PHASE_TAU_ACCUMULATOR_DTYPE,
        "phase_tau_channel_policy": PHASE_TAU_CHANNEL_POLICY,
        "phase_tau_reduction_policy": PHASE_TAU_REDUCTION_POLICY,
    }


def _cfg(rotation_enabled=False, *, ann_mode="vanilla", use_post=True):
    return {
        "experiment": {"name": "test_conversion", "ann_mode": ann_mode},
        "ann_finetuning": {"mode": ann_mode},
        "rotation": {"enabled": rotation_enabled},
        "prefix": {"enabled": ann_mode != "vanilla"},
        "ann_training": {"prefix_enabled": ann_mode != "vanilla"},
        "post_finetuning": {"prefix_enabled": ann_mode != "vanilla"},
        "conversion": {"use_post_finetuning_artifacts": use_post},
        "replacement": {"common_clip_enabled": False},
        "calibration": {"group_size": -1, "num_samples": 128, "expected_sites_per_layer": 10},
        "phase": {"T": 4, "base": 2.0, "surrogate_slope": 1.0},
        "gif": {"base_bits": 4, "add_bits": 1, "low_ratio": 0.5},
        "mtn": {"T": 4, "K": 6, "threshold_factor": 0.75},
    }


def _prepare(tmp_path, *, rotation_enabled=False):
    layout = _Layout(tmp_path)
    for index in SITE_IDS:
        site = layout.post_finetuning_site_dir / "layer_000" / f"site_{index:02d}_{SITE_NAMES[index]}"
        site.mkdir(parents=True)
        torch.save(_statistics(index), site / "statistics.pt")
    global_directory = layout.post_finetuning_site_dir / "_global" / "final_rmsnorm"
    global_directory.mkdir(parents=True)
    torch.save(_statistics(None), global_directory / "statistics.pt")
    materialize_calibration_states(
        layout.post_finetuning_site_dir,
        _cfg(rotation_enabled),
        {
            "purpose": "post_finetuning_conversion_calibration",
            "eligible_for_ann_training": False,
            "eligible_for_conversion": True,
            "conversion_reuse_policy": "final_ann_only",
            "post_finetuning_recalibration": True,
            "state_profile": "stage_a_common_states",
            "common_clip_required": False,
            "common_clip_generated": False,
            "common_clip_application_control": "replacement.common_clip_enabled",
            "prefix_enabled": False,
        },
        expected_num_hidden_layers=1,
    )
    layout.ann_checkpoint_dir.mkdir(parents=True)
    ann_config = layout.ann_checkpoint_dir / "config.json"
    ann_config.write_text('{"num_hidden_layers": 1}' + '\n', encoding="utf-8")
    manifest_path = layout.post_finetuning_site_dir / "calibration_state_manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload.update({
        "source_model_stage": "final_ann_checkpoint",
        "source_ann_mode": "vanilla",
        "source_ann_checkpoint": str(layout.ann_checkpoint_dir.resolve()),
        "source_ann_config_sha256": sha256_file(ann_config),
    })
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    output = layout.snn_conversion_dir("gif")
    output.mkdir(parents=True)
    rotation_path = layout.rotation_dir / "rotation_state.pt"
    if rotation_enabled:
        rotation_path.parent.mkdir(parents=True)
        rotation_path.write_bytes(b"rotation-v1")
    manifest = layout.post_finetuning_site_dir / "calibration_state_manifest.json"
    final_norm = json.loads(manifest.read_text(encoding="utf-8"))["global_states"]["final_rmsnorm"]
    metadata = {
        "format_version": CONVERSION_METADATA_FORMAT_VERSION,
        "deployment_neuron": "gif",
        "full_temporal_steps": GIF_LOCAL_STEPS,
        "source_ann_checkpoint": str(layout.ann_checkpoint_dir.resolve()),
        "source_ann_config_sha256": sha256_file(ann_config),
        "post_finetuning_recalibration": True,
        "rotation_enabled": rotation_enabled,
        "rotation_state_sha256": (
            sha256_file(rotation_path) if rotation_enabled else None
        ),
        "expected_num_hidden_layers": 1,
        "prefix_enabled": False,
        "prefix_token_ids": [],
        "prefix_state_sha256": None,
        "prefix_kv_sha256": None,
        "calibration_root": str(layout.post_finetuning_site_dir.resolve()),
        "calibration_state_manifest_sha256": sha256_file(manifest),
        "calibration_source_stage": "post_finetuning",
        "prefix_source_stage": "post_finetuning",
        "reused_ann_training_artifacts": False,
        "use_post_finetuning_artifacts": True,
        "snn_clip_applied": False,
        "source_ann_common_clip_enabled": False,
        "ordinary_gif_local_decomposition_steps": GIF_LOCAL_STEPS,
        "calibration_group_size": -1,
        "calibration_num_samples": 128,
        "source_ann_training_phase_T": None,
        "source_ann_training_mtn_T": None,
        "deployment_phase_T": None,
        "deployment_mtn_T": None,
        "deployment_mtn_K": None,
        "calibration_grouping_policy": CALIBRATION_GROUPING_POLICY,
        "statistics_format_version": STATISTICS_FORMAT_VERSION,
        "softmax_site5_grouping_policy": SOFTMAX_SITE5_GROUPING_POLICY,
        "softmax_site5_gif_policy": SOFTMAX_SITE5_GIF_POLICY,
        "softmax_site5_clip_policy": SOFTMAX_SITE5_CLIP_POLICY,
        "final_norm_phase_state_sha256": final_norm["phase_state_sha256"],
        "final_norm_mtn_state_sha256": final_norm["mtn_state_sha256"],
        "final_norm_gif_state_present": False,
        "final_norm_clip_state_present": False,
        **temporal_policy_metadata(),
    }
    path = output / "conversion_metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return layout, path


def test_conversion_metadata_current_schema_is_accepted(tmp_path):
    layout, _ = _prepare(tmp_path)
    metadata = validate_conversion_metadata(_cfg(), layout, "gif")
    assert metadata["ordinary_gif_high_qmax"] == 30
    assert metadata["source_ann_common_clip_enabled"] is False
    assert metadata["snn_clip_applied"] is False
    assert metadata["gif_saliency_selection_policy"] == "spikellm_global_per_channel_threshold_leq"
    assert metadata["gif_saliency_tie_policy"] == "mask_low_equals_score_le_threshold"
    assert metadata["gif_linear_saliency_dtype"] == "float32"
    assert metadata["gif_matmul_saliency_dtype"] == "float64"


def test_conversion_rejects_source_ann_common_clip_mismatch(monkeypatch, tmp_path):
    layout, _ = _prepare(tmp_path)
    monkeypatch.setattr(
        "snn2.conversion.training_common_clip_enabled", lambda cfg: True
    )
    with pytest.raises(ValueError, match="source_ann_common_clip_enabled"):
        validate_conversion_metadata(_cfg(), layout, "gif")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("format_version", 3),
        ("ordinary_gif_high_qmax", 31),
        (
            "softmax_site5_gif_policy",
            "fixed_range_u16_quantized_cumulative_difference",
        ),
        ("full_temporal_steps", 3),
        ("snn_clip_applied", True),
    ],
)
def test_conversion_metadata_rejects_legacy_or_mismatched_policy(tmp_path, key, value):
    layout, path = _prepare(tmp_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata[key] = value
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_conversion_metadata(_cfg(), layout, "gif")


def test_conversion_rotation_hash_matches_current_file(tmp_path):
    layout, _ = _prepare(tmp_path, rotation_enabled=True)
    metadata = validate_conversion_metadata(_cfg(True), layout, "gif")
    assert metadata["rotation_state_sha256"] == sha256_file(
        layout.rotation_dir / "rotation_state.pt"
    )


def test_conversion_rejects_modified_rotation_file(tmp_path):
    layout, _ = _prepare(tmp_path, rotation_enabled=True)
    (layout.rotation_dir / "rotation_state.pt").write_bytes(b"rotation-v2")
    with pytest.raises(ValueError, match="rotation_state_sha256"):
        validate_conversion_metadata(_cfg(True), layout, "gif")


def test_conversion_rejects_tampered_rotation_hash(tmp_path):
    layout, path = _prepare(tmp_path, rotation_enabled=True)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["rotation_state_sha256"] = "0" * 64
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="rotation_state_sha256"):
        validate_conversion_metadata(_cfg(True), layout, "gif")


def test_conversion_disabled_rotation_requires_null_hash(tmp_path):
    layout, path = _prepare(tmp_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["rotation_state_sha256"] = "0" * 64
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="rotation_state_sha256"):
        validate_conversion_metadata(_cfg(), layout, "gif")


def test_conversion_rejects_ann_manifest_layer_mismatch(tmp_path):
    layout, _ = _prepare(tmp_path)
    (layout.ann_checkpoint_dir / "config.json").write_text(
        json.dumps({"num_hidden_layers": 2}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="ANN config num_hidden_layers"):
        validate_conversion_metadata(_cfg(), layout, "gif")


def test_conversion_rejects_stale_post_finetuning_clip_state(tmp_path):
    layout, _ = _prepare(tmp_path)
    stale = next(layout.post_finetuning_site_dir.glob("layer_*/site_*"))
    torch.save({}, stale / "clip_state.pt")
    with pytest.raises(ValueError, match="Conversion Stage A calibration must be clip-free"):
        validate_conversion_metadata(_cfg(), layout, "gif")


@pytest.mark.parametrize(
    ("ann_mode", "expected_reused", "expected_clip_policy"),
    [
        ("phase_aware", False, "forbid_all"),
        ("gif_aware", False, "forbid_all"),
        ("vanilla", False, "forbid_all"),
        ("unaware", False, "forbid_all"),
    ],
)
def test_create_conversion_selects_clip_policy_and_records_reuse(
    monkeypatch, tmp_path, ann_mode, expected_reused, expected_clip_policy
):
    layout, _ = _prepare(tmp_path)
    cfg = _cfg()
    cfg["experiment"]["ann_mode"] = ann_mode
    cfg["prefix"] = {"enabled": ann_mode != "vanilla"}
    cfg["ann_training"] = {"prefix_enabled": ann_mode != "vanilla"}
    cfg["replacement"] = {"common_clip_enabled": False}
    cfg["post_finetuning"]["prefix_enabled"] = False
    manifest = (
        layout.post_finetuning_site_dir / "calibration_state_manifest.json"
    )
    captured = {}

    monkeypatch.setattr(
        "snn2.conversion._source_bundle",
        lambda _cfg, _layout: (
            {
                "prefix_source_stage": "pre_finetuning"
                if expected_reused
                else "post_finetuning",
                "prefix_token_ids": [],
                "prefix_state_sha256": None,
                "prefix_kv_sha256": None,
                "prefix_root": None,
            },
            {"temporal_steps": {"phase": 4}},
            manifest,
            {},
        ),
    )

    def fake_set_deployment(self, neuron, *, clip_bundle_policy):
        captured["neuron"] = neuron
        captured["clip_bundle_policy"] = clip_bundle_policy
        return 4

    monkeypatch.setattr(SiteController, "set_deployment", fake_set_deployment)

    metadata = create_conversion(cfg, layout, "phase")

    assert captured == {
        "neuron": "phase",
        "clip_bundle_policy": expected_clip_policy,
    }
    assert metadata["deployment_neuron"] == "phase"
    assert metadata["reused_ann_training_artifacts"] is expected_reused
    assert (
        layout.snn_conversion_dir("phase") / "conversion_metadata.json"
    ).exists()



class _SelectorLayout:
    def __init__(self, root: Path, *, use_post: bool):
        self.root = root
        self._use_post = use_post
        self.ann_dir = root / "ann"
        self.ann_checkpoint_dir = self.ann_dir / "final"
        self.rotation_dir = root / "rotation"
        self.calibration_data_manifest_path = root / "data" / "calibration_manifest.json"
        self.ann_training_prefix_dir = root / "shared" / "pre_prefix" / "num_samples_128"
        self.post_finetuning_prefix_dir = root / "post" / "prefix" / "num_samples_128"
        self.ann_training_site_dir = root / "shared" / "ann_training_sites"
        self.post_finetuning_site_dir = root / "post" / "sites"

    @property
    def conversion_prefix_dir(self):
        return self.post_finetuning_prefix_dir if self._use_post else self.ann_training_prefix_dir

    @property
    def conversion_site_dir(self):
        return self.post_finetuning_site_dir if self._use_post else self.ann_training_site_dir

    def snn_conversion_dir(self, neuron):
        return self.root / "snn" / str(self._use_post).lower() / neuron / "conversion"


def _write_prefix_fixture(layout, directory):
    layout.calibration_data_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    layout.calibration_data_manifest_path.write_text(json.dumps({"num_samples": 128}), encoding="utf-8")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "prefix_state.json").write_text(json.dumps({
        "prefix_token_ids": [],
        "discovery_num_samples": 128,
        "discovery_data_source": "stage_a_calibration_selection",
        "discovery_manifest_path": str(layout.calibration_data_manifest_path.resolve()),
        "discovery_manifest_sha256": sha256_file(layout.calibration_data_manifest_path),
    }), encoding="utf-8")


def _write_stage_a_fixture(root, cfg, *, post, ann_checkpoint):
    for index in SITE_IDS:
        directory = root / "layer_000" / f"site_{index:02d}_{SITE_NAMES[index]}"
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(_statistics(index), directory / "statistics.pt")
    global_directory = root / "_global" / "final_rmsnorm"
    global_directory.mkdir(parents=True, exist_ok=True)
    torch.save(_statistics(None), global_directory / "statistics.pt")
    metadata = {
        "purpose": "post_finetuning_conversion_calibration" if post else "ann_training_calibration",
        "eligible_for_ann_training": not post,
        "eligible_for_conversion": True,
        "conversion_reuse_policy": "final_ann_only" if post else "non_vanilla_when_selected",
        "post_finetuning_recalibration": post,
        "state_profile": "stage_a_common_states",
        "common_clip_required": False,
        "common_clip_generated": False,
        "common_clip_application_control": "replacement.common_clip_enabled",
        "prefix_enabled": True,
        "source_model_stage": "final_ann_checkpoint" if post else "rotated_fused_base",
        "source_ann_mode": cfg["experiment"]["ann_mode"] if post else None,
        "source_ann_checkpoint": str(ann_checkpoint.resolve()) if post else None,
        "source_ann_config_sha256": sha256_file(ann_checkpoint / "config.json") if post else None,
    }
    materialize_calibration_states(root, cfg, metadata, expected_num_hidden_layers=1)
    return root / "calibration_state_manifest.json"


def _prepare_selector_fixture(tmp_path, *, ann_mode, use_post, include_training_result=False):
    cfg = _cfg(True, ann_mode=ann_mode, use_post=use_post)
    layout = _SelectorLayout(tmp_path, use_post=use_post)
    layout.ann_checkpoint_dir.mkdir(parents=True)
    (layout.ann_checkpoint_dir / "config.json").write_text(json.dumps({"num_hidden_layers": 1}), encoding="utf-8")
    layout.rotation_dir.mkdir(parents=True)
    (layout.rotation_dir / "rotation_state.pt").write_bytes(b"rotation")
    _write_prefix_fixture(layout, layout.ann_training_prefix_dir)
    _write_prefix_fixture(layout, layout.post_finetuning_prefix_dir)
    ann_manifest = _write_stage_a_fixture(
        layout.ann_training_site_dir, cfg, post=False, ann_checkpoint=layout.ann_checkpoint_dir
    )
    post_manifest = _write_stage_a_fixture(
        layout.post_finetuning_site_dir, cfg, post=True, ann_checkpoint=layout.ann_checkpoint_dir
    )
    if include_training_result:
        profile_root = tmp_path / "shared" / "clip_profile"
        profile_root.mkdir(parents=True)
        profile_manifest = profile_root / "clip_profile_manifest.json"
        profile_manifest.write_text("{}", encoding="utf-8")
        prefix_state = layout.ann_training_prefix_dir / "prefix_state.json"
        result = {
            "ann_training_prefix_root": str(layout.ann_training_prefix_dir.resolve()),
            "ann_training_prefix_state_sha256": sha256_file(prefix_state),
            "ann_training_prefix_kv_sha256": None,
            "ann_training_prefix_num_samples": 128,
            "ann_training_prefix_discovery_manifest_sha256": json.loads(prefix_state.read_text(encoding="utf-8"))["discovery_manifest_sha256"],
            "ann_training_prefix_token_ids": [],
            "ann_training_calibration_root": str(layout.ann_training_site_dir.resolve()),
            "ann_training_calibration_manifest_sha256": sha256_file(ann_manifest),
            "ann_training_calibration_group_size": -1,
            "ann_training_calibration_grouping_policy": CALIBRATION_GROUPING_POLICY,
            "statistics_format_version": STATISTICS_FORMAT_VERSION,
            "ann_training_phase_T": 4,
            "ann_training_mtn_T": 4,
            "ann_training_calibration_num_samples": 128,
            "ann_training_clip_profile_root": str(profile_root.resolve()),
            "ann_training_clip_profile_manifest_sha256": sha256_file(profile_manifest),
        }
        (layout.ann_dir / "training_result.json").write_text(json.dumps(result), encoding="utf-8")
    return cfg, layout, ann_manifest, post_manifest


def test_unaware_pre_bundle_conversion_is_end_to_end_and_skips_aware_provenance(monkeypatch, tmp_path):
    cfg, layout, ann_manifest, _ = _prepare_selector_fixture(
        tmp_path, ann_mode="unaware", use_post=False
    )
    monkeypatch.setattr(
        "snn2.conversion._validate_aware_training_provenance",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unaware must not require aware training provenance")),
    )
    metadata = create_conversion(cfg, layout, "phase")
    assert validate_conversion_metadata(cfg, layout, "phase") == metadata
    assert metadata["use_post_finetuning_artifacts"] is False
    assert metadata["prefix_source_stage"] == "pre_finetuning"
    assert metadata["calibration_source_stage"] == "ann_training"
    assert metadata["reused_ann_training_artifacts"] is True
    assert metadata["post_finetuning_recalibration"] is False
    manifest = json.loads(ann_manifest.read_text(encoding="utf-8"))
    assert manifest["source_model_stage"] == "rotated_fused_base"
    assert manifest["source_ann_mode"] is None
    assert manifest["source_ann_checkpoint"] is None


@pytest.mark.parametrize(
    ("ann_mode", "neuron"),
    [("phase_aware", "phase"), ("gif_aware", "gif")],
)
def test_aware_pre_bundle_conversion_validates_frozen_provenance(tmp_path, ann_mode, neuron):
    cfg, layout, ann_manifest, _ = _prepare_selector_fixture(
        tmp_path, ann_mode=ann_mode, use_post=False, include_training_result=True
    )
    metadata = create_conversion(cfg, layout, neuron)
    assert validate_conversion_metadata(cfg, layout, neuron) == metadata
    assert metadata["reused_ann_training_artifacts"] is True
    assert metadata["post_finetuning_recalibration"] is False
    assert metadata["source_ann_training_phase_T"] == 4
    assert metadata["source_ann_training_mtn_T"] == 4
    result_path = layout.ann_dir / "training_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["ann_training_calibration_manifest_sha256"] = "0" * 64
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="Aware conversion artifacts differ from those fixed during ANN training"):
        validate_conversion_metadata(cfg, layout, neuron)


def test_phase_aware_post_bundle_conversion_does_not_require_aware_training_provenance(monkeypatch, tmp_path):
    cfg, layout, ann_manifest, post_manifest = _prepare_selector_fixture(
        tmp_path, ann_mode="phase_aware", use_post=True
    )
    monkeypatch.setattr(
        "snn2.conversion._validate_aware_training_provenance",
        lambda *_args: (_ for _ in ()).throw(AssertionError("post bundle must not require aware training provenance")),
    )
    metadata = create_conversion(cfg, layout, "phase")
    assert validate_conversion_metadata(cfg, layout, "phase") == metadata
    assert metadata["use_post_finetuning_artifacts"] is True
    assert metadata["prefix_source_stage"] == "post_finetuning"
    assert metadata["calibration_source_stage"] == "post_finetuning"
    assert metadata["reused_ann_training_artifacts"] is False
    assert metadata["post_finetuning_recalibration"] is True
    manifest = json.loads(post_manifest.read_text(encoding="utf-8"))
    assert sha256_file(post_manifest) != sha256_file(ann_manifest)
    assert manifest["source_model_stage"] == "final_ann_checkpoint"
    assert manifest["source_ann_mode"] == "phase_aware"
    assert manifest["source_ann_checkpoint"] == str(layout.ann_checkpoint_dir.resolve())
