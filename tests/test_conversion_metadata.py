from __future__ import annotations

import json

import pytest
import torch

from snn2.artifacts import sha256_file
from snn2.calibration import materialize_calibration_states
from tests.calibration_fixtures import write_stage_a_statistics
from snn2.conversion import create_conversion, validate_conversion_metadata
from snn2.controller import SiteController
from snn2.state_validation import deployment_state_fingerprint
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
        self.post_finetuning_statistics_dir = root / "statistics"
        self.post_finetuning_state_dir = root / "states"
        self.post_finetuning_site_dir = self.post_finetuning_state_dir
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
    if site_index in {2, 3, 4, 6}:
        shape, layout, heads, width, channels = (1, 4), "attention_head", 1, 4, 4
    elif site_index == 5:
        shape, layout, heads, width, channels = (1,), "attention_softmax", 1, None, 1
    else:
        shape, layout, heads, width, channels = (4,), "last_dim", None, None, 4
    saliency_shape = (0,) if site_index == 5 else shape
    return {
        "format_version": STATISTICS_FORMAT_VERSION, "site_index": site_index,
        "layout_kind": layout, "num_heads": heads, "channels_per_head": width,
        "channels": channels, "value_min": torch.full(shape, -1.0),
        "value_max": torch.full(shape, 1.0),
        "saliency_row_count": torch.ones(saliency_shape, dtype=torch.long),
        "saliency_sum": torch.zeros(saliency_shape, dtype=torch.float64),
        "phase_ema_abs_max": torch.ones(shape), "phase_ema_updates": torch.ones(shape, dtype=torch.long),
        "phase_tau_calibration": PHASE_TAU_CALIBRATION,
        "phase_tau_ema_factor": PHASE_TAU_EMA_FACTOR,
        "phase_tau_accumulator_dtype": PHASE_TAU_ACCUMULATOR_DTYPE,
        "phase_tau_channel_policy": PHASE_TAU_CHANNEL_POLICY,
        "phase_tau_reduction_policy": PHASE_TAU_REDUCTION_POLICY,
    }


def _cfg(rotation_enabled=False):
    return {
        "experiment": {"name": "test_conversion", "ann_mode": "vanilla"},
        "ann_finetuning": {"mode": "vanilla"},
        "rotation": {"enabled": rotation_enabled},
        "post_finetuning": {"prefix_enabled": False},
        "calibration": {"group_size": -1, "num_samples": 128, "seed": 42, "expected_sites_per_layer": 10},
        "phase": {"T": 4, "base": 2.0, "surrogate_slope": 1.0},
        "gif": {"base_bits": 4, "add_bits": 1, "low_ratio": 0.5, "salient_ratio": 0.5},
        "mtn": {"T": 4, "K": 6, "threshold_factor": 0.75},
        "replacement": {"common_clip_enabled": False},
        "prefix": {"enabled": False},
    }


def _prepare(tmp_path, *, rotation_enabled=False, neuron="gif"):
    layout = _Layout(tmp_path)
    cfg = _cfg(rotation_enabled)
    write_stage_a_statistics(layout.post_finetuning_statistics_dir, cfg, _statistics)
    metadata_base = {
        "purpose": "post_finetuning_conversion_calibration", "eligible_for_ann_training": False,
        "eligible_for_conversion": True, "conversion_reuse_policy": "final_ann_only",
        "post_finetuning_recalibration": True, "state_profile": "snn_conversion_without_clip",
        "common_clip_required": False, "common_clip_generated": False,
        "common_clip_application_control": "replacement.common_clip_enabled", "prefix_enabled": False,
        "source_model_stage": None, "source_ann_mode": None, "source_ann_checkpoint": None,
        "source_ann_config_sha256": None, "prefix_state_sha256": None, "prefix_kv_sha256": None,
        "rotation_enabled": False, "rotation_state_sha256": None, "calibration_data_manifest_sha256": None,
        "calibration_grouping_policy": CALIBRATION_GROUPING_POLICY,
    }
    materialize_calibration_states(layout.post_finetuning_statistics_dir, layout.post_finetuning_state_dir, cfg, metadata_base, include_clip=False, expected_num_hidden_layers=1)
    layout.ann_checkpoint_dir.mkdir(parents=True)
    ann_config = layout.ann_checkpoint_dir / "config.json"
    ann_config.write_text('{"num_hidden_layers": 1}\n', encoding="utf-8")
    output = layout.snn_conversion_dir(neuron); output.mkdir(parents=True)
    rotation_path = layout.rotation_dir / "rotation_state.pt"
    if rotation_enabled:
        rotation_path.parent.mkdir(parents=True); rotation_path.write_bytes(b"rotation-v1")
    manifest = layout.post_finetuning_site_dir / "calibration_state_manifest.json"
    state_manifest = json.loads(manifest.read_text())
    fingerprint = deployment_state_fingerprint(layout.post_finetuning_site_dir, neuron)
    deployment_parameters = (
        {"phase_T": int(cfg["phase"]["T"])} if neuron == "phase" else
        ({"mtn_T": int(cfg["mtn"]["T"]), "mtn_K": int(cfg["mtn"]["K"])} if neuron == "mtn" else
         {"gif_low_ratio": float(cfg["gif"]["low_ratio"]), "gif_salient_ratio": float(cfg["gif"]["salient_ratio"])})
    )
    temporal_steps = {"phase": int(cfg["phase"]["T"]), "mtn": int(cfg["mtn"]["T"]), "gif": GIF_LOCAL_STEPS}[neuron]
    metadata = {
        "format_version": CONVERSION_METADATA_FORMAT_VERSION, "deployment_neuron": neuron,
        "deployment_state_kinds": [neuron], "deployment_state_fingerprint_sha256": fingerprint["sha256"],
        "deployment_state_file_hashes": fingerprint["file_hashes"],
        "deployment_parameters": deployment_parameters,
        "full_temporal_steps": temporal_steps, "source_ann_checkpoint": str(layout.ann_checkpoint_dir.resolve()),
        "source_ann_config_sha256": sha256_file(ann_config), "post_finetuning_recalibration": True,
        "rotation_enabled": rotation_enabled, "rotation_state_sha256": sha256_file(rotation_path) if rotation_enabled else None,
        "expected_num_hidden_layers": 1, "prefix_enabled": False, "prefix_token_ids": [],
        "prefix_state_sha256": None, "prefix_kv_sha256": None,
        "calibration_root": str(layout.post_finetuning_site_dir.resolve()),
        "calibration_state_manifest_sha256": sha256_file(manifest),
        "source_statistics_manifest_path": state_manifest["source_statistics_manifest_path"],
        "source_statistics_manifest_sha256": state_manifest["source_statistics_manifest_sha256"],
        "calibration_source_stage": "post_finetuning", "prefix_source_stage": "post_finetuning",
        "reused_ann_training_artifacts": False, "snn_clip_applied": False,
        "source_ann_common_clip_enabled": False, "ordinary_gif_local_decomposition_steps": GIF_LOCAL_STEPS,
        "calibration_group_size": -1, "calibration_num_samples": 128,
        "calibration_grouping_policy": CALIBRATION_GROUPING_POLICY,
        "statistics_format_version": STATISTICS_FORMAT_VERSION,
        "softmax_site5_grouping_policy": SOFTMAX_SITE5_GROUPING_POLICY,
        "softmax_site5_gif_policy": SOFTMAX_SITE5_GIF_POLICY,
        "softmax_site5_clip_policy": SOFTMAX_SITE5_CLIP_POLICY,
        **temporal_policy_metadata(),
    }
    path = output / "conversion_metadata.json"; path.write_text(json.dumps(metadata), encoding="utf-8")
    return layout, path


def test_conversion_metadata_v10_is_accepted(tmp_path):
    layout, _ = _prepare(tmp_path)
    metadata = validate_conversion_metadata(_cfg(), layout, "gif")
    assert metadata["ordinary_gif_high_qmax"] == 30
    assert metadata["source_ann_common_clip_enabled"] is False
    assert metadata["snn_clip_applied"] is False


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
    with pytest.raises(ValueError, match="clip-free"):
        validate_conversion_metadata(_cfg(), layout, "gif")


@pytest.mark.parametrize(
    ("ann_mode", "expected_reused", "expected_clip_policy"),
    [
        ("phase_aware", True, "allow_eligible"),
        ("gif_aware", True, "allow_eligible"),
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


def _rematerialize_post_finetuning(layout, cfg):
    metadata = {
        "purpose": "post_finetuning_conversion_calibration",
        "eligible_for_ann_training": False, "eligible_for_conversion": True,
        "conversion_reuse_policy": "final_ann_only",
        "post_finetuning_recalibration": True,
        "state_profile": "snn_conversion_without_clip",
        "common_clip_required": False, "common_clip_generated": False,
        "common_clip_application_control": "replacement.common_clip_enabled",
        "source_model_stage": None, "source_ann_mode": None, "source_ann_checkpoint": None, "source_ann_config_sha256": None,
        "prefix_enabled": False, "prefix_state_sha256": None, "prefix_kv_sha256": None,
        "rotation_enabled": False, "rotation_state_sha256": None,
        "calibration_data_manifest_sha256": None,
        "calibration_grouping_policy": CALIBRATION_GROUPING_POLICY,
    }
    materialize_calibration_states(
        layout.post_finetuning_statistics_dir, layout.post_finetuning_state_dir,
        cfg, metadata, include_clip=False, expected_num_hidden_layers=1,
    )


def test_mtn_k_change_preserves_phase_conversion_identity(tmp_path):
    layout, _ = _prepare(tmp_path, neuron="phase")
    before = deployment_state_fingerprint(layout.post_finetuning_state_dir, "phase")["sha256"]
    changed = _cfg(); changed["mtn"]["K"] = 8
    _rematerialize_post_finetuning(layout, changed)
    assert deployment_state_fingerprint(layout.post_finetuning_state_dir, "phase")["sha256"] == before
    assert validate_conversion_metadata(changed, layout, "phase")["deployment_neuron"] == "phase"


def test_mtn_k_change_invalidates_mtn_conversion_identity(tmp_path):
    layout, _ = _prepare(tmp_path, neuron="mtn")
    before = deployment_state_fingerprint(layout.post_finetuning_state_dir, "mtn")["sha256"]
    changed = _cfg(); changed["mtn"]["K"] = 8
    _rematerialize_post_finetuning(layout, changed)
    assert deployment_state_fingerprint(layout.post_finetuning_state_dir, "mtn")["sha256"] != before
    with pytest.raises(ValueError, match="deployment_state_fingerprint_sha256|deployment_parameters"):
        validate_conversion_metadata(changed, layout, "mtn")


def test_gif_ratio_change_preserves_phase_and_mtn_conversion_identities(tmp_path):
    phase_layout, _ = _prepare(tmp_path / "phase", neuron="phase")
    mtn_layout, _ = _prepare(tmp_path / "mtn", neuron="mtn")
    changed = _cfg(); changed["gif"] = {**changed["gif"], "low_ratio": 0.25, "salient_ratio": 0.75}
    for layout, neuron in ((phase_layout, "phase"), (mtn_layout, "mtn")):
        before = deployment_state_fingerprint(layout.post_finetuning_state_dir, neuron)["sha256"]
        _rematerialize_post_finetuning(layout, changed)
        assert deployment_state_fingerprint(layout.post_finetuning_state_dir, neuron)["sha256"] == before
        assert validate_conversion_metadata(changed, layout, neuron)["deployment_neuron"] == neuron


def test_phase_t_change_only_invalidates_phase_conversion_identity(tmp_path):
    phase_layout, _ = _prepare(tmp_path / "phase", neuron="phase")
    mtn_layout, _ = _prepare(tmp_path / "mtn", neuron="mtn")
    gif_layout, _ = _prepare(tmp_path / "gif", neuron="gif")
    changed = _cfg(); changed["phase"]["T"] = 8
    for layout, neuron, should_fail in ((phase_layout, "phase", True), (mtn_layout, "mtn", False), (gif_layout, "gif", False)):
        before = deployment_state_fingerprint(layout.post_finetuning_state_dir, neuron)["sha256"]
        _rematerialize_post_finetuning(layout, changed)
        after = deployment_state_fingerprint(layout.post_finetuning_state_dir, neuron)["sha256"]
        assert (after != before) is should_fail
        if should_fail:
            with pytest.raises(ValueError, match="deployment_state_fingerprint_sha256|deployment_parameters"):
                validate_conversion_metadata(changed, layout, neuron)
        else:
            assert validate_conversion_metadata(changed, layout, neuron)["deployment_neuron"] == neuron


def test_conversion_root_is_informational_not_semantic_identity(tmp_path):
    layout, path = _prepare(tmp_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["calibration_root"] = "/historical/stage-b-variant"
    metadata["calibration_state_manifest_sha256"] = "0" * 64
    metadata["source_statistics_manifest_path"] = "/historical/statistics"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    assert validate_conversion_metadata(_cfg(), layout, "gif")["deployment_neuron"] == "gif"
