import json

import pytest
import torch

from snn2.calibration import materialize_calibration_states
from snn2.conversion import validate_calibration
from snn2.phase_statistics import (
    PHASE_TAU_ACCUMULATOR_DTYPE, PHASE_TAU_CALIBRATION,
    PHASE_TAU_CHANNEL_POLICY, PHASE_TAU_EMA_FACTOR, PHASE_TAU_REDUCTION_POLICY,
)
from snn2.sites import SITE_IDS, SITE_NAMES
from snn2.temporal_ops import STATISTICS_FORMAT_VERSION
from tests.calibration_fixtures import write_stage_a_statistics


def _statistics(site_index=None):
    if site_index in {2, 3, 4, 6}:
        shape, layout, heads, width, channels = (1, 4), "attention_head", 1, 4, 4
    elif site_index == 5:
        shape, layout, heads, width, channels = (1,), "attention_softmax", 1, None, 1
    else:
        shape, layout, heads, width, channels = (4,), "last_dim", None, None, 4
    return {
        "format_version": STATISTICS_FORMAT_VERSION, "site_index": 1 if site_index is None else site_index,
        "layout_kind": layout, "num_heads": heads, "channels_per_head": width, "channels": channels,
        "value_min": torch.full(shape, -1.0), "value_max": torch.full(shape, 1.0),
        "saliency_row_count": torch.ones((0,) if site_index == 5 else shape, dtype=torch.long),
        "saliency_sum": torch.zeros((0,) if site_index == 5 else shape, dtype=torch.float64),
        "phase_ema_abs_max": torch.ones(shape), "phase_ema_updates": torch.ones(shape, dtype=torch.long),
        "phase_tau_calibration": PHASE_TAU_CALIBRATION, "phase_tau_ema_factor": PHASE_TAU_EMA_FACTOR,
        "phase_tau_accumulator_dtype": PHASE_TAU_ACCUMULATOR_DTYPE,
        "phase_tau_channel_policy": PHASE_TAU_CHANNEL_POLICY,
        "phase_tau_reduction_policy": PHASE_TAU_REDUCTION_POLICY,
    }


def _cfg():
    return {"calibration": {"group_size": -1, "num_samples": 128, "seed": 42, "expected_sites_per_layer": 10},
            "phase": {"T": 4, "base": 2.0},
            "gif": {"base_bits": 4, "add_bits": 1, "low_ratio": 0.5, "salient_ratio": 0.5},
            "mtn": {"T": 4, "K": 6, "threshold_factor": 0.75}}


def _materialize(tmp_path, *, layers=1, include_clip=False):
    cfg = _cfg(); stats, states = tmp_path / "statistics", tmp_path / "states"
    write_stage_a_statistics(stats, cfg, _statistics, layers=layers)
    return cfg, stats, states, materialize_calibration_states(
        stats, states, cfg, {"purpose": "post_finetuning_conversion_calibration", "source_model_stage": None,
        "source_ann_mode": None, "source_ann_checkpoint": None, "source_ann_config_sha256": None,
        "prefix_enabled": False, "prefix_state_sha256": None, "prefix_kv_sha256": None,
        "rotation_enabled": False, "rotation_state_sha256": None, "calibration_data_manifest_sha256": None,
        "calibration_grouping_policy": "per_head_within_head_groups_v1"}, include_clip=include_clip,
        expected_num_hidden_layers=layers)


def test_stage_a_and_b_are_physically_separate(tmp_path):
    _, stats, states, manifest = _materialize(tmp_path, include_clip=True)
    first_stats = stats / "layer_000" / f"site_01_{SITE_NAMES[1]}"
    first_state = states / "layer_000" / f"site_01_{SITE_NAMES[1]}"
    assert (first_stats / "statistics.pt").exists()
    assert (first_stats / "statistics_summary.json").exists()
    assert not (first_stats / "phase_state.pt").exists()
    assert (first_state / "phase_state.pt").exists()
    assert (first_state / "gif_state.pt").exists()
    assert (first_state / "mtn_state.pt").exists()
    assert (first_state / "clip_state.pt").exists()
    assert not (first_state / "statistics.pt").exists()
    assert manifest["source_statistics_manifest_sha256"]


def test_post_finetuning_bundle_is_clip_free(tmp_path):
    _, _, states, _ = _materialize(tmp_path, include_clip=False)
    assert not list(states.glob("layer_*/site_*/clip_state.pt"))
    assert validate_calibration(states, clip_policy="forbid_all")["sites"] == len(SITE_IDS)


def test_site5_never_gets_clip(tmp_path):
    _, _, states, _ = _materialize(tmp_path, include_clip=True)
    assert not (states / "layer_000" / f"site_05_{SITE_NAMES[5]}" / "clip_state.pt").exists()


def test_stage_b_rejects_statistics_hash_tampering(tmp_path):
    cfg = _cfg(); stats, states = tmp_path / "statistics", tmp_path / "states"
    write_stage_a_statistics(stats, cfg, _statistics)
    (stats / "layer_000" / f"site_01_{SITE_NAMES[1]}" / "statistics.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        materialize_calibration_states(stats, states, cfg, {"purpose": "post_finetuning_conversion_calibration", "source_model_stage": None, "source_ann_mode": None, "source_ann_checkpoint": None, "source_ann_config_sha256": None, "prefix_enabled": False, "prefix_state_sha256": None, "prefix_kv_sha256": None, "rotation_enabled": False, "rotation_state_sha256": None, "calibration_data_manifest_sha256": None, "calibration_grouping_policy": "per_head_within_head_groups_v1"}, include_clip=False, expected_num_hidden_layers=1)


def test_state_manifest_rejects_legacy_schema(tmp_path):
    _, _, states, _ = _materialize(tmp_path)
    manifest_path = states / "calibration_state_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="legacy calibration manifest"):
        validate_calibration(states, clip_policy="forbid_all")
