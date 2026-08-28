from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

import snn2.phase_conversion_regression as regression
from snn2.artifacts import sha256_file
from snn2.controller import SiteController
from snn2.evaluation import build_evaluation_controller
from snn2.neurons import PhaseSurrogate
from snn2.phase_statistics import PHASE_TAU_CHANNEL_POLICY, PHASE_TAU_REDUCTION_POLICY
from snn2.phase_conversion_regression import (
    PhaseConversionRegressionRecorder,
    summarize_first_divergence,
    tensor_metrics,
    validate_phase_conversion_artifacts,
)
from snn2.temporal_ops import (
    PHASE_TAU_CALIBRATION,
    PHASE_TAU_EMA_FACTOR,
    SITE_STATE_FORMAT_VERSION,
    TEMPORAL_IMPLEMENTATION_VERSION,
    from_temporal,
)


def _phase_state():
    return {
        "state_kind": "phase",
        "format_version": SITE_STATE_FORMAT_VERSION,
        "temporal_implementation_version": TEMPORAL_IMPLEMENTATION_VERSION,
        "T": 4,
        "base": 2.0,
        "parameter_layout": "last_dim_grouped",
        "configured_group_size": -1,
        "group_size": 9,
        "num_heads": None,
        "channels_per_head": 9,
        "groups_per_head": 1,
        "tau": torch.tensor([2.0]),
        "v0": torch.tensor([0.0625]),
        "tau_calibration": PHASE_TAU_CALIBRATION,
        "tau_ema_factor": PHASE_TAU_EMA_FACTOR,
        "tau_accumulator_dtype": "float32",
        "tau_channel_policy": PHASE_TAU_CHANNEL_POLICY,
        "tau_reduction_policy": PHASE_TAU_REDUCTION_POLICY,
    }


@pytest.mark.parametrize("decomposition", ["first", "uniform", "random"])
def test_phase_static_matches_temporal_sum_for_all_decompositions(decomposition):
    torch.manual_seed(101)
    module = PhaseSurrogate(_phase_state()).eval()
    x = torch.randn(2, 3, 9)
    if decomposition == "first":
        incoming = torch.cat((x.unsqueeze(0), torch.zeros(3, *x.shape)), dim=0)
    elif decomposition == "uniform":
        incoming = x.unsqueeze(0).expand(4, *x.shape) / 4
    else:
        prefix = torch.randn(3, *x.shape)
        incoming = torch.cat((prefix, (x - prefix.sum(0)).unsqueeze(0)), dim=0)
    static = module(x)
    temporal = module.temporal(incoming).sum(0)
    assert torch.equal(static, temporal)
    metric = tensor_metrics(static, temporal, name=decomposition)
    assert metric["relative_l2_error"] <= 1e-7
    assert metric["max_abs_error"] <= 1e-7


def test_recorder_temporal_reduction_uses_sum_dim_zero():
    temporal = torch.arange(4 * 2 * 3, dtype=torch.bfloat16).reshape(4, 2, 3)
    recorder = PhaseConversionRegressionRecorder("S", temporal_steps=4)
    recorder.record("node", from_temporal(temporal), temporal=True)
    torch.testing.assert_close(recorder.tensors["node"], temporal.float().sum(0))


def test_recorder_reports_first_divergence_and_site_amplification():
    reference = PhaseConversionRegressionRecorder("P")
    test = PhaseConversionRegressionRecorder("S")
    for name, ref, actual in (
        ("layer_000/site_01/pre", torch.ones(4), torch.ones(4) + 1e-4),
        ("layer_000/site_01/post", torch.ones(4), torch.ones(4) + 0.1),
    ):
        reference.record(name, ref)
        test.record(name, actual)
    rows = reference.compare(test)
    summary = summarize_first_divergence(rows)
    assert summary["first_relative_l2_gt_1e-2"]["name"].endswith("/post")
    assert rows[1]["local_error_amplification"] > 100


def test_final_norm_bypass_only_changes_deploy_phase_regression_path():
    value = torch.randn(8, 3)
    controller = SiteController(mode="identity")
    controller.mode = "deploy_phase"
    controller.temporal_steps = 4
    controller.regression_bypass_final_norm_phase = True
    assert controller.apply_final_norm_phase(value) is value

    identity = SiteController(mode="identity")
    identity.regression_bypass_final_norm_phase = True
    assert identity.apply_final_norm_phase(value) is value


@pytest.mark.parametrize(
    "field,value",
    [
        ("ann_mode", "gif_aware"),
        ("common_clip_enabled", True),
        ("prefix_enabled", False),
    ],
)
def test_regression_rejects_invalid_prerequisites(tmp_path, field, value):
    cfg = {
        "experiment": {"ann_mode": "phase_aware"},
        "replacement": {"common_clip_enabled": False},
        "ann_training": {"prefix_enabled": True},
    }
    if field == "ann_mode":
        cfg["experiment"][field] = value
    elif field == "common_clip_enabled":
        cfg["replacement"][field] = value
    else:
        cfg["ann_training"][field] = value
    with pytest.raises(ValueError, match="prerequisites"):
        validate_phase_conversion_artifacts(cfg, SimpleNamespace())


def test_regression_rejects_training_provenance_mismatch(tmp_path, monkeypatch):
    ann_checkpoint = tmp_path / "ann" / "final"
    prefix_dir = tmp_path / "prefix"
    site_dir = tmp_path / "sites"
    conversion_dir = tmp_path / "conversion"
    for directory in (ann_checkpoint, prefix_dir, site_dir, conversion_dir):
        directory.mkdir(parents=True)
    (prefix_dir / "prefix_state.json").write_text(
        json.dumps({"prefix_token_ids": [7]}), encoding="utf-8"
    )
    (prefix_dir / "prefixed_key_values.pt").write_bytes(b"kv")
    (site_dir / "calibration_state_manifest.json").write_text("{}", encoding="utf-8")
    phase_dir = site_dir / "layer_000" / "site_01_test"
    phase_dir.mkdir(parents=True)
    (phase_dir / "phase_state.pt").write_bytes(b"phase")
    (conversion_dir / "conversion_metadata.json").write_text("{}", encoding="utf-8")
    (ann_checkpoint / "config.json").write_text(
        json.dumps(
            {
                "snn2_ann_mode": "phase_aware",
                "snn2_ann_common_clip_enabled": False,
            }
        ),
        encoding="utf-8",
    )
    training_result = {
        "ann_training_common_clip_enabled": False,
        "ann_training_common_clip_applied": False,
        "ann_training_common_clip_state_required": True,
        "ann_training_prefix_state_sha256": "wrong",
        "ann_training_prefix_kv_sha256": sha256_file(prefix_dir / "prefixed_key_values.pt"),
        "ann_training_state_dependency_kinds": ["phase"],
        "ann_training_state_fingerprint_sha256": "placeholder",
        "final_model_checkpoint": str(ann_checkpoint.resolve()),
    }
    (tmp_path / "ann" / "training_result.json").write_text(
        json.dumps(training_result), encoding="utf-8"
    )
    layout = SimpleNamespace(
        ann_checkpoint_dir=ann_checkpoint,
        ann_training_prefix_dir=prefix_dir,
        ann_training_site_dir=site_dir,
        ann_dir=tmp_path / "ann",
        snn_conversion_dir=lambda _neuron: conversion_dir,
    )
    cfg = {
        "experiment": {"ann_mode": "phase_aware"},
        "replacement": {"common_clip_enabled": False},
        "ann_training": {"prefix_enabled": True},
    }
    monkeypatch.setattr(regression, "validate_conversion_metadata", lambda *_args: {})
    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_phase_conversion_artifacts(cfg, layout)


def test_official_phase_ann_controller_matches_graph_p(monkeypatch):
    monkeypatch.setattr("snn2.evaluation.validate_site_state_bundle", lambda *_a, **_k: {"manifest": {"calibration_group_size": -1, "calibration_grouping_policy": "per_head_within_head_groups_v1"}})
    layout = SimpleNamespace(ann_training_site_dir="training", conversion_site_dir="conversion")
    cfg = {
        "experiment": {"ann_mode": "phase_aware"},
        "phase": {"surrogate_slope": 2.0},
        "replacement": {"common_clip_enabled": False},
        "calibration": {"group_size": -1, "num_samples": 128},
    }
    controller, steps = build_evaluation_controller(cfg, layout, neuron="ann")
    assert controller.mode == "phase"
    assert controller.phase_surrogate_slope == 2.0
    assert str(controller.site_root) == "training"
    assert controller.common_clip_enabled is False
    assert steps == 1
