from __future__ import annotations

from pathlib import Path

import pytest
import yaml
import copy

from scripts.materialize_configs import materialize_configs
from snn2.config import validate_config
from snn2.evaluation import final_ann_replacement_mode
from snn2.temporal_ops import (
    GIF_HIGH_QMAX,
    GIF_LOCAL_STEPS,
    GIF_STEP_QMAX,
    EMBEDDING_TEMPORAL_POLICY,
    SOFTMAX_PREFIX_NEURON_POLICY,
    PHASE_FINAL_NORM_POLICY,
    PHASE_TAU_CALIBRATION,
    PHASE_TAU_EMA_FACTOR,
    PHASE_TAU_ACCUMULATOR_DTYPE,
    PREFIX_TEMPORAL_POLICY,
    TEMPORAL_IMPLEMENTATION,
    TEMPORAL_LAYOUT,
    TEMPORAL_LINEAR_BIAS_POLICY,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def generated_configs(tmp_path):
    return materialize_configs(
        ROOT / "configs" / "experiment_matrix.yaml",
        tmp_path / "generated",
    )


def test_all_twelve_generated_configs_carry_temporal_v2_and_qmax30(generated_configs):
    paths = generated_configs
    assert len(paths) == 12
    for path in paths:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        validate_config(cfg)
        assert cfg["deployment"] == {
            "temporal_implementation": TEMPORAL_IMPLEMENTATION,
            "temporal_layout": TEMPORAL_LAYOUT,
            "linear_bias_policy": TEMPORAL_LINEAR_BIAS_POLICY,
            "prefix_temporal_policy": PREFIX_TEMPORAL_POLICY,
            "embedding_temporal_policy": EMBEDDING_TEMPORAL_POLICY,
            "softmax_prefix_neuron_policy": SOFTMAX_PREFIX_NEURON_POLICY,
            "phase_final_norm_policy": PHASE_FINAL_NORM_POLICY,
            "phase_tau_calibration": PHASE_TAU_CALIBRATION,
            "phase_tau_ema_factor": PHASE_TAU_EMA_FACTOR,
            "phase_tau_accumulator_dtype": PHASE_TAU_ACCUMULATOR_DTYPE,
        }
        assert cfg["phase"]["surrogate_slope"] == 1.0
        expected_clip = cfg["experiment"]["ann_mode"] in {"phase_aware", "gif_aware"}
        assert cfg["replacement"]["common_clip_enabled"] is expected_clip
        assert cfg["gif"]["high_qmax"] == GIF_HIGH_QMAX
        assert cfg["gif"]["temporal_steps"] == GIF_LOCAL_STEPS
        assert cfg["gif"]["per_step_qmax"] == GIF_STEP_QMAX


def test_qwen17_quick_tldr_evaluation_remains_128_samples(generated_configs):
    for path in generated_configs:
        if "qwen3_1_7b_tldr" in path.name:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            assert cfg["evaluation"]["tldr_test_samples"] == 128


@pytest.mark.parametrize("mode", ["phase_aware", "gif_aware"])
@pytest.mark.parametrize("enabled", [True, False])
def test_aware_common_clip_boolean_variants_are_valid(generated_configs, mode, enabled):
    path = next(
        path for path in generated_configs if path.stem.endswith(f"__{mode}")
    )
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["replacement"]["common_clip_enabled"] = enabled
    validate_config(cfg)


def test_common_clip_rejects_non_boolean(generated_configs):
    cfg = yaml.safe_load(generated_configs[0].read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg["replacement"]["common_clip_enabled"] = "false"
    with pytest.raises(ValueError, match="must be true or false"):
        validate_config(cfg)


@pytest.mark.parametrize("slope", [0.5, 1.0, 2.0, 4.0])
def test_surrogate_slope_accepts_positive_finite_values(generated_configs, slope):
    cfg = yaml.safe_load(generated_configs[0].read_text(encoding="utf-8"))
    cfg["phase"]["surrogate_slope"] = slope
    validate_config(cfg)


@pytest.mark.parametrize(
    "slope", [0.0, -1.0, float("inf"), float("nan"), "invalid"]
)
def test_surrogate_slope_rejects_non_positive_or_non_finite_values(
    generated_configs, slope
):
    cfg = yaml.safe_load(generated_configs[0].read_text(encoding="utf-8"))
    cfg["phase"]["surrogate_slope"] = slope
    with pytest.raises(ValueError, match="positive finite number"):
        validate_config(cfg)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("deployment", "prefix_temporal_policy", "full_prefix_each_timestep"),
        ("deployment", "common_clip_temporal_policy", "cumulative_then_difference"),
        ("gif", "high_qmax", 31),
        ("gif", "per_step_qmax", 16),
    ],
)
def test_config_rejects_legacy_or_unsupported_policy(
    generated_configs, section, key, value
):
    cfg = yaml.safe_load(generated_configs[0].read_text(encoding="utf-8"))
    cfg[section][key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_generated_configs_define_final_ann_forward_semantics(generated_configs):
    expected = {
        "vanilla": "identity",
        "unaware": "identity",
        "phase_aware": "phase",
        "gif_aware": "gif",
    }
    for path in generated_configs:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert final_ann_replacement_mode(cfg) == expected[cfg["experiment"]["ann_mode"]]
