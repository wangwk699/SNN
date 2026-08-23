from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.materialize_configs import materialize_configs
from snn2.config import validate_config
from snn2.temporal_ops import (
    COMMON_CLIP_TEMPORAL_POLICY,
    GIF_HIGH_QMAX,
    GIF_LOCAL_STEPS,
    GIF_STEP_QMAX,
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
            "common_clip_temporal_policy": COMMON_CLIP_TEMPORAL_POLICY,
        }
        assert cfg["gif"]["high_qmax"] == GIF_HIGH_QMAX
        assert cfg["gif"]["temporal_steps"] == GIF_LOCAL_STEPS
        assert cfg["gif"]["per_step_qmax"] == GIF_STEP_QMAX


def test_qwen17_quick_tldr_evaluation_remains_128_samples(generated_configs):
    for path in generated_configs:
        if "qwen3_1_7b_tldr" in path.name:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            assert cfg["evaluation"]["tldr_test_samples"] == 128


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("deployment", "prefix_temporal_policy", "full_prefix_each_timestep"),
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
