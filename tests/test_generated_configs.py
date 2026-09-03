from __future__ import annotations

from pathlib import Path

import pytest
import yaml
import copy
import json

from scripts.materialize_configs import materialize_configs
from snn2.config import validate_config
from snn2.evaluation import final_ann_replacement_mode
from snn2.temporal_ops import (
    GIF_HIGH_QMAX,
    GIF_LOCAL_STEPS,
    GIF_STEP_QMAX,
    EMBEDDING_TEMPORAL_POLICY,
    SOFTMAX_PREFIX_NEURON_POLICY,
    SOFTMAX_SITE5_GIF_POLICY,
    FINAL_NORM_NEURON_POLICY,
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


def test_all_twelve_generated_configs_carry_temporal_v5_and_ordinary_qmax30(generated_configs):
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
            "softmax_site5_gif_policy": SOFTMAX_SITE5_GIF_POLICY,
            "final_norm_neuron_policy": FINAL_NORM_NEURON_POLICY,
            "phase_tau_calibration": PHASE_TAU_CALIBRATION,
            "phase_tau_ema_factor": PHASE_TAU_EMA_FACTOR,
            "phase_tau_accumulator_dtype": PHASE_TAU_ACCUMULATOR_DTYPE,
        }
        assert cfg["phase"]["surrogate_slope"] == 1.0
        assert isinstance(cfg["replacement"]["common_clip_enabled"], bool)
        assert cfg["conversion"]["use_post_finetuning_artifacts"] is True
        if cfg["experiment"]["ann_mode"] in {"vanilla", "unaware"}:
            assert cfg["replacement"]["common_clip_enabled"] is False
        if cfg["experiment"]["ann_mode"] in {"phase_aware", "gif_aware"}:
            assert cfg["post_finetuning"] == {
                "rediscover_prefix": True,
                "recalibrate_sites": True,
                "prefix_enabled": True,
                "post_finetuning_recalibration": True,
            }
        assert cfg["gif"]["high_qmax"] == GIF_HIGH_QMAX
        assert cfg["gif"]["temporal_steps"] == GIF_LOCAL_STEPS
        assert cfg["gif"]["per_step_qmax"] == GIF_STEP_QMAX

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


@pytest.mark.parametrize("group_size", [-1, 1, 32])
def test_calibration_group_size_accepts_minus_one_or_positive(generated_configs, group_size):
    cfg = yaml.safe_load(generated_configs[0].read_text(encoding="utf-8"))
    cfg["calibration"]["group_size"] = group_size
    validate_config(cfg)


@pytest.mark.parametrize("group_size", [0, -2, 1.5, True])
def test_calibration_group_size_rejects_other_values(generated_configs, group_size):
    cfg = yaml.safe_load(generated_configs[0].read_text(encoding="utf-8"))
    cfg["calibration"]["group_size"] = group_size
    with pytest.raises(ValueError, match="group_size"):
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


def test_qwen3_8b_memory_optimized_training_configuration(generated_configs):
    qwen3_8b = [
        yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in generated_configs
        if path.stem.startswith("exp1_qwen3_8b_tldr__")
    ]
    assert len(qwen3_8b) == 4
    for cfg in qwen3_8b:
        training = cfg["training"]
        assert training["gradient_checkpointing"] is False
        assert training["deepspeed_config"] == "configs/deepspeed_zero3_cpu_offload.json"
        assert training["per_device_train_batch_size"] == 1
        assert training["gradient_accumulation_steps"] == 16
        assert cfg["data"]["max_seq_length"] == 2048
        assert training["bf16"] is True
        assert training["fp16"] is False


def test_deepspeed_zero3_cpu_offload_is_optimizer_only():
    config = json.loads(
        (ROOT / "configs" / "deepspeed_zero3_cpu_offload.json").read_text(
            encoding="utf-8"
        )
    )
    optimizer = config["optimizer"]
    assert optimizer["type"] == "AdamW"
    assert optimizer["params"] == {
        "lr": "auto",
        "betas": "auto",
        "eps": "auto",
        "weight_decay": "auto",
    }
    assert "torch_adam" not in optimizer["params"]
    assert "zero_force_ds_cpu_optimizer" not in config

    zero = config["zero_optimization"]
    assert zero["stage"] == 3
    assert zero["offload_optimizer"] == {"device": "cpu", "pin_memory": True}
    assert "offload_param" not in zero
    assert zero["overlap_comm"] is True
    assert zero["contiguous_gradients"] is True
    assert zero["stage3_gather_16bit_weights_on_model_save"] is True
    assert config["bf16"]["enabled"] is True
    assert config["fp16"]["enabled"] is False


def test_non_qwen3_8b_training_memory_settings_remain_unchanged(generated_configs):
    for path in generated_configs:
        if path.stem.startswith("exp1_qwen3_8b_tldr__"):
            continue
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert cfg["training"]["gradient_checkpointing"] is False
        assert cfg["training"]["deepspeed_config"] == "configs/deepspeed_zero3.json"

@pytest.mark.parametrize("mode", ["unaware", "phase_aware", "gif_aware"])
def test_non_vanilla_accepts_pre_finetuning_conversion_bundle(generated_configs, mode):
    cfg = yaml.safe_load(next(path for path in generated_configs if path.stem.endswith(f"__{mode}")).read_text(encoding="utf-8"))
    cfg["conversion"]["use_post_finetuning_artifacts"] = False
    validate_config(cfg)


def test_vanilla_rejects_pre_finetuning_conversion_bundle(generated_configs):
    cfg = yaml.safe_load(next(path for path in generated_configs if path.stem.endswith("__vanilla")).read_text(encoding="utf-8"))
    cfg["conversion"]["use_post_finetuning_artifacts"] = False
    with pytest.raises(ValueError, match="vanilla requires conversion.use_post_finetuning_artifacts=true"):
        validate_config(cfg)
