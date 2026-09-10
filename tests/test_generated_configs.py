from __future__ import annotations

from pathlib import Path

import pytest
import yaml
import copy
import json

from scripts.materialize_configs import materialize_configs
from snn2.config import validate_config
from snn2.evaluation import final_ann_replacement_mode, lm_eval_batch_size
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


def test_all_twelve_generated_configs_carry_temporal_v8_and_ordinary_qmax30(generated_configs):
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


def test_qwen3_1_7b_training_memory_settings_remain_unchanged(generated_configs):
    qwen3_1_7b = [
        yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in generated_configs
        if path.stem.startswith("exp1_qwen3_1_7b_tldr__")
    ]
    assert len(qwen3_1_7b) == 4
    for cfg in qwen3_1_7b:
        assert cfg["training"]["gradient_checkpointing"] is False
        assert cfg["training"]["deepspeed_config"] == "configs/deepspeed_zero3.json"


def test_tulu3_training_uses_cpu_optimizer_offload(generated_configs):
    tulu3 = [
        yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in generated_configs
        if path.stem.startswith("exp2_llama3_8b_tulu3__")
    ]
    assert len(tulu3) == 4
    for cfg in tulu3:
        assert cfg["training"]["gradient_checkpointing"] is False
        assert (
            cfg["training"]["deepspeed_config"]
            == "configs/deepspeed_zero3_cpu_offload.json"
        )

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


def test_generated_evaluation_configs_are_task_specific(generated_configs):
    expected_names = [
        "truthfulqa_mc1", "agieval", "arc_challenge", "piqa", "winogrande",
        "boolq", "mmlu_pro", "bbh", "gsm8k_cot", "minerva_math",
    ]
    expected_enabled = expected_names[:6]
    tldr_only = {"max_new_tokens", "tldr_input_length", "tldr_test_samples", "tldr_test_seed", "rouge_types"}
    lm_eval_only = {"lm_eval_revision", "apply_chat_template", "lm_eval_task_specs", "limit"}
    for path in generated_configs:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        evaluation = cfg["evaluation"]
        if cfg["experiment"]["task"] == "tldr":
            assert tldr_only <= set(evaluation)
            assert not (lm_eval_only & set(evaluation))
            assert evaluation["batch_size"] == 8
            assert "snn_batch_size" not in evaluation
        else:
            assert not (tldr_only & set(evaluation))
            assert {
                "prefix_enabled", "batch_size", "snn_batch_size",
                "lm_eval_revision", "apply_chat_template", "lm_eval_task_specs",
            } <= set(evaluation)
            assert evaluation["batch_size"] == 8
            assert evaluation["snn_batch_size"] == 1
            specs = evaluation["lm_eval_task_specs"]
            assert [spec["name"] for spec in specs] == expected_names
            assert [spec["name"] for spec in specs if spec["enabled"]] == expected_enabled
            assert all(spec["enabled"] is False for spec in specs[6:])


@pytest.mark.parametrize(
    ("neuron", "expected"),
    [("ann", 8), ("phase", 1), ("gif", 1), ("mtn", 1)],
)
def test_tulu3_lm_eval_batch_size_uses_snn_setting_for_snn_neurons(
    generated_configs, neuron, expected
):
    cfg = yaml.safe_load(next(
        path for path in generated_configs
        if path.stem.startswith("exp2_llama3_8b_tulu3__")
    ).read_text(encoding="utf-8"))
    assert lm_eval_batch_size(cfg, neuron=neuron) == expected


@pytest.mark.parametrize("neuron", ["ann", "phase", "gif", "mtn"])
def test_tldr_lm_eval_batch_size_keeps_existing_setting(generated_configs, neuron):
    for path in generated_configs:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        if cfg["experiment"]["task"] == "tldr":
            assert lm_eval_batch_size(cfg, neuron=neuron) == cfg["evaluation"]["batch_size"]


def test_tulu3_snn_batch_size_is_required(generated_configs):
    cfg = yaml.safe_load(next(
        path for path in generated_configs
        if path.stem.startswith("exp2_llama3_8b_tulu3__")
    ).read_text(encoding="utf-8"))
    cfg["evaluation"].pop("snn_batch_size")
    with pytest.raises(ValueError, match="snn_batch_size must be a positive integer"):
        validate_config(cfg)


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "1", None])
def test_tulu3_snn_batch_size_must_be_positive_integer(generated_configs, value):
    cfg = yaml.safe_load(next(
        path for path in generated_configs
        if path.stem.startswith("exp2_llama3_8b_tulu3__")
    ).read_text(encoding="utf-8"))
    cfg["evaluation"]["snn_batch_size"] = value
    with pytest.raises(ValueError, match="snn_batch_size must be a positive integer"):
        validate_config(cfg)


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "1", None])
def test_tulu3_batch_size_must_be_positive_integer(generated_configs, value):
    cfg = yaml.safe_load(next(
        path for path in generated_configs
        if path.stem.startswith("exp2_llama3_8b_tulu3__")
    ).read_text(encoding="utf-8"))
    cfg["evaluation"]["batch_size"] = value
    with pytest.raises(ValueError, match="evaluation.batch_size must be a positive integer"):
        validate_config(cfg)


def test_tldr_and_tulu_aware_run_paths_include_epochs_before_calibration_identity(generated_configs):
    from snn2.artifacts import ArtifactLayout
    for path in generated_configs:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        layout = ArtifactLayout(cfg)
        if cfg["experiment"]["task"] in {"tldr", "tulu3"}:
            prefix = f"epochs_{cfg['training']['num_train_epochs']}_"
            if cfg["experiment"]["ann_mode"] in {"phase_aware", "gif_aware"}:
                assert any(part.startswith(prefix) for part in layout.root.parts)
                assert any(part.startswith(prefix + "num_samples_") for part in layout.root.parts)
            else:
                assert not any(part.startswith("epochs_") for part in layout.root.parts)


def test_tulu_task_result_path_container_is_isolated_from_tldr(generated_configs):
    from snn2.artifacts import lm_eval_spec_dirname, safe_name
    for path in generated_configs:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        if cfg["experiment"]["task"] == "tulu3":
            spec = next(spec for spec in cfg["evaluation"]["lm_eval_task_specs"] if spec["enabled"])
            task_path = Path("evaluation") / "task_results" / safe_name(spec["name"]) / lm_eval_spec_dirname(spec)
            assert task_path.parts[1] == "task_results"
        else:
            assert "task_results" not in cfg["evaluation"]


@pytest.mark.parametrize("mode", ["vanilla", "unaware", "phase_aware", "gif_aware"])
def test_tldr_full_run_paths_record_scheduler_warmup_and_aware_accumulation(
    generated_configs, mode
):
    from snn2.artifacts import ArtifactLayout, safe_name

    matching = [
        path for path in generated_configs
        if "tldr" in path.stem and path.stem.endswith(f"__{mode}")
    ]
    assert len(matching) == 2
    for path in matching:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        training = cfg["training"]
        model = safe_name(cfg["experiment"]["model_name"])
        prefix = "prefix_enabled_false" if mode == "vanilla" else "prefix_enabled_ture"
        learning = (
            f"epochs_{training['num_train_epochs']}_"
            + (f"num_samples_{cfg['calibration']['num_samples']}_" if mode in {"phase_aware", "gif_aware"} else "")
            + f"lr{training['learning_rate']}_train_samples_{training['tldr_train_samples']}"
            + (f"_calibration_group_size_{cfg['calibration']['group_size']}" if mode in {"phase_aware", "gif_aware"} else "")
        )
        if mode in {"phase_aware", "gif_aware"}:
            prefix += f"_common_clip_enabled_{str(cfg['replacement']['common_clip_enabled']).lower()}"
            phase_part = f"phase_T_{cfg['phase']['T']}_mtn_T_{cfg['mtn']['T']}"
            if mode == "phase_aware":
                phase_part += f"_surrogate_slope_{float(cfg['phase']['surrogate_slope'])}"
            training_identity = (
                f"{phase_part}_lr_scheduler_type_{training['lr_scheduler_type']}_"
                f"warmup_ratio_{float(training['warmup_ratio'])}_"
                f"gradient_accumulation_steps_{training['gradient_accumulation_steps']}"
            )
        else:
            training_identity = (
                f"lr_scheduler_type_{training['lr_scheduler_type']}_"
                f"warmup_ratio_{float(training['warmup_ratio'])}"
            )
        expected = Path(
            f"artifacts/{cfg['experiment']['id']}/tldr/{model}/{mode}/{learning}/"
            f"{prefix}/{training_identity}/seed{cfg['experiment']['seed']}"
        )
        layout = ArtifactLayout(cfg)
        assert layout.root == expected
        if mode in {"vanilla", "unaware"}:
            assert "gradient_accumulation_steps_" not in str(layout.root)


def test_tulu_full_run_paths_include_aware_training_identity(generated_configs):
    from snn2.artifacts import ArtifactLayout

    expected = {
        "vanilla": "artifacts/snn2_main_v1/tulu3/meta-llama_Meta-Llama-3-8B/vanilla/lr1e-06_train_samples_10000/prefix_enabled_false/lr_scheduler_type_cosine_warmup_ratio_0.0/seed42",
        "unaware": "artifacts/snn2_main_v1/tulu3/meta-llama_Meta-Llama-3-8B/unaware/lr1e-06_train_samples_10000/prefix_enabled_ture/lr_scheduler_type_cosine_warmup_ratio_0.0/seed42",
        "phase_aware": "artifacts/snn2_main_v1/tulu3/meta-llama_Meta-Llama-3-8B/phase_aware/epochs_1_num_samples_128_lr1e-06_train_samples_10000_calibration_group_size_128/prefix_enabled_ture_common_clip_enabled_true/phase_T_4_mtn_T_4_surrogate_slope_1.0_lr_scheduler_type_cosine_warmup_ratio_0.0_gradient_accumulation_steps_16/seed42",
        "gif_aware": "artifacts/snn2_main_v1/tulu3/meta-llama_Meta-Llama-3-8B/gif_aware/epochs_1_num_samples_128_lr1e-06_train_samples_10000_calibration_group_size_128/prefix_enabled_ture_common_clip_enabled_true/phase_T_4_mtn_T_4_lr_scheduler_type_cosine_warmup_ratio_0.0_gradient_accumulation_steps_16/seed42",
    }
    for path in generated_configs:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        if cfg["experiment"]["task"] == "tulu3":
            assert str(ArtifactLayout(cfg).root) == expected[cfg["experiment"]["ann_mode"]]


@pytest.mark.parametrize("mode", ["vanilla", "unaware", "phase_aware", "gif_aware"])
def test_tldr_scheduler_and_warmup_are_run_identity(generated_configs, mode):
    import copy
    from snn2.artifacts import ArtifactLayout

    path = next(
        path for path in generated_configs
        if path.stem.startswith("exp1_qwen3_1_7b_tldr__")
        and path.stem.endswith(f"__{mode}")
    )
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    baseline = ArtifactLayout(cfg).root
    scheduler = copy.deepcopy(cfg)
    scheduler["training"]["lr_scheduler_type"] = "linear"
    warmup = copy.deepcopy(cfg)
    warmup["training"]["warmup_ratio"] = 0.25
    assert ArtifactLayout(scheduler).root != baseline
    assert ArtifactLayout(warmup).root != baseline


def test_only_tldr_and_tulu_aware_modes_use_gradient_accumulation_as_run_identity(generated_configs):
    import copy
    from snn2.artifacts import ArtifactLayout

    for path in generated_configs:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        if cfg["experiment"]["task"] not in {"tldr", "tulu3"}:
            continue
        changed = copy.deepcopy(cfg)
        changed["training"]["gradient_accumulation_steps"] += 1
        mode = cfg["experiment"]["ann_mode"]
        if mode in {"phase_aware", "gif_aware"}:
            assert ArtifactLayout(changed).root != ArtifactLayout(cfg).root
        else:
            assert ArtifactLayout(changed).root == ArtifactLayout(cfg).root


def test_tulu_gradient_accumulation_changes_only_aware_run_paths(generated_configs):
    import copy
    from snn2.artifacts import ArtifactLayout

    for path in generated_configs:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        if cfg["experiment"]["task"] != "tulu3":
            continue
        changed = copy.deepcopy(cfg)
        changed["training"]["gradient_accumulation_steps"] += 1
        mode = cfg["experiment"]["ann_mode"]
        if mode in {"phase_aware", "gif_aware"}:
            assert ArtifactLayout(changed).root != ArtifactLayout(cfg).root
        else:
            assert ArtifactLayout(changed).root == ArtifactLayout(cfg).root
