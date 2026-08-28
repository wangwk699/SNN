import json
from pathlib import Path

import pytest

from snn2.artifacts import ArtifactLayout
from snn2.config import (
    conversion_calibration_stage,
    conversion_prefix_enabled,
    conversion_reuses_ann_training_artifacts,
    final_evaluation_prefix_artifact_stage,
    requires_ann_training_calibration,
    requires_post_finetuning_artifacts,
    requires_pre_finetuning_prefix,
)
from snn2.conversion import validate_conversion_prefix


def _cfg(mode, root="artifacts", common_clip_enabled=True):
    aware = mode in {"phase_aware", "gif_aware"}
    return {
        "experiment": {
            "id": "e", "task": "t", "model_name": "m", "seed": 42,
            "output_root": str(root), "ann_mode": mode,
        },
        "training": {"learning_rate": 1e-6, "warmup_ratio": 0.03},
        "rotation": {"enabled": mode != "vanilla"},
        "prefix": {"enabled": mode != "vanilla"},
        "phase": {"T": 4, "surrogate_slope": 1.0},
        "mtn": {"T": 4, "K": 6},
        "gif": {"low_ratio": 0.9, "salient_ratio": 0.1},
        "ann_training": {"prefix_enabled": mode != "vanilla"},
        "post_finetuning": {"prefix_enabled": not aware},
        "replacement": {"common_clip_enabled": common_clip_enabled},
        "calibration": {"group_size": -1, "num_samples": 128, "seed": 42},
    }


@pytest.mark.parametrize(
    ("mode", "pre", "ann_cal", "post", "reused", "stage"),
    [
        ("vanilla", False, False, True, False, "post_finetuning"),
        ("unaware", True, False, True, False, "post_finetuning"),
        ("phase_aware", True, True, False, True, "pre_finetuning"),
        ("gif_aware", True, True, False, True, "pre_finetuning"),
    ],
)
def test_mode_aware_protocol_table(mode, pre, ann_cal, post, reused, stage):
    cfg = _cfg(mode)
    assert requires_pre_finetuning_prefix(cfg) is pre
    assert requires_ann_training_calibration(cfg) is ann_cal
    assert requires_post_finetuning_artifacts(cfg) is post
    assert conversion_reuses_ann_training_artifacts(cfg) is reused
    assert conversion_calibration_stage(cfg) == (
        "ann_training" if reused else "post_finetuning"
    )
    assert final_evaluation_prefix_artifact_stage(cfg) == stage
    assert conversion_prefix_enabled(cfg) is True


def test_mode_aware_conversion_roots():
    aware = ArtifactLayout(_cfg("phase_aware"))
    assert aware.conversion_prefix_dir == aware.ann_training_prefix_dir
    assert aware.conversion_site_dir == aware.ann_training_site_dir
    unaware = ArtifactLayout(_cfg("unaware"))
    assert unaware.conversion_prefix_dir == unaware.post_finetuning_prefix_dir
    assert unaware.conversion_site_dir == unaware.post_finetuning_site_dir


@pytest.mark.parametrize("mode", ["phase_aware", "gif_aware"])
@pytest.mark.parametrize("enabled", [True, False])
def test_aware_run_root_records_common_clip_variant(mode, enabled):
    layout = ArtifactLayout(_cfg(mode, common_clip_enabled=enabled))
    expected = f"prefix_enabled_ture_common_clip_enabled_{str(enabled).lower()}"
    assert layout.root.parent.parent.name == expected
    expected_params = "phase_T_4_mtn_T_4_low_ratio_0.9_salient_ratio_0.1"
    if mode == "phase_aware": expected_params += "_surrogate_slope_1.0"
    expected_params += "_warmup_ratio_0.03"
    assert layout.root.parent.name == expected_params
    assert layout.root.parent.parent.parent.name == "lr1e-06_calibration_group_size_-1_num_samples_128"


def test_phase_aware_run_root_records_slope_and_warmup_ratio():
    first_cfg = _cfg("phase_aware")
    second_cfg = _cfg("phase_aware")
    second_cfg["phase"]["surrogate_slope"] = 0.5
    second_cfg["training"]["warmup_ratio"] = 0.1
    first = ArtifactLayout(first_cfg)
    second = ArtifactLayout(second_cfg)

    assert first.root.parent.name == "phase_T_4_mtn_T_4_low_ratio_0.9_salient_ratio_0.1_surrogate_slope_1.0_warmup_ratio_0.03"
    assert second.root.parent.name == "phase_T_4_mtn_T_4_low_ratio_0.9_salient_ratio_0.1_surrogate_slope_0.5_warmup_ratio_0.1"
    assert first.root.parent.parent.parent.name == "lr1e-06_calibration_group_size_-1_num_samples_128"
    assert first.root != second.root
    assert first.ann_training_prefix_dir == second.ann_training_prefix_dir
    assert first.ann_training_calibration_dir == second.ann_training_calibration_dir


def test_aware_modes_and_surrogate_slopes_share_calibration():
    phase = ArtifactLayout(_cfg("phase_aware"))
    gif_cfg = _cfg("gif_aware")
    gif_cfg["phase"]["surrogate_slope"] = 2.0
    gif = ArtifactLayout(gif_cfg)
    assert phase.ann_training_calibration_dir == gif.ann_training_calibration_dir


def test_common_clip_variants_share_prefix_and_calibration_but_not_run_root():
    enabled = ArtifactLayout(_cfg("phase_aware", common_clip_enabled=True))
    disabled = ArtifactLayout(_cfg("phase_aware", common_clip_enabled=False))
    assert enabled.ann_training_prefix_dir == disabled.ann_training_prefix_dir
    assert enabled.ann_training_calibration_dir == disabled.ann_training_calibration_dir
    assert enabled.root != disabled.root


def test_group_size_isolates_calibration_aware_runs_and_snn_but_not_identity_ann():
    aware_a = _cfg("phase_aware")
    aware_b = _cfg("phase_aware")
    aware_b["calibration"]["group_size"] = 2
    assert ArtifactLayout(aware_a).ann_training_calibration_dir != ArtifactLayout(aware_b).ann_training_calibration_dir
    assert ArtifactLayout(aware_a).root != ArtifactLayout(aware_b).root
    vanilla_a = _cfg("vanilla")
    vanilla_b = _cfg("vanilla")
    vanilla_b["calibration"]["group_size"] = 2
    layout_a, layout_b = ArtifactLayout(vanilla_a), ArtifactLayout(vanilla_b)
    assert layout_a.ann_checkpoint_dir == layout_b.ann_checkpoint_dir
    assert layout_a.post_finetuning_site_dir != layout_b.post_finetuning_site_dir
    assert layout_a.snn_dir("phase") != layout_b.snn_dir("phase")


@pytest.mark.parametrize("mode", ["phase_aware", "gif_aware"])
def test_aware_snn_path_contains_group_size_exactly_once(mode):
    layout = ArtifactLayout(_cfg(mode))
    path = layout.snn_dir("phase")
    assert sum(
        part.count("calibration_group_size_-1") for part in path.parts
    ) == 1
    assert path.parts[-4:] == ("seed42", "snn", "phase", "T_4")
    assert "lr1e-06_calibration_group_size_-1_num_samples_128" in path.parts


@pytest.mark.parametrize("mode", ["vanilla", "unaware"])
def test_identity_ann_snn_path_groups_below_snn(mode):
    layout = ArtifactLayout(_cfg(mode))
    path = layout.snn_dir("phase")
    assert path.parts.count("calibration_group_size_-1_num_samples_128") == 1
    assert path.parts[-4:] == ("snn", "calibration_group_size_-1_num_samples_128", "phase", "T_4")


def test_calibration_config_logs_and_sites_are_group_isolated_but_shared_inputs_are_not():
    first_cfg = _cfg("phase_aware")
    second_cfg = _cfg("phase_aware")
    second_cfg["calibration"]["group_size"] = 32
    first = ArtifactLayout(first_cfg)
    second = ArtifactLayout(second_cfg)

    assert first.ann_training_calibration_config_dir != second.ann_training_calibration_config_dir
    assert first.ann_training_calibration_logs_dir != second.ann_training_calibration_logs_dir
    assert first.ann_training_site_dir != second.ann_training_site_dir
    assert first.ann_training_prefix_dir == second.ann_training_prefix_dir
    assert first.rotation_dir == second.rotation_dir
    assert first.data_dir == second.data_dir


@pytest.mark.parametrize("mode", ["vanilla", "unaware"])
def test_post_finetuning_calibration_config_logs_and_sites_are_group_isolated(mode):
    first_cfg = _cfg(mode)
    second_cfg = _cfg(mode)
    second_cfg["calibration"]["group_size"] = 32
    first = ArtifactLayout(first_cfg)
    second = ArtifactLayout(second_cfg)

    assert first.post_finetuning_conversion_calibration_config_dir != second.post_finetuning_conversion_calibration_config_dir
    assert first.post_finetuning_conversion_calibration_logs_dir != second.post_finetuning_conversion_calibration_logs_dir
    assert first.post_finetuning_site_dir != second.post_finetuning_site_dir
    assert first.ann_checkpoint_dir == second.ann_checkpoint_dir
    assert first.post_finetuning_prefix_dir == second.post_finetuning_prefix_dir


def test_vanilla_analysis_calibration_config_logs_and_sites_are_group_isolated():
    first_cfg = _cfg("vanilla")
    second_cfg = _cfg("vanilla")
    second_cfg["calibration"]["group_size"] = 32
    first = ArtifactLayout(first_cfg)
    second = ArtifactLayout(second_cfg)

    assert first.vanilla_analysis_calibration_config_dir != second.vanilla_analysis_calibration_config_dir
    assert first.vanilla_analysis_calibration_logs_dir != second.vanilla_analysis_calibration_logs_dir
    assert first.vanilla_analysis_site_dir != second.vanilla_analysis_site_dir
    assert first.data_dir == second.data_dir


@pytest.mark.parametrize(
    ("configured", "suffix"),
    [(None, "lr1e-06_train_samples_full/prefix_enabled_false/seed42"),
     (128, "lr1e-06_train_samples_128/prefix_enabled_false/seed42")],
)
def test_vanilla_tldr_path_records_no_pretraining_prefix(configured, suffix):
    cfg = _cfg("vanilla")
    cfg["experiment"]["task"] = "tldr"
    cfg["training"]["tldr_train_samples"] = configured
    assert ArtifactLayout(cfg).root.parts[-3:] == Path(suffix).parts[-3:]


def test_conversion_prefix_validator_uses_aware_pre_finetuning_root(tmp_path):
    cfg = _cfg("phase_aware", tmp_path)
    layout = ArtifactLayout(cfg)
    state = layout.ann_training_prefix_dir / "prefix_state.json"
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({"prefix_token_ids": []}), encoding="utf-8")
    metadata = validate_conversion_prefix(cfg, layout)
    assert metadata["prefix_source_stage"] == "pre_finetuning"
    assert metadata["prefix_root"] == str(layout.ann_training_prefix_dir.resolve())


def test_conversion_prefix_validator_uses_post_finetuning_root(tmp_path):
    cfg = _cfg("unaware", tmp_path)
    layout = ArtifactLayout(cfg)
    state = layout.post_finetuning_prefix_dir / "prefix_state.json"
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({"prefix_token_ids": [123]}), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="KV cache"):
        validate_conversion_prefix(cfg, layout)
