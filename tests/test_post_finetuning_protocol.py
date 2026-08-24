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
        "training": {"learning_rate": 1e-6},
        "rotation": {"enabled": mode != "vanilla"},
        "prefix": {"enabled": mode != "vanilla"},
        "ann_training": {"prefix_enabled": mode != "vanilla"},
        "post_finetuning": {"prefix_enabled": not aware},
        "replacement": {"common_clip_enabled": common_clip_enabled},
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
    expected = (
        "prefix_enabled_ture_common_clip_enabled_true"
        if enabled
        else "prefix_enabled_ture_common_clip_enabled_false"
    )
    assert layout.root.parent.name == expected


def test_common_clip_variants_share_prefix_and_calibration_but_not_run_root():
    enabled = ArtifactLayout(_cfg("phase_aware", common_clip_enabled=True))
    disabled = ArtifactLayout(_cfg("phase_aware", common_clip_enabled=False))
    assert enabled.ann_training_prefix_dir == disabled.ann_training_prefix_dir
    assert enabled.ann_training_calibration_dir == disabled.ann_training_calibration_dir
    assert enabled.root != disabled.root


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
