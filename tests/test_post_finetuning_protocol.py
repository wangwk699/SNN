import json

import pytest
from pathlib import Path

from snn2.artifacts import ArtifactLayout
from snn2.config import post_finetuning_prefix_enabled, training_prefix_enabled
from snn2.conversion import validate_post_finetuning_prefix


def _cfg(mode):
    return {"experiment": {"id": "e", "task": "t", "model_name": "m", "seed": 42, "output_root": "artifacts", "ann_mode": mode}, "training": {"learning_rate": 1e-6}, "rotation": {"enabled": mode != "vanilla"}, "prefix": {"enabled": True}, "post_finetuning": {"prefix_enabled": True}}


def test_stage_specific_artifact_paths():
    layout = ArtifactLayout(_cfg("phase_aware"))
    assert layout.ann_training_prefix_dir.parts[-1:] == ("pre_finetuning_prefix",)
    assert layout.ann_training_site_dir.parts[-3:] == (
        "ann_training_calibration", "prefix_enabled_ture", "sites"
    )
    assert layout.vanilla_analysis_site_dir.parts[-2:] == (
        "vanilla_analysis_calibration", "sites"
    )
    assert layout.post_finetuning_prefix_dir.parts[-2:] == ("post_finetuning", "prefix")
    assert layout.post_finetuning_site_dir.parts[-4:] == (
        "post_finetuning", "conversion_calibration", "prefix_enabled_ture", "sites"
    )


@pytest.mark.parametrize(
    ("configured", "suffix"),
    [
        (None, "lr1e-06_train_samples_full/prefix_enabled_ture/seed42"),
        (128, "lr1e-06_train_samples_128/prefix_enabled_ture/seed42"),
    ],
)
def test_tldr_training_sample_count_is_part_of_run_path(configured, suffix):
    cfg = _cfg("vanilla")
    cfg["experiment"]["task"] = "tldr"
    cfg["training"]["tldr_train_samples"] = configured
    assert ArtifactLayout(cfg).root.parts[-3:] == Path(suffix).parts[-3:]


def test_vanilla_prefix_policy_and_shared_analysis_paths():
    cfg = _cfg("vanilla")
    layout = ArtifactLayout(cfg)
    assert training_prefix_enabled(cfg)
    assert post_finetuning_prefix_enabled(cfg)
    assert layout.policy_root.parts[-1] == "vanilla_original"
    assert "_shared" in layout.policy_config_dir.parts
    assert "_shared" in layout.policy_logs_dir.parts


@pytest.mark.parametrize("mode", ["unaware", "phase_aware", "gif_aware"])
def test_rotated_modes_have_both_prefix_stages(mode):
    cfg = _cfg(mode)
    assert training_prefix_enabled(cfg)
    assert post_finetuning_prefix_enabled(cfg)


def _layout_at(tmp_path):
    cfg = _cfg("vanilla")
    cfg["experiment"]["output_root"] = str(tmp_path)
    return ArtifactLayout(cfg)


def test_conversion_prefix_validation_requires_state(tmp_path):
    with pytest.raises(FileNotFoundError, match="discover_prefix"):
        validate_post_finetuning_prefix(_layout_at(tmp_path))


def test_empty_post_finetuning_prefix_needs_no_kv(tmp_path):
    layout = _layout_at(tmp_path)
    state = layout.post_finetuning_prefix_dir / "prefix_state.json"
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({"prefix_token_ids": []}), encoding="utf-8")
    metadata = validate_post_finetuning_prefix(layout)
    assert metadata["prefix_token_ids"] == []
    assert metadata["prefix_kv_sha256"] is None


def test_nonempty_post_finetuning_prefix_requires_kv(tmp_path):
    layout = _layout_at(tmp_path)
    state = layout.post_finetuning_prefix_dir / "prefix_state.json"
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({"prefix_token_ids": [123]}), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="KV cache"):
        validate_post_finetuning_prefix(layout)
