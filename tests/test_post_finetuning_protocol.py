import json

import pytest

from snn2.artifacts import ArtifactLayout
from snn2.config import post_finetuning_prefix_enabled, training_prefix_enabled
from snn2.conversion import validate_post_finetuning_prefix


def _cfg(mode):
    return {"experiment": {"id": "e", "task": "t", "model_name": "m", "seed": 42, "output_root": "artifacts", "ann_mode": mode}, "training": {"learning_rate": 1e-6}, "rotation": {"enabled": mode != "vanilla"}, "prefix": {"enabled": mode != "vanilla"}, "post_finetuning": {"prefix_enabled": True}}


def test_stage_specific_artifact_paths():
    layout = ArtifactLayout(_cfg("phase_aware"))
    assert "ann_training_prefix" in str(layout.ann_training_prefix_dir)
    assert "ann_training_calibration/sites" in str(layout.ann_training_site_dir)
    assert "vanilla_analysis_calibration/sites" in str(layout.vanilla_analysis_site_dir)
    assert "post_finetuning/prefix" in str(layout.post_finetuning_prefix_dir)
    assert "post_finetuning/conversion_calibration/sites" in str(layout.post_finetuning_site_dir)


@pytest.mark.parametrize(
    ("configured", "suffix"),
    [
        (None, "lr1e-06_train_samples_full/seed42"),
        (128, "lr1e-06_train_samples_128/seed42"),
    ],
)
def test_tldr_training_sample_count_is_part_of_run_path(configured, suffix):
    cfg = _cfg("vanilla")
    cfg["experiment"]["task"] = "tldr"
    cfg["training"]["tldr_train_samples"] = configured
    assert str(ArtifactLayout(cfg).root).endswith(suffix)


def test_vanilla_prefix_policy_and_shared_analysis_paths():
    cfg = _cfg("vanilla")
    layout = ArtifactLayout(cfg)
    assert not training_prefix_enabled(cfg)
    assert post_finetuning_prefix_enabled(cfg)
    assert str(layout.policy_root).endswith("vanilla_original")
    assert "_shared" in str(layout.policy_config_dir)
    assert "_shared" in str(layout.policy_logs_dir)


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
