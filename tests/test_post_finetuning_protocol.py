from snn2.artifacts import ArtifactLayout
from snn2.config import post_finetuning_prefix_enabled, training_prefix_enabled


def _cfg(mode):
    return {"experiment": {"id": "e", "task": "t", "model_name": "m", "seed": 42, "output_root": "artifacts", "ann_mode": mode}, "training": {"learning_rate": 1e-6}, "rotation": {"enabled": mode != "vanilla"}, "prefix": {"enabled": mode != "vanilla"}, "post_finetuning": {"prefix_enabled": True}}


def test_stage_specific_artifact_paths():
    layout = ArtifactLayout(_cfg("phase_aware"))
    assert "ann_training_prefix" in str(layout.ann_training_prefix_dir)
    assert "ann_training_calibration/sites" in str(layout.ann_training_site_dir)
    assert "vanilla_analysis_calibration/sites" in str(layout.vanilla_analysis_site_dir)
    assert "post_finetuning/prefix" in str(layout.post_finetuning_prefix_dir)
    assert "post_finetuning/conversion_calibration/sites" in str(layout.post_finetuning_site_dir)


def test_vanilla_has_no_training_prefix_but_has_post_finetuning_prefix():
    cfg = _cfg("vanilla")
    assert not training_prefix_enabled(cfg)
    assert post_finetuning_prefix_enabled(cfg)


def test_rotated_mode_has_both_prefix_stages():
    cfg = _cfg("gif_aware")
    assert training_prefix_enabled(cfg)
    assert post_finetuning_prefix_enabled(cfg)
