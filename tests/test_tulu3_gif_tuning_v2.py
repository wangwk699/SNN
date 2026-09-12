from decimal import Decimal
from pathlib import Path

import yaml

from scripts.run_tulu3_gif_aware_tuning_v2 import (
    CALIBRATION_NUM_SAMPLES,
    GIF_RATIO_PAIRS,
    candidate_config,
    candidate_label,
    fixed_base_config,
)
from snn2.artifacts import ArtifactLayout


SOURCE = Path("configs/generated/exp2_llama3_8b_tulu3__gif_aware.yaml")


def _base():
    source = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    return fixed_base_config(source)


def test_v2_grid_has_twelve_unique_candidates_and_paths() -> None:
    base = _base()
    labels = set()
    ann_dirs = set()
    calibration_dirs = set()

    for num_samples in CALIBRATION_NUM_SAMPLES:
        for low_ratio, salient_ratio in GIF_RATIO_PAIRS:
            cfg = candidate_config(
                base,
                num_samples=num_samples,
                low_ratio=low_ratio,
                salient_ratio=salient_ratio,
            )
            labels.add(candidate_label(num_samples, low_ratio, salient_ratio))
            layout = ArtifactLayout(cfg)
            ann_dirs.add(layout.ann_dir)
            calibration_dirs.add(layout.ann_training_calibration_dir)

    assert len(labels) == 12
    assert len(ann_dirs) == 12
    assert len(calibration_dirs) == 12


def test_v2_fixed_settings_and_ratio_scoped_paths() -> None:
    cfg = candidate_config(
        _base(),
        num_samples=512,
        low_ratio=Decimal("0.7"),
        salient_ratio=Decimal("0.3"),
    )
    layout = ArtifactLayout(cfg)

    assert cfg["training"]["train_samples"] == 10000
    assert cfg["training"]["learning_rate"] == 5e-6
    assert cfg["training"]["lr_scheduler_type"] == "cosine"
    assert cfg["training"]["warmup_ratio"] == 0.01
    assert cfg["training"]["gradient_accumulation_steps"] == 16
    assert cfg["ann_training_memory"] == {
        "attention_core_checkpoint": True,
        "mlp_checkpoint": True,
    }
    assert layout.ann_training_calibration_dir.name == (
        "calibration_group_size_128_num_samples_512_gif_low_ratio_0.7"
    )
    assert (
        "epochs_1_num_samples_512_gif_low_ratio_0.7_"
        "lr5e-06_train_samples_10000_calibration_group_size_128"
    ) in layout.ann_dir.parts
    assert "tuning_v2_calibration_ratio_grid" in layout.ann_dir.parts
