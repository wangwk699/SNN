from pathlib import Path

import yaml

from snn2.sites import SITE_COUNT


def test_generated_configs_preserve_qwen3_1_7b_overrides():
    root = Path(__file__).resolve().parents[1] / "configs" / "generated"
    expected = {
        "exp1_qwen3_1_7b_tldr__vanilla.yaml": (1e-6, 32),
        "exp1_qwen3_1_7b_tldr__unaware.yaml": (5e-6, 32),
        "exp1_qwen3_1_7b_tldr__phase_aware.yaml": (1e-6, 32),
        "exp1_qwen3_1_7b_tldr__gif_aware.yaml": (1e-6, 32),
    }
    configs = sorted(root.glob("*.yaml"))
    assert len(configs) == 12
    for path in configs:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert int(cfg["calibration"]["expected_sites_per_layer"]) == SITE_COUNT
        assert float(cfg["rotation"]["regression_relative_l2_threshold"]) == 0.05
        assert float(cfg["rotation"]["regression_top1_agreement_threshold"]) == 0.95
        assert cfg["post_finetuning"] == {
            "rediscover_prefix": True,
            "recalibrate_sites": True,
            "prefix_enabled": True,
            "post_finetuning_recalibration": True,
        }
    for name, (learning_rate, batch_size) in expected.items():
        cfg = yaml.safe_load((root / name).read_text(encoding="utf-8"))
        assert float(cfg["training"]["learning_rate"]) == learning_rate
        assert int(cfg["evaluation"]["batch_size"]) == batch_size
