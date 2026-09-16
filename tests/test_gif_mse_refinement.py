from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from snn2.artifacts import ArtifactLayout, gif_qparam_calibration_suffix
from snn2.config import load_config, resolve_config, validate_config
from snn2.gif_mse_calibration import (
    GIFMSEHistogramStore,
    normalize_mse_refinement_config,
    optimize_static_qparams,
    runtime_fake_quant,
)


def _histogram(values: torch.Tensor, bins: int = 256):
    counts, edges = torch.histogram(values.double(), bins=bins)
    return {
        "counts": counts.long(),
        "bin_edges": edges,
        "sample_count": values.numel(),
        "sum_sq": float(values.double().square().sum()),
    }


def _direct(values: torch.Tensor, qmax: int):
    scale = max(float(values.max() - values.min()) / qmax, 1e-8)
    zero = int(torch.round(torch.tensor(-float(values.min()) / scale)).clamp(0, qmax))
    return scale, zero


def _optimize(values: torch.Tensor, qmax: int = 15):
    scale, zero = _direct(values, qmax)
    return optimize_static_qparams(
        _histogram(values), direct_scale=scale, direct_zero=zero,
        qmin=0, qmax=qmax,
        config=normalize_mse_refinement_config({"histogram_bins": 256}),
    )


def test_synthetic_outlier_improves_over_direct_min_max():
    values = torch.cat((torch.linspace(-1.0, 1.0, 10_000), torch.tensor([10.0])))
    result = _optimize(values)
    assert result["refinement_applied"]
    assert result["refined_mse"] < result["baseline_mse"]


@pytest.mark.parametrize("values", [torch.linspace(0.1, 3.0, 1000), torch.linspace(-3.0, -0.1, 1000)])
def test_one_sided_ranges_preserve_zero_and_never_regress(values):
    result = _optimize(values)
    assert result["refined_mse"] <= result["baseline_mse"] + 1e-12
    zero = runtime_fake_quant(torch.tensor([0.0]), result["scale"], result["zero"], qmin=0, qmax=15)
    torch.testing.assert_close(zero, torch.zeros_like(zero))
    if torch.all(values > 0):
        assert result["optimized_lower"] == pytest.approx(0.0)
    else:
        assert result["optimized_upper"] == pytest.approx(0.0)


def test_empty_branch_is_direct_fallback_without_nan():
    result = optimize_static_qparams(
        {"counts": torch.zeros(8, dtype=torch.long), "bin_edges": torch.linspace(-1, 1, 9), "sample_count": 0, "sum_sq": 0.0},
        direct_scale=0.1, direct_zero=8, qmin=0, qmax=15,
        config=normalize_mse_refinement_config({"histogram_bins": 8}),
    )
    assert result == {
        "scale": 0.1, "zero": 8, "refinement_applied": False,
        "fallback_reason": "empty_branch", "sample_count": 0,
    }


def test_low_and_high_qmax_are_both_supported_and_deterministic():
    values = torch.cat((torch.linspace(-1, 1, 2000), torch.tensor([8.0])))
    low_a, low_b = _optimize(values, 15), _optimize(values, 15)
    high = _optimize(values, 30)
    assert low_a == low_b
    assert 0 <= low_a["zero"] <= 15
    assert 0 <= high["zero"] <= 30


@pytest.mark.parametrize("group_size,groups", [(128, 2), (64, 4), (-1, 1)])
def test_histogram_group_size_comes_from_configured_layout(group_size, groups):
    effective = 256 if group_size == -1 else group_size
    spec = {
        "layout_kind": "last_dim", "parameter_layout": "last_dim_grouped",
        "configured_group_size": group_size, "effective_group_size": effective,
        "mask_low_by_role": {"default": torch.ones(256, dtype=torch.bool)},
        "branches": ["low"], "low_lower": torch.full((groups,), -1.0),
        "low_upper": torch.full((groups,), 1.0),
    }
    store = GIFMSEHistogramStore({"layer_000/site_08": spec}, histogram_bins=16)
    store.update(0, 8, torch.zeros(2, 3, 256))
    assert store.data["layer_000/site_08"]["low"]["sample_count"].shape == (groups,)
    assert int(store.data["layer_000/site_08"]["low"]["sample_count"].sum()) == 2 * 3 * 256


def test_multi_role_contributes_to_shared_low_and_high_histograms():
    spec = {
        "layout_kind": "last_dim", "parameter_layout": "last_dim_grouped",
        "configured_group_size": 4, "effective_group_size": 4,
        "mask_low_by_role": {
            "q": torch.tensor([True, True, False, False]),
            "k": torch.tensor([False, False, True, True]),
        },
        "branches": ["low", "high"],
        "low_lower": torch.tensor([-2.0]), "low_upper": torch.tensor([2.0]),
        "high_lower": torch.tensor([-2.0]), "high_upper": torch.tensor([2.0]),
    }
    store = GIFMSEHistogramStore({"layer_000/site_01": spec}, histogram_bins=16)
    store.update(0, 1, torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]))
    low = store.data["layer_000/site_01"]["low"]
    high = store.data["layer_000/site_01"]["high"]
    assert int(low["sample_count"].sum()) == 4
    assert int(high["sample_count"].sum()) == 4


def test_old_config_defaults_and_validation():
    raw = load_config("configs/generated/exp2_llama3_8b_tulu3__gif_aware.yaml")
    assert raw["gif"]["mse_scale_refinement"] is False
    assert raw["gif"]["mse_refinement"]["version"] == "static_mse_v1"
    bad = deepcopy(raw)
    bad["gif"]["mse_refinement"]["coarse_alpha_values"] = [1.0, 1.0]
    with pytest.raises(ValueError, match="duplicates"):
        validate_config(bad)


def test_direct_paths_are_unchanged_and_mse_configs_are_isolated():
    direct = load_config("configs/generated/exp2_llama3_8b_tulu3__gif_aware.yaml")
    direct_layout = ArtifactLayout(direct)
    mse_x = deepcopy(direct)
    mse_x["gif"]["mse_scale_refinement"] = True
    mse_y = deepcopy(mse_x)
    mse_y["gif"]["mse_refinement"]["histogram_bins"] = 2048
    layout_x, layout_y = ArtifactLayout(mse_x), ArtifactLayout(mse_y)
    assert gif_qparam_calibration_suffix(direct) is None
    assert "gif_qparams_mse_refined_v1_" not in str(direct_layout.root)
    assert direct_layout.ann_training_calibration_dir != layout_x.ann_training_calibration_dir
    assert layout_x.ann_training_calibration_dir != layout_y.ann_training_calibration_dir
    assert layout_x.post_finetuning_conversion_calibration_dir != layout_y.post_finetuning_conversion_calibration_dir
    assert layout_x.root != layout_y.root


def test_group_size_and_mse_suffix_both_isolate_paths():
    cfg = load_config("configs/generated/exp2_llama3_8b_tulu3__gif_aware.yaml")
    cfg["gif"]["mse_scale_refinement"] = True
    group_128 = ArtifactLayout(cfg)
    cfg_64 = deepcopy(cfg)
    cfg_64["calibration"]["group_size"] = 64
    group_64 = ArtifactLayout(cfg_64)
    assert group_128.ann_training_calibration_dir != group_64.ann_training_calibration_dir
    assert "calibration_group_size_128" in str(group_128.ann_training_calibration_dir)
    assert "calibration_group_size_64" in str(group_64.ann_training_calibration_dir)
