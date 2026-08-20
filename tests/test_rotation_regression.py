import json
import math
from types import SimpleNamespace

import pytest
import torch

from snn2.rotation import (
    RotationRegressionError,
    _StreamingAbsErrorHistogram,
    compute_logits_error_metrics,
    enforce_rotation_regression,
    fuse_rmsnorm_scale,
    validate_rotation_logits,
)


def test_rotation_logits_metrics_ignore_padding_and_are_serializable():
    base = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    rotated = torch.tensor([[[0.0, 2.0], [30.0, 40.0]]])
    attention_mask = torch.tensor([[1, 0]])

    metrics = compute_logits_error_metrics(base, rotated, attention_mask)

    assert metrics["num_samples"] == 1
    assert metrics["num_tokens_compared"] == 1
    assert metrics["max_abs_error"] == pytest.approx(1.0)
    assert metrics["mean_abs_error"] == pytest.approx(0.5)
    assert metrics["relative_l2_error"] == pytest.approx(1.0 / math.sqrt(5.0))
    assert metrics["top1_agreement"] == pytest.approx(1.0)
    assert metrics["top1_agreement_count"] == 1
    assert metrics["top1_disagreement_count"] == 0
    assert metrics["p99_abs_error"] <= metrics["p999_abs_error"]
    estimator = metrics["absolute_error_percentile_estimator"]
    bin_width = estimator["final_range_max"] / estimator["num_bins"]
    assert metrics["p999_abs_error"] <= metrics["max_abs_error"] + bin_width + 1e-12
    assert estimator == {
        "method": "streaming_linear_histogram",
        "num_bins": 8192,
        "final_range_max": 1.0,
        "reported_value": "bin_upper_edge",
        "exact": False,
    }
    json.dumps(metrics)


def test_rotation_logits_metrics_record_top1_disagreement():
    metrics = compute_logits_error_metrics(
        torch.tensor([[[10.0, 0.0]]]),
        torch.tensor([[[0.0, 10.0]]]),
        torch.tensor([[1]]),
    )

    assert metrics["top1_agreement"] == pytest.approx(0.0)
    assert metrics["top1_agreement_count"] == 0
    assert metrics["top1_disagreement_count"] == 1


def test_rotation_logits_top1_ignores_padding_disagreement():
    metrics = compute_logits_error_metrics(
        torch.tensor([[[10.0, 0.0], [10.0, 0.0]]]),
        torch.tensor([[[9.0, 0.0], [0.0, 10.0]]]),
        torch.tensor([[1, 0]]),
    )

    assert metrics["num_tokens_compared"] == 1
    assert metrics["top1_agreement"] == pytest.approx(1.0)
    assert metrics["top1_agreement_count"] == 1
    assert metrics["top1_disagreement_count"] == 0


def test_streaming_histogram_expands_and_preserves_counts():
    histogram = _StreamingAbsErrorHistogram(num_bins=8, initial_max=1.0)
    histogram.update(torch.tensor([0.1, 0.5, 0.9]))
    histogram.update(torch.tensor([2.5]))

    assert histogram.range_max == pytest.approx(4.0)
    assert histogram.total_count == 4
    assert int(histogram.counts.sum().item()) == histogram.total_count
    assert 0.0 <= histogram.percentile(0.99) <= histogram.percentile(0.999)


def test_streaming_histogram_multiple_doublings_preserve_counts():
    histogram = _StreamingAbsErrorHistogram(num_bins=8, initial_max=1.0)
    histogram.update(torch.tensor([0.0, 0.25, 0.75]))
    histogram.update(torch.tensor([9.0, 17.0]))

    assert histogram.range_max == pytest.approx(32.0)
    assert histogram.total_count == 5
    assert int(histogram.counts.sum().item()) == histogram.total_count
    assert histogram.percentile(0.99) <= histogram.percentile(0.999)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_fuse_rmsnorm_scale_supports_cross_device_linear():
    norm = torch.nn.Module()
    norm.weight = torch.nn.Parameter(
        torch.tensor([2.0, 3.0, 4.0], device="cuda:0")
    )
    linear = torch.nn.Linear(3, 2, bias=False, device="cuda:1")
    original = linear.weight.detach().cpu().double()

    fuse_rmsnorm_scale(norm, [linear])

    expected = original * torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
    assert linear.weight.device == torch.device("cuda:1")
    torch.testing.assert_close(linear.weight.detach().cpu().double(), expected)
    torch.testing.assert_close(norm.weight.detach().cpu(), torch.ones(3))


def test_rotation_regression_passes_at_threshold_and_is_serializable():
    result = enforce_rotation_regression(
        {"relative_l2_error": 0.005, "num_samples": 128},
        relative_l2_threshold=0.01,
    )

    assert result["passed"] is True
    assert result["threshold"] == {"relative_l2_error": 0.01}
    json.dumps(result)


def test_rotation_regression_hard_fails_and_carries_result():
    with pytest.raises(RotationRegressionError) as caught:
        enforce_rotation_regression(
            {"relative_l2_error": 0.02, "num_samples": 128},
            relative_l2_threshold=0.01,
        )

    assert caught.value.result["passed"] is False
    assert caught.value.result["threshold"] == {"relative_l2_error": 0.01}


def test_rotation_regression_requires_identity_controller():
    cfg = {"calibration": {"num_samples": 128}}

    with pytest.raises(RuntimeError, match=r"SiteController\(mode='identity'\)"):
        validate_rotation_logits(
            None,
            None,
            None,
            [None] * 128,
            cfg,
            SimpleNamespace(mode="phase"),
            calibration_manifest_path="calibration_manifest.json",
            calibration_manifest_sha256="unused",
        )


def test_rotation_regression_requires_online_r3_and_r4():
    cfg = {"calibration": {"num_samples": 128}}
    rotated_model = SimpleNamespace(
        config=SimpleNamespace(
            snn2_site_integration=True,
            snn2_online_rotations=["R3"],
        )
    )

    with pytest.raises(RuntimeError, match="online R3 and R4"):
        validate_rotation_logits(
            None,
            rotated_model,
            None,
            [None] * 128,
            cfg,
            SimpleNamespace(mode="identity"),
            calibration_manifest_path="calibration_manifest.json",
            calibration_manifest_sha256="unused",
        )
