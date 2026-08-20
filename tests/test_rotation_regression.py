import json
import math
from types import SimpleNamespace

import pytest
import torch

from snn2.rotation import (
    RotationRegressionError,
    compute_logits_error_metrics,
    enforce_rotation_regression,
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
    json.dumps(metrics)


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
