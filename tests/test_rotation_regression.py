import json
import math
from types import SimpleNamespace

import pytest
import torch

from snn2.data import encode_generation_prompt, encode_tldr_generation_prompt, tldr_prompt_and_reference
from snn2.hadamard import make_spec
from snn2.rotation import (
    _PromptEndDecisionAccumulator,
    _last_valid_logits,
    assess_rotation_regression,
    diagnose_rotation_comparisons,
    load_specs,
    RotationRegressionError,
    _StreamingAbsErrorHistogram,
    compute_logits_error_metrics,
    enforce_rotation_regression,
    fuse_rmsnorm_scale,
    validate_rotation_regression_suite,
    RotationRegressionSuiteError,
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


def test_margin_safe_token_has_top1_stability_guarantee():
    metrics = compute_logits_error_metrics(
        torch.tensor([[[10.0, 0.0, -1.0]]]),
        torch.tensor([[[9.5, 0.5, -1.0]]]),
        torch.tensor([[1]]),
    )
    diagnostic = metrics["margin_aware_diagnostic"]

    assert diagnostic["margin_safe_token_count"] == 1
    assert diagnostic["margin_unsafe_token_count"] == 0
    assert diagnostic["margin_safe_agreement_count"] == 1
    assert diagnostic["margin_safe_disagreement_count"] == 0
    assert diagnostic["margin_safe_fraction"] == pytest.approx(1.0)
    assert diagnostic["base_top1_margin_all_tokens"]["mean"] == pytest.approx(10.0)
    assert diagnostic["per_token_max_abs_error_all_tokens"]["max"] == pytest.approx(0.5)
    assert diagnostic["base_top1_margin_disagreement_tokens"] == {
        "count": 0,
        "mean": None,
        "p50": None,
        "p90": None,
        "p99": None,
        "max": None,
    }
    assert diagnostic["stability_ratio_disagreement_tokens"] == {
        "definition": "2_delta_over_base_top1_margin_plus_1e-12",
        "count": 0,
        "mean": None,
        "p50": None,
        "p90": None,
        "p99": None,
    }
    json.dumps(metrics)


def test_margin_unsafe_tokens_can_disagree_or_still_agree():
    base = torch.tensor([[[10.0, 9.9, 0.0], [10.0, 9.9, 0.0]]])
    rotated = torch.tensor([[[9.9, 10.0, 0.0], [10.2, 9.8, 0.0]]])
    metrics = compute_logits_error_metrics(base, rotated, torch.tensor([[1, 1]]))
    diagnostic = metrics["margin_aware_diagnostic"]

    assert diagnostic["margin_safe_token_count"] == 0
    assert diagnostic["margin_unsafe_token_count"] == 2
    assert diagnostic["margin_unsafe_agreement_count"] == 1
    assert diagnostic["margin_unsafe_disagreement_count"] == 1
    assert diagnostic["margin_safe_disagreement_count"] == 0
    assert diagnostic["disagreement_margin_unsafe_fraction"] == pytest.approx(1.0)
    assert diagnostic["base_top1_margin_disagreement_tokens"]["count"] == 1
    assert diagnostic["per_token_max_abs_error_disagreement_tokens"]["count"] == 1
    ratio = diagnostic["stability_ratio_disagreement_tokens"]
    assert ratio["count"] == 1
    assert ratio["mean"] == pytest.approx(2.0)


def test_margin_diagnostics_exclude_padding_and_reconcile_partitions():
    base = torch.tensor(
        [[[10.0, 0.0, -1.0], [100.0, 0.0, -1.0], [20.0, 19.9, 0.0]]]
    )
    rotated = torch.tensor(
        [[[9.5, 0.5, -1.0], [0.0, 200.0, -1.0], [19.8, 20.1, 0.0]]]
    )
    metrics = compute_logits_error_metrics(base, rotated, torch.tensor([[1, 0, 1]]))
    diagnostic = metrics["margin_aware_diagnostic"]

    assert metrics["num_tokens_compared"] == 2
    assert metrics["top1_agreement_count"] == 1
    assert metrics["top1_disagreement_count"] == 1
    assert diagnostic["margin_safe_token_count"] == 1
    assert diagnostic["margin_unsafe_token_count"] == 1
    assert diagnostic["margin_safe_agreement_count"] == 1
    assert diagnostic["margin_safe_disagreement_count"] == 0
    assert diagnostic["margin_unsafe_agreement_count"] == 0
    assert diagnostic["margin_unsafe_disagreement_count"] == 1
    assert (
        diagnostic["margin_safe_token_count"] + diagnostic["margin_unsafe_token_count"]
        == metrics["num_tokens_compared"]
    )
    assert (
        diagnostic["margin_safe_agreement_count"]
        + diagnostic["margin_unsafe_agreement_count"]
        == metrics["top1_agreement_count"]
    )
    assert (
        diagnostic["margin_safe_disagreement_count"]
        + diagnostic["margin_unsafe_disagreement_count"]
        == metrics["top1_disagreement_count"]
    )
    all_delta = diagnostic["per_token_max_abs_error_all_tokens"]
    assert all_delta["max"] == pytest.approx(metrics["max_abs_error"])
    disagreement_distributions = (
        diagnostic["base_top1_margin_disagreement_tokens"],
        diagnostic["per_token_max_abs_error_disagreement_tokens"],
        diagnostic["stability_ratio_disagreement_tokens"],
    )
    for distribution in disagreement_distributions:
        assert distribution["count"] == metrics["top1_disagreement_count"]
        assert distribution["p50"] <= distribution["p90"] <= distribution["p99"]
    assert all_delta["p50"] <= all_delta["p90"] <= all_delta["p99"] <= all_delta["max"]


@pytest.mark.parametrize(
    ("relative_l2_error", "top1_agreement", "passed"),
    [
        (0.04, 0.96, True),
        (0.06, 0.99, False),
        (0.01, 0.94, False),
        (0.01, 0.95, False),
        (0.05, 0.96, True),
    ],
)
def test_rotation_regression_uses_joint_hard_gates(
    relative_l2_error, top1_agreement, passed
):
    result = {
        "relative_l2_error": relative_l2_error,
        "top1_agreement": top1_agreement,
        "num_samples": 128,
    }
    if not passed:
        with pytest.raises(RotationRegressionError) as caught:
            enforce_rotation_regression(
                result,
                relative_l2_threshold=0.05,
                top1_agreement_threshold=0.95,
            )
        checked = caught.value.result
    else:
        checked = enforce_rotation_regression(
            result,
            relative_l2_threshold=0.05,
            top1_agreement_threshold=0.95,
        )

    assert checked["passed"] is passed
    assert checked["threshold"] == {
        "relative_l2_error": 0.05,
        "top1_agreement": 0.95,
    }
    if not passed:
        message = str(caught.value)
        assert "relative_l2_error=" in message
        assert "top1_agreement=" in message


def _comparison(passed, prompt_top1=1.0):
    return {
        "passed": passed,
        "all_tokens": {"passed": passed},
        "prompt_end": {"top1_agreement": prompt_top1, "gating": False},
    }


@pytest.mark.parametrize(
    ("states", "code"),
    [
        ((True, True, True), "no_regression_detected"),
        ((False, True, False), "snn2_integration_mismatch"),
        ((True, False, False), "rotation_mismatch"),
        ((False, False, False), "integration_and_rotation_mismatch"),
        ((False, False, True), "integration_and_rotation_mismatch"),
        ((True, True, False), "end_to_end_accumulation_mismatch"),
        ((False, True, True), "mixed_regression_failure"),
    ],
)
def test_three_way_diagnosis_mapping_keeps_all_pairs(states, code):
    comparisons = {
        name: _comparison(passed)
        for name, passed in zip(("A_vs_B", "B_vs_C", "A_vs_C"), states)
    }
    diagnosis = diagnose_rotation_comparisons(comparisons)

    assert set(diagnosis["pair_pass"]) == {"A_vs_B", "B_vs_C", "A_vs_C"}
    assert diagnosis["code"] == code
    if states == (False, False, True):
        assert "possible cancellation" in diagnosis["summary"]


def test_assessment_does_not_raise_on_failure():
    checked = assess_rotation_regression(
        {"relative_l2_error": 0.06, "top1_agreement": 0.99},
        relative_l2_threshold=0.05,
        top1_agreement_threshold=0.95,
    )
    assert checked["passed"] is False


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        (torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]), torch.tensor([[2.0, -2.0], [7.0, -7.0]])),
        (torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]]), torch.tensor([[4.0, -4.0], [8.0, -8.0]])),
    ],
)
def test_prompt_end_selects_last_valid_position_for_both_padding_sides(mask, expected):
    values = torch.arange(1, 9, dtype=torch.float32).reshape(2, 4)
    logits = torch.stack((values, -values), dim=-1)
    torch.testing.assert_close(_last_valid_logits(logits, mask), expected)


def test_prompt_end_diagnostic_does_not_gate_pair():
    accumulator = _PromptEndDecisionAccumulator()
    accumulator.update(
        torch.tensor([[10.0, 0.0], [10.0, 0.0]]),
        torch.tensor([[0.0, 10.0], [9.0, 0.0]]),
        [0, 1],
        [100, 101],
    )
    prompt_end = accumulator.metrics()
    all_tokens = assess_rotation_regression(
        {"relative_l2_error": 0.01, "top1_agreement": 0.99}, 0.05, 0.95
    )
    pair = {"all_tokens": all_tokens, "prompt_end": prompt_end, "passed": all_tokens["passed"]}

    assert prompt_end["top1_agreement"] == pytest.approx(0.5)
    assert prompt_end["gating"] is False
    assert pair["passed"] is True
    assert prompt_end["mismatch_examples"][0]["dataset_index"] == 100


def test_prompt_only_encoding_excludes_completion_and_uses_evaluation_length():
    calls = []

    class Tokenizer:
        def encode(self, text, **kwargs):
            calls.append((text, kwargs))
            return [1, 2, 3]

    row = {"prompt": "PROMPT", "completion": "COMPLETION"}
    cfg = {"evaluation": {"tldr_input_length": 77}, "data": {"max_seq_length": 999}}
    prompt, reference = tldr_prompt_and_reference(row)
    ids = encode_tldr_generation_prompt(row, Tokenizer(), cfg)

    assert (prompt, reference) == ("PROMPT", "COMPLETION")
    assert ids == [1, 2, 3]
    assert calls == [
        (
            "PROMPT",
            {"add_special_tokens": True, "truncation": True, "max_length": 77},
        )
    ]


def test_rotation_state_rejects_legacy_and_accepts_du_metadata():
    specs = {name: make_spec(name, 8, seed).state_dict() for name, seed in (("R3", 1), ("R4", 2))}
    with pytest.raises(RuntimeError, match="Legacy/incompatible"):
        load_specs({"format_version": 1, "specs": specs})

    loaded = load_specs(
        {
            "format_version": 2,
            "random_hadamard_orientation": "DU",
            "precision_policy": "roste_aligned_v1",
            "specs": specs,
        }
    )
    assert set(loaded) == {"R3", "R4"}
    assert all(spec.orientation == "DU" for spec in loaded.values())


class _FakeDataset(list):
    @property
    def column_names(self):
        return list(self[0].keys())

    def map(self, function, remove_columns=None, desc=None):
        return _FakeDataset(function(row) for row in self)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    padding_side = "right"

    def encode(self, text, add_special_tokens=True, truncation=False, max_length=None):
        ids = [1, 2] if add_special_tokens else [2]
        return ids[:max_length] if truncation and max_length is not None else ids

    def pad(self, encoded, padding=True, return_tensors="pt"):
        rows = encoded["input_ids"]
        width = max(len(row) for row in rows)
        padded, masks = [], []
        for row in rows:
            count = width - len(row)
            if self.padding_side == "left":
                padded.append([0] * count + list(row))
                masks.append([0] * count + [1] * len(row))
            else:
                padded.append(list(row) + [0] * count)
                masks.append([1] * len(row) + [0] * count)
        return {"input_ids": torch.tensor(padded), "attention_mask": torch.tensor(masks)}


class _FakeRegressionModel(torch.nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.embedding = torch.nn.Embedding(3, 2)
        self.variant = variant
        attrs = {}
        if variant == "B":
            attrs["snn2_site_integration"] = True
        if variant == "C":
            attrs.update(
                snn2_site_integration=True,
                snn2_rotation_fused=True,
                snn2_online_rotations=["R3", "R4"],
                snn2_rotation_format_version=2,
                snn2_random_hadamard_orientation="DU",
                snn2_rotation_precision_policy="roste_aligned_v1",
            )
        self.config = SimpleNamespace(**attrs)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, use_cache=False):
        leading = 0 if self.variant == "A" else 1
        logits = torch.zeros((*input_ids.shape, 3), device=input_ids.device)
        logits[..., leading] = 10.0
        return SimpleNamespace(logits=logits)


def test_three_way_suite_finishes_all_pairs_before_raising(tmp_path):
    dataset = _FakeDataset(
        {"prompt": f"prompt {index}", "completion": "completion"}
        for index in range(128)
    )
    manifest = tmp_path / "calibration_manifest.json"
    manifest.write_text(json.dumps({"indices": list(range(1000, 1128))}), encoding="utf-8")
    cfg = {
        "calibration": {"num_samples": 128, "batch_size": 32},
        "evaluation": {"tldr_input_length": 16},
        "data": {"max_seq_length": 16, "truncation_side": "right"},
        "rotation": {
            "seed": 42,
            "regression_relative_l2_threshold": 0.05,
            "regression_top1_agreement_threshold": 0.95,
        },
        "training": {"dtype": "float32"},
        "experiment": {"model_name": "fake", "task": "tldr"},
    }
    identity = SimpleNamespace(mode="identity")

    with pytest.raises(RotationRegressionSuiteError) as caught:
        validate_rotation_regression_suite(
            _FakeRegressionModel("A"),
            _FakeRegressionModel("B"),
            _FakeRegressionModel("C"),
            _FakeTokenizer(),
            dataset,
            cfg,
            identity,
            identity,
            calibration_manifest_path=manifest,
            calibration_manifest_sha256="hash",
        )

    result = caught.value.result
    assert set(result["comparisons"]) == {"A_vs_B", "B_vs_C", "A_vs_C"}
    assert result["diagnosis"]["code"] == "snn2_integration_mismatch"
    assert result["comparisons"]["B_vs_C"]["passed"] is True
    for pair in result["comparisons"].values():
        assert pair["prompt_end"]["num_prompts_compared"] == 128
        assert pair["prompt_end"]["gating"] is False


def test_prompt_end_identical_tied_logits_use_consistent_argmax():
    accumulator = _PromptEndDecisionAccumulator()
    logits = torch.tensor([[1.0, 1.0, 0.0]])
    accumulator.update(logits, logits.clone(), [0], [42])
    metrics = accumulator.metrics()

    assert metrics["top1_agreement"] == 1.0
    assert metrics["top1_disagreement_count"] == 0
    assert metrics["mismatch_examples"] == []


def test_tulu_prompt_encoding_excludes_final_assistant_response():
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return [4, 5]

    row = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    cfg = {"experiment": {"task": "tulu3"}}

    assert encode_generation_prompt(row, Tokenizer(), cfg) == [4, 5]
    assert calls == [
        (
            [{"role": "user", "content": "question"}],
            {"tokenize": True, "add_generation_prompt": True},
        )
    ]
