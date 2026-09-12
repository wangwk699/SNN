import pytest

from snn2.data import (
    _causal_supervised_token_count,
    _truncate_with_assistant_fallback,
)


def test_right_truncation_falls_back_to_final_supervised_window() -> None:
    input_ids = list(range(10))
    labels = [-100] * 8 + [8, 9]

    truncated_ids, truncated_labels, applied = _truncate_with_assistant_fallback(
        input_ids,
        labels,
        max_length=4,
        truncation_side="right",
        assistant_aware=True,
    )

    assert truncated_ids == [6, 7, 8, 9]
    assert truncated_labels == [-100, -100, 8, 9]
    assert applied is True
    assert _causal_supervised_token_count(truncated_labels) == 2


def test_right_truncation_keeps_original_window_when_supervised() -> None:
    input_ids = list(range(8))
    labels = [-100, -100, 2, 3, -100, -100, 6, 7]

    truncated_ids, truncated_labels, applied = _truncate_with_assistant_fallback(
        input_ids,
        labels,
        max_length=4,
        truncation_side="right",
        assistant_aware=True,
    )

    assert truncated_ids == [0, 1, 2, 3]
    assert truncated_labels == [-100, -100, 2, 3]
    assert applied is False


def test_assistant_aware_fallback_rejects_no_supervision() -> None:
    with pytest.raises(ValueError, match="no causal assistant supervision"):
        _truncate_with_assistant_fallback(
            list(range(8)),
            [-100] * 8,
            max_length=4,
            truncation_side="right",
            assistant_aware=True,
        )
