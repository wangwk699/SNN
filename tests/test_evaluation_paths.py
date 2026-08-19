import pytest

from snn2.evaluation import resolve_tldr_evaluation_layout


@pytest.mark.parametrize(
    ("configured", "selected", "is_full", "dirname"),
    [
        (None, 6553, True, "test_samples_6553_full"),
        (128, 128, False, "test_samples_128"),
        (9999, 6553, True, "test_samples_6553_full"),
    ],
)
def test_resolve_tldr_evaluation_layout(configured, selected, is_full, dirname):
    layout = resolve_tldr_evaluation_layout(6553, configured)
    assert layout == {
        "selected_test_samples": selected,
        "is_full_test": is_full,
        "dirname": dirname,
    }


@pytest.mark.parametrize("configured", [0, -1])
def test_resolve_tldr_evaluation_layout_rejects_non_positive_counts(configured):
    with pytest.raises(ValueError):
        resolve_tldr_evaluation_layout(6553, configured)
