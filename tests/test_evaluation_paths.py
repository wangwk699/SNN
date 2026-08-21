import random
from types import SimpleNamespace

from snn2.data import prepare_manifests
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



class _Dataset(list):
    column_names = ()

    def select(self, indices):
        return _Dataset(self[index] for index in indices)


def test_tldr_train_subset_is_fixed_random_without_replacement(monkeypatch, tmp_path):
    raw = {
        "train": _Dataset({} for _ in range(20)),
        "validation": _Dataset({} for _ in range(5)),
        "test": _Dataset({} for _ in range(6)),
    }
    monkeypatch.setattr("snn2.data._load_raw", lambda cfg: raw)
    cfg = {
        "experiment": {"task": "tldr", "seed": 7},
        "data": {
            "dataset_name": "fake/tldr",
            "train_split": "train",
            "validation_split": "validation",
            "evaluation_split": "test",
        },
        "training": {"tldr_train_samples": 8, "tldr_train_seed": 42},
        "calibration": {"seed": 42, "num_samples": 4, "with_replacement": False},
    }
    manifests = prepare_manifests(cfg, SimpleNamespace(data_dir=tmp_path))
    expected = random.Random(42).sample(range(20), k=8)
    expected.sort()

    assert manifests["train"]["indices"] == expected
    assert len(set(manifests["train"]["indices"])) == 8
    assert manifests["train"]["sampling"] == "seeded_random_without_replacement"
    assert manifests["train"]["tldr_train_seed"] == 42
    assert set(manifests["calibration"]["indices"]) <= set(expected)
