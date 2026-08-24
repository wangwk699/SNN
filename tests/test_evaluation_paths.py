import random
from types import SimpleNamespace

from snn2.data import load_selected_raw, prepare_manifests
import pytest

from snn2.artifacts import prefix_enabled_dirname
from snn2.evaluation import (
    activation_neuron_operators_per_temporal_forward,
    resolve_tldr_evaluation_layout,
)


@pytest.mark.parametrize(
    ("neuron", "expected"),
    [("phase", 281), ("gif", 280), ("mtn", 280), ("ann", 280)],
)
def test_activation_neuron_operator_count_includes_global_phase(neuron, expected):
    assert activation_neuron_operators_per_temporal_forward(
        num_hidden_layers=28, neuron=neuron
    ) == expected


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

@pytest.mark.parametrize(
    ("configured", "enabled", "path"),
    [
        (128, True, "test_samples_128/prefix_enabled_ture"),
        (128, False, "test_samples_128/prefix_enabled_false"),
        (None, True, "test_samples_6553_full/prefix_enabled_ture"),
        (None, False, "test_samples_6553_full/prefix_enabled_false"),
    ],
)
def test_rotated_pre_finetuning_prefix_result_path(configured, enabled, path):
    layout = resolve_tldr_evaluation_layout(6553, configured)
    assert f"{layout['dirname']}/{prefix_enabled_dirname(enabled)}" == path



class _Dataset(list):
    column_names = ()

    def select(self, indices):
        return _Dataset(self[index] for index in indices)


def test_tulu_full_training_uses_all_rows_except_fixed_validation(monkeypatch, tmp_path):
    raw = {"train": _Dataset({"value": index} for index in range(20))}
    monkeypatch.setattr("snn2.data._load_raw", lambda cfg: raw)
    cfg = {
        "experiment": {"task": "tulu3", "seed": 7},
        "data": {
            "dataset_name": "fake/tulu",
            "train_split": "train",
            "train_size": None,
            "validation_size": 5,
        },
        "training": {},
        "calibration": {"seed": 42, "num_samples": 4, "with_replacement": False},
    }
    manifests = prepare_manifests(cfg, SimpleNamespace(data_dir=tmp_path))
    assert len(manifests["train"]["indices"]) == 15
    assert len(manifests["validation"]["indices"]) == 5
    assert set(manifests["train"]["indices"]).isdisjoint(
        manifests["validation"]["indices"]
    )
    assert set(manifests["train"]["indices"]) | set(
        manifests["validation"]["indices"]
    ) == set(range(20))


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


def test_ann_training_subset_uses_current_config_even_when_shared_manifest_is_full(
    monkeypatch, tmp_path
):
    raw = {
        "train": _Dataset({"value": index} for index in range(20)),
        "validation": _Dataset({"value": index} for index in range(5)),
        "test": _Dataset({"value": index} for index in range(6)),
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
        "training": {"tldr_train_samples": None, "tldr_train_seed": 42},
        "calibration": {"seed": 42, "num_samples": 4, "with_replacement": False},
    }
    layout = SimpleNamespace(data_dir=tmp_path)
    prepare_manifests(cfg, layout)
    assert len(load_selected_raw(cfg, layout).train) == 20

    cfg["training"]["tldr_train_samples"] = 8
    bundle = load_selected_raw(cfg, layout, use_configured_train_subset=True)
    expected = random.Random(42).sample(range(20), k=8)
    expected.sort()

    assert len(bundle.train) == 8
    assert [row["value"] for row in bundle.train] == expected
    assert bundle.manifests["train"]["indices"] == expected
    assert bundle.manifests["train"]["selection_scope"] == "current_ann_training_config"
    assert len(bundle.calibration) == 4


def test_ann_training_subset_rejects_more_rows_than_raw_split(monkeypatch, tmp_path):
    raw = {
        "train": _Dataset({"value": index} for index in range(5)),
        "validation": _Dataset({"value": index} for index in range(2)),
        "test": _Dataset({"value": index} for index in range(2)),
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
        "training": {"tldr_train_samples": None, "tldr_train_seed": 42},
        "calibration": {"seed": 42, "num_samples": 2, "with_replacement": False},
    }
    layout = SimpleNamespace(data_dir=tmp_path)
    prepare_manifests(cfg, layout)
    cfg["training"]["tldr_train_samples"] = 6

    with pytest.raises(ValueError, match="contains only 5 rows"):
        load_selected_raw(cfg, layout, use_configured_train_subset=True)
