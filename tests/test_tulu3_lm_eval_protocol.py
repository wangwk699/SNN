import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pytest

from snn2.data import _manifest_split_selection, prepare_manifests
from snn2.lm_eval_protocol import (build_test_selection, correct_effective_sample_counts, prune_empty_selected_leaves,
    selection_by_leaf, validate_lm_eval_task_specs)


def _cfg(samples=10, seed=42):
    return {"experiment": {"task": "tulu3", "seed": 7}, "data": {"train_split": "train", "validation_size": 5}, "training": {"train_samples": samples, "train_seed": seed}}


def test_tulu_validation_precedes_independent_training_subset():
    raw = {"train": list(range(40))}
    _, _, train, sampling, _, validation = _manifest_split_selection(_cfg(), raw)
    assert len(train) == 10 and len(validation) == 5
    assert sampling == "seeded_random_without_replacement"
    assert set(train).isdisjoint(validation) and len(set(train)) == 10
    _, _, changed, _, _, same_validation = _manifest_split_selection(_cfg(seed=99), raw)
    assert validation == same_validation and train != changed


def test_group_selection_is_global_and_deterministic():
    first = build_test_selection({"a": 10, "b": 20, "c": 30}, task="group", test_samples=15, test_seed=42)
    second = build_test_selection({"a": 10, "b": 20, "c": 30}, task="group", test_samples=15, test_seed=42)
    assert first == second
    assert first["selected_count"] == 15
    assert len({(x["leaf_task"], x["local_index"]) for x in first["selected_leaf_docs"]}) == 15

def test_lm_eval_cot_validation_rejects_conflict():
    cfg = {"evaluation": {"lm_eval_revision": "6d2abda4fd171e68a8789330c4149e37c1ca0bda", "limit": None, "lm_eval_task_specs": [{"name": "bbh", "enabled": True, "num_fewshot": 3, "metric": "exact_match", "cot": False, "test_samples": None, "test_seed": 42}]}}
    with pytest.raises(ValueError, match="conflicts"):
        validate_lm_eval_task_specs(cfg)


def test_tulu_shared_calibration_is_independent_of_ann_subset(monkeypatch, tmp_path):
    class Dataset(list):
        column_names = ()
        def select(self, indices):
            return Dataset(self[index] for index in indices)
    raw = {"train": Dataset({"id": i} for i in range(200))}
    monkeypatch.setattr("snn2.data._load_raw", lambda cfg: raw)
    base = _cfg(samples=10, seed=42)
    base.update({"data": {"dataset_name": "fake", "train_split": "train", "validation_size": 5}, "calibration": {"seed": 42, "num_samples": 4, "with_replacement": False}})
    first = prepare_manifests(base, type("L", (), {"data_dir": tmp_path / "one"})())
    changed = {**base, "training": {"train_samples": 20, "train_seed": 99}}
    second = prepare_manifests(changed, type("L", (), {"data_dir": tmp_path / "two"})())
    assert first["validation"]["indices"] == second["validation"]["indices"]
    assert first["calibration"]["indices"] == second["calibration"]["indices"]
    assert first["train"]["indices"] != second["train"]["indices"]
    assert first["calibration"]["retained_in_ann_training_subset"] is None

def test_prune_empty_group_leaves():
    tree = {"group": {"a": object(), "b": object(), "c": object()}}
    pruned = prune_empty_selected_leaves(tree, {"b": {2}})
    assert list(pruned["group"]) == ["b"]


def test_effective_counts_leave_non_leaf_aliases_untouched():
    result = {"n-samples": {"leaf": {"original": 10, "effective": 10}, "group": {"original": 10, "effective": 10}}}
    selection = {"selected_leaf_docs": [{"leaf_task": "leaf", "local_index": 2}]}
    correct_effective_sample_counts(result, selection)
    assert result["n-samples"]["leaf"]["effective"] == 1
    assert result["n-samples"]["group"]["effective"] == 10

def test_execution_counter_delta_is_task_local():
    from evaluate_lm_harness import execution_counter_delta
    assert execution_counter_delta({}, {"model_forward_calls": 10, "temporal_sample_step_forwards": 20}) == {"model_forward_calls": 10, "temporal_sample_step_forwards": 20}
    assert execution_counter_delta({"model_forward_calls": 10, "temporal_sample_step_forwards": 20}, {"model_forward_calls": 17, "temporal_sample_step_forwards": 35}) == {"model_forward_calls": 7, "temporal_sample_step_forwards": 15}
