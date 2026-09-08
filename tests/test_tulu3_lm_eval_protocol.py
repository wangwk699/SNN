import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pytest

from snn2.data import _manifest_split_selection, prepare_manifests
from snn2.lm_eval_protocol import (LM_EVAL_0_4_8_TASK_COT, LM_EVAL_0_4_8_TASK_METRIC, build_test_selection, correct_effective_sample_counts, prune_empty_selected_leaves,
    result_contains_metric, selection_by_leaf, validate_lm_eval_task_specs)


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


@pytest.mark.parametrize("task, metric", [("truthfulqa_mc1", "acc"), ("mmlu_pro", "exact_match"), ("bbh", "exact_match"), ("agieval", "acc"), ("gsm8k_cot", "exact_match"), ("minerva_math", "exact_match"), ("arc_challenge", "acc_norm"), ("piqa", "acc_norm"), ("winogrande", "acc"), ("boolq", "acc")])
def test_pinned_metric_mapping(task, metric):
    assert LM_EVAL_0_4_8_TASK_METRIC[task] == metric

@pytest.mark.parametrize("task", ["arc_challenge", "piqa", "winogrande", "boolq"])
def test_new_fast_tasks_are_non_cot(task):
    assert LM_EVAL_0_4_8_TASK_COT[task] is False

def test_truthfulqa_metric_conflict_is_rejected():
    cfg = {"evaluation": {"lm_eval_revision": "6d2abda4fd171e68a8789330c4149e37c1ca0bda", "limit": None, "lm_eval_task_specs": [{"name": "truthfulqa_mc1", "enabled": True, "num_fewshot": 0, "metric": "acc_mc1", "cot": False, "test_samples": None, "test_seed": 42}]}}
    with pytest.raises(ValueError, match="metric"):
        validate_lm_eval_task_specs(cfg)

def test_result_contains_metric_accepts_filter_suffix_but_not_stderr():
    assert result_contains_metric({"results": {"x": {"exact_match,custom-extract": 1.0}}}, "exact_match")
    assert result_contains_metric({"results": {"x": {"acc,none": 1.0}}}, "acc")
    assert not result_contains_metric({"results": {"x": {"acc_stderr,none": 0.1}}}, "acc")

def test_selection_records_population_and_replays_exactly():
    selection = build_test_selection({"a": 10, "b": 20, "c": 30}, task="group", test_samples=15, test_seed=42)
    assert selection["leaf_population"] == {"a": 10, "b": 20, "c": 30}
    replay = build_test_selection(selection["leaf_population"], task="group", test_samples=15, test_seed=42)
    assert selection["selected_leaf_docs"] == replay["selected_leaf_docs"]
    tampered = [*selection["selected_leaf_docs"]]
    tampered[0] = {"leaf_task": "a", "local_index": 9}
    assert tampered != replay["selected_leaf_docs"]

def test_full_selection_records_every_leaf_document():
    selection = build_test_selection({"a": 2, "b": 3}, task="group", test_samples=None, test_seed=42)
    assert selection["selected_count"] == selection["total_population_size"] == 5
    assert selection["selected_leaf_docs"] == [{"leaf_task": "a", "local_index": 0}, {"leaf_task": "a", "local_index": 1}, {"leaf_task": "b", "local_index": 0}, {"leaf_task": "b", "local_index": 1}, {"leaf_task": "b", "local_index": 2}]


@pytest.mark.parametrize("task", ["arc_challenge", "piqa"])
def test_acc_norm_tasks_reject_acc_metric(task):
    cfg = {"evaluation": {"lm_eval_revision": "6d2abda4fd171e68a8789330c4149e37c1ca0bda", "lm_eval_task_specs": [{"name": task, "enabled": True, "num_fewshot": 0, "metric": "acc", "cot": False, "test_samples": None, "test_seed": 42}]}}
    with pytest.raises(ValueError, match="metric"):
        validate_lm_eval_task_specs(cfg)

@pytest.mark.parametrize("task", ["winogrande", "boolq"])
def test_acc_fast_tasks_accept_acc_metric(task):
    cfg = {"evaluation": {"lm_eval_revision": "6d2abda4fd171e68a8789330c4149e37c1ca0bda", "lm_eval_task_specs": [{"name": task, "enabled": True, "num_fewshot": 0, "metric": "acc", "cot": False, "test_samples": None, "test_seed": 42}]}}
    assert validate_lm_eval_task_specs(cfg)[0]["name"] == task
