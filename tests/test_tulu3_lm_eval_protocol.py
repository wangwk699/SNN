import random
import pytest

from snn2.data import _manifest_split_selection
from snn2.lm_eval_protocol import build_test_selection, validate_lm_eval_task_specs


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
