"""Project-side protocol for the pinned lm-eval 0.4.8 tasks."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

LM_EVAL_PINNED_REVISION = "6d2abda4fd171e68a8789330c4149e37c1ca0bda"
LM_EVAL_0_4_8_TASK_COT = {
    "truthfulqa_mc1": False, "mmlu_pro": True, "bbh": True,
    "agieval": False, "gsm8k_cot": True, "minerva_math": False,
}

def _positive_int_or_none(value: Any, field: str) -> None:
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
        raise ValueError(f"{field} must be a positive integer or null")

def validate_lm_eval_task_specs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate Tulu's lm-eval contract before loading an evaluation model."""
    evaluation = cfg["evaluation"]
    if evaluation.get("lm_eval_revision") != LM_EVAL_PINNED_REVISION:
        raise ValueError("evaluation.lm_eval_revision must equal the pinned audited lm-eval revision")
    if evaluation.get("limit") is not None:
        raise ValueError("Tulu-3 evaluation.limit is deprecated; use per-task test_samples")
    specs = evaluation.get("lm_eval_task_specs")
    if not isinstance(specs, list) or not specs:
        raise ValueError("evaluation.lm_eval_task_specs must contain at least one task")
    names: set[str] = set()
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("Each lm_eval_task_specs entry must be a mapping")
        name = spec.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("lm-eval task name must be a non-empty string")
        if name in names:
            raise ValueError(f"Duplicate lm-eval task name: {name}")
        names.add(name)
        if name not in LM_EVAL_0_4_8_TASK_COT:
            raise ValueError(f"lm-eval task {name!r} has no audited CoT semantic")
        for field in ("enabled", "cot"):
            if not isinstance(spec.get(field), bool):
                raise ValueError(f"lm-eval spec {name}.{field} must be a bool")
        fewshot = spec.get("num_fewshot")
        if not isinstance(fewshot, int) or isinstance(fewshot, bool) or fewshot < 0:
            raise ValueError(f"lm-eval spec {name}.num_fewshot must be a non-negative integer")
        if not isinstance(spec.get("metric"), str) or not spec["metric"]:
            raise ValueError(f"lm-eval spec {name}.metric must be a non-empty string")
        _positive_int_or_none(spec.get("test_samples"), f"lm-eval spec {name}.test_samples")
        if not isinstance(spec.get("test_seed"), int) or isinstance(spec.get("test_seed"), bool):
            raise ValueError(f"lm-eval spec {name}.test_seed must be an integer")
        expected = LM_EVAL_0_4_8_TASK_COT[name]
        if spec["cot"] != expected:
            raise ValueError(f"Configured cot={str(spec['cot']).lower()} conflicts with pinned lm-eval task {name!r}, whose audited CoT semantic is {str(expected).lower()} at revision {LM_EVAL_PINNED_REVISION}")
    return specs

def enabled_lm_eval_task_specs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    enabled = [spec for spec in validate_lm_eval_task_specs(cfg) if spec["enabled"]]
    if not enabled:
        raise ValueError("no enabled lm-eval tasks")
    return enabled

def build_test_selection(leaf_population: dict[str, int], *, task: str, test_samples: int | None, test_seed: int) -> dict[str, Any]:
    """Globally sample logical docs, rather than N documents per group leaf."""
    population = [(leaf, index) for leaf in sorted(leaf_population) for index in range(leaf_population[leaf])]
    total = len(population)
    if test_samples is None:
        selected, sampling = population, "full_evaluation_population"
    else:
        if test_samples > total:
            raise ValueError(f"Requested {test_samples} test samples for {task}, but population is only {total}")
        selected = random.Random(test_seed).sample(population, k=test_samples)
        selected.sort()
        sampling = "seeded_random_without_replacement"
    return {"task": task, "test_samples": test_samples, "test_seed": test_seed, "sampling": sampling, "total_population_size": total, "selected_count": len(selected), "selected_leaf_docs": [{"leaf_task": leaf, "local_index": index} for leaf, index in selected]}

def selection_by_leaf(selection: dict[str, Any]) -> dict[str, set[int]]:
    result: dict[str, set[int]] = defaultdict(set)
    for item in selection["selected_leaf_docs"]:
        result[item["leaf_task"]].add(int(item["local_index"]))
    return dict(result)


def prune_empty_selected_leaves(task_tree: dict[str, Any], selected_by_leaf: dict[str, set[int]]) -> dict[str, Any]:
    """Return a task-tree excluding leaves with no selected evaluation documents."""
    result: dict[str, Any] = {}
    for name, value in task_tree.items():
        if isinstance(value, dict):
            nested = prune_empty_selected_leaves(value, selected_by_leaf)
            if nested:
                result[name] = nested
        elif selected_by_leaf.get(name):
            result[name] = value
    return result


def correct_effective_sample_counts(task_result: dict[str, Any], selection: dict[str, Any]) -> None:
    """Keep lm-eval's original population count but correct its wrapper-visible effective count."""
    counts = task_result.get("n-samples")
    if not isinstance(counts, dict):
        return
    selected = selection_by_leaf(selection)
    for leaf_name, count in counts.items():
        if leaf_name not in selected:
            continue
        effective = len(selected[leaf_name])
        if isinstance(count, dict):
            count["effective"] = effective
        else:
            counts[leaf_name] = {"original": count, "effective": effective}
