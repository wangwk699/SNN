from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_lm_harness as evaluator


def _spec(name: str = "task_a") -> dict:
    return {
        "name": name,
        "num_fewshot": 0,
        "metric": "acc",
        "test_seed": 42,
    }


def test_tulu_simple_evaluate_disables_sample_logging():
    calls = []

    def fake_simple_evaluate(**kwargs):
        calls.append(kwargs)
        return {"results": {"task_a": {"acc": 0.5}}}

    result = evaluator._run_simple_evaluate(
        fake_simple_evaluate,
        harness_model=object(),
        spec=_spec(),
        task_manager=object(),
        batch_size=2,
        experiment_seed=7,
        apply_chat_template=True,
        is_tulu_lm_eval=True,
    )

    assert result["results"]["task_a"]["acc"] == 0.5
    assert calls[0]["log_samples"] is False
    assert calls[0]["random_seed"] == calls[0]["numpy_random_seed"] == 7
    assert calls[0]["torch_random_seed"] == calls[0]["fewshot_random_seed"] == 7


def test_non_tulu_simple_evaluate_keeps_sample_logging_enabled():
    calls = []

    def fake_simple_evaluate(**kwargs):
        calls.append(kwargs)
        return {}

    evaluator._run_simple_evaluate(
        fake_simple_evaluate,
        harness_model=object(),
        spec=_spec(),
        task_manager=object(),
        batch_size=1,
        experiment_seed=42,
        apply_chat_template=False,
        is_tulu_lm_eval=False,
    )

    assert calls[0]["log_samples"] is True


def test_atomic_summary_replaces_complete_json_and_removes_temporary_file(tmp_path):
    path = tmp_path / "evaluation_summary.json"
    first = {"task_times": {"task_a": "00:01:00"}, "task_metrics": {"task_a": 0.5}}
    second = {"task_times": {"task_a": "00:01:00", "task_b": "00:02:00"}, "task_metrics": {"task_a": 0.5, "task_b": 0.6}}

    evaluator._write_json_atomic(path, first)
    evaluator._write_json_atomic(path, second)

    assert json.loads(path.read_text(encoding="utf-8")) == second
    assert not (tmp_path / ".evaluation_summary.json.tmp").exists()


def test_tulu_summary_is_incrementally_persisted_with_existing_schema(tmp_path):
    task_times = {"task_a": "00:01:00"}
    task_metrics = {"task_a": 0.5}
    first = evaluator._write_tulu_summary(
        tmp_path, task_times=task_times, task_metrics=task_metrics
    )
    assert set(first) == {"task_times", "task_metrics"}
    assert json.loads((tmp_path / "evaluation_summary.json").read_text()) == first

    task_times["task_b"] = "00:02:00"
    task_metrics["task_b"] = 0.6
    second = evaluator._write_tulu_summary(
        tmp_path, task_times=task_times, task_metrics=task_metrics
    )
    assert set(second) == {"task_times", "task_metrics"}
    assert json.loads((tmp_path / "evaluation_summary.json").read_text()) == second


def test_task_files_are_persisted_independently_without_samples(tmp_path):
    first_dir = tmp_path / "task_results" / "task_a" / "spec"
    first_result = {"tasks": {"task_a": {"results": {"task_a": {"acc": 0.5}}}}}
    first_selection = {"selected_count": 1, "sampling": {"seed": 42}}
    evaluator._write_task_result_files(
        first_dir, result=first_result, test_selection=first_selection
    )
    assert json.loads((first_dir / "results.json").read_text()) == first_result
    assert json.loads((first_dir / "test_selection.json").read_text()) == first_selection
    assert "samples" not in first_result["tasks"]["task_a"]

    second_dir = tmp_path / "task_results" / "task_b" / "spec"
    second_result = {"tasks": {"task_b": {"results": {"task_b": {"acc": 0.6}}}}}
    evaluator._write_task_result_files(
        second_dir,
        result=second_result,
        test_selection={"selected_count": 2, "sampling": {"seed": 42}},
    )
    assert (first_dir / "results.json").exists()
    assert json.loads((second_dir / "results.json").read_text()) == second_result
