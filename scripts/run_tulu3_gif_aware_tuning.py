#!/usr/bin/env python3
"""Adaptive Tulu-3 GIF-aware ANN tuning driver.

This orchestrator never mutates the source YAML.  Every candidate is launched
from a temporary copy, and completed candidates are reused on restart.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import time
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_NAMES = (
    "truthfulqa_mc1",
    "agieval",
    "arc_challenge",
    "piqa",
    "winogrande",
    "boolq",
)
STAGE1_LEARNING_RATES = (
    Decimal("3e-6"),
    Decimal("4e-6"),
    Decimal("5e-6"),
    Decimal("6e-6"),
    Decimal("7e-6"),
)
STAGE2_SCHEDULERS = (
    ("cosine", Decimal("0.01")),
    ("cosine", Decimal("0.03")),
    ("constant_with_warmup", Decimal("0.01")),
    ("constant_with_warmup", Decimal("0.03")),
)
STAGE3_BACKWARD_POLICIES = (
    ("hard_clip", "ste"),
    ("ste", "hard_clip"),
    ("ste", "ste"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the adaptive Tulu-3 GIF-aware ANN tuning plan"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-processes", required=True, type=int)
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def decimal_text(value: Decimal | float | str) -> str:
    decimal = value if isinstance(value, Decimal) else Decimal(str(value))
    if not decimal.is_finite():
        raise ValueError(f"Non-finite decimal: {value}")
    return str(float(decimal))


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_source(cfg: dict[str, Any]) -> None:
    expected = {
        "experiment.task": cfg["experiment"].get("task") == "tulu3",
        "experiment.ann_mode": cfg["experiment"].get("ann_mode") == "gif_aware",
        "phase.T": cfg["phase"].get("T") == 4,
        "mtn.T": cfg["mtn"].get("T") == 4,
        "calibration.group_size": cfg["calibration"].get("group_size") == 128,
        "ann_training.prefix_enabled": cfg["ann_training"].get("prefix_enabled") is True,
        "replacement.common_clip_enabled": cfg["replacement"].get("common_clip_enabled") is True,
    }
    invalid = [name for name, valid in expected.items() if not valid]
    if invalid:
        raise ValueError(f"Source config violates fixed tuning constraints: {invalid}")
    if cfg["training"].get("train_samples") not in {1024, 10000}:
        raise ValueError("Expected the Tulu GIF source config to use 1024 or 10000 samples")


def fixed_base_config(source: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(source)
    cfg["training"].update(
        {
            "train_samples": 10000,
            "train_seed": 42,
            "num_train_epochs": 1,
            "gradient_accumulation_steps": 16,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "eval_strategy": "epoch",
            "save_strategy": "no",
            "load_best_model_at_end": False,
            "resume_from_checkpoint": None,
            "replacement_diagnostics_max_calls_per_site": 1,
        }
    )
    cfg["gif"]["quantizer_clip_backward"] = "hard_clip"
    cfg["replacement"]["outer_clip_backward"] = "hard_clip"
    cfg["phase"]["T"] = 4
    cfg["mtn"]["T"] = 4
    cfg["calibration"]["group_size"] = 128
    cfg["ann_training"]["prefix_enabled"] = True
    cfg["replacement"]["common_clip_enabled"] = True
    return cfg


def model_root(cfg: dict[str, Any]) -> Path:
    output_root = Path(cfg["experiment"]["output_root"])
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    return (
        output_root
        / str(cfg["experiment"]["id"])
        / "tulu3"
        / safe_name(str(cfg["experiment"]["model_name"]))
        / "gif_aware"
    )


def ann_dir_for(cfg: dict[str, Any], *, include_tuning_id: bool = True) -> Path:
    training = cfg["training"]
    identity = (
        f"epochs_{training['num_train_epochs']}_"
        f"num_samples_{int(cfg['calibration']['num_samples'])}_"
        f"lr{training['learning_rate']}_"
        f"train_samples_{int(training['train_samples'])}_"
        f"calibration_group_size_{int(cfg['calibration']['group_size'])}"
    )
    run_variant = "prefix_enabled_ture_common_clip_enabled_true"
    optimizer = (
        f"phase_T_{int(cfg['phase']['T'])}_mtn_T_{int(cfg['mtn']['T'])}_"
        f"lr_scheduler_type_{training['lr_scheduler_type']}_"
        f"warmup_ratio_{float(training['warmup_ratio'])}_"
        f"gradient_accumulation_steps_{int(training['gradient_accumulation_steps'])}"
    )
    root = model_root(cfg) / identity / run_variant / optimizer
    tuning_id = training.get("tuning_run_id")
    if include_tuning_id and tuning_id is not None:
        root = root / f"tuning_{tuning_id}"
    return root / f"seed{int(cfg['experiment']['seed'])}" / "ann"


def completed_summary(ann_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    if not (ann_dir / "training_result.json").is_file():
        return None
    candidates = sorted(
        (ann_dir / "evaluation").rglob("evaluation_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            metrics = summary["task_metrics"]
            if set(metrics) == set(TASK_NAMES) and all(
                math.isfinite(float(metrics[name])) for name in TASK_NAMES
            ):
                return path, summary
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return None


def record_from_summary(
    *,
    stage: str,
    label: str,
    cfg: dict[str, Any],
    ann_dir: Path,
    summary_path: Path,
    summary: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    metrics = {name: float(summary["task_metrics"][name]) for name in TASK_NAMES}
    training_result_path = ann_dir / "training_result.json"
    training_result = (
        json.loads(training_result_path.read_text(encoding="utf-8"))
        if training_result_path.is_file()
        else {}
    )
    history_path = ann_dir / "trainer_log_history.json"
    history = (
        json.loads(history_path.read_text(encoding="utf-8"))
        if history_path.is_file()
        else []
    )
    return {
        "stage": stage,
        "label": label,
        "status": status,
        "mean": sum(metrics.values()) / len(TASK_NAMES),
        "task_metrics": metrics,
        "training": {
            "learning_rate": float(cfg["training"]["learning_rate"]),
            "lr_scheduler_type": cfg["training"]["lr_scheduler_type"],
            "warmup_ratio": float(cfg["training"]["warmup_ratio"]),
            "num_train_epochs": float(cfg["training"]["num_train_epochs"]),
            "gradient_accumulation_steps": int(
                cfg["training"]["gradient_accumulation_steps"]
            ),
            "max_grad_norm": float(cfg["training"]["max_grad_norm"]),
            "effective_global_batch_size": training_result.get(
                "effective_global_batch_size"
            ),
            "train_loss": training_result.get("train_loss"),
        },
        "backward": {
            "gif_quantizer_clip_backward": cfg["gif"][
                "quantizer_clip_backward"
            ],
            "outer_clip_backward": cfg["replacement"]["outer_clip_backward"],
        },
        "ann_dir": str(ann_dir),
        "evaluation_summary": str(summary_path),
        "training_curve": history,
        "training_diagnostics": str(
            ann_dir / "training_replacement_diagnostics.json"
        ),
        "evaluation_diagnostics": str(
            summary_path.parent / "replacement_diagnostics_rank0.json"
        ),
    }


class TuningRun:
    def __init__(
        self,
        source_path: Path,
        source: dict[str, Any],
        num_processes: int,
    ):
        self.source_path = source_path
        self.source = source
        self.num_processes = num_processes
        self.sweep_root = model_root(source) / "_sweeps" / "tulu3_gif_aware_tuning_v1"
        self.manifest_path = self.sweep_root / "manifest.json"
        self.status_path = self.sweep_root / "run_status.json"
        self.leaderboard_json = self.sweep_root / "leaderboard.json"
        self.leaderboard_csv = self.sweep_root / "leaderboard.csv"
        self.records: list[dict[str, Any]] = []
        self.statuses: dict[str, Any] = {}
        if self.status_path.is_file():
            try:
                self.statuses = json.loads(self.status_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.statuses = {}

    def write_manifest(self, *, winners: dict[str, Any] | None = None) -> None:
        atomic_json(
            self.manifest_path,
            {
                "plan_version": "tulu3_gif_aware_tuning_v1",
                "source_config": str(self.source_path),
                "source_config_sha256": source_sha256(self.source_path),
                "selection_metric": "unweighted_mean_of_six_task_metrics",
                "task_names": list(TASK_NAMES),
                "num_processes": self.num_processes,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "fixed_constraints": {
                    "training.train_samples": 10000,
                    "training.gradient_accumulation_steps": 16,
                    "phase.T": 4,
                    "mtn.T": 4,
                    "calibration.group_size": 128,
                    "ann_training.prefix_enabled": True,
                    "replacement.common_clip_enabled": True,
                },
                "stages": {
                    "stage1_learning_rate": [float(x) for x in STAGE1_LEARNING_RATES],
                    "stage2_scheduler_warmup": [
                        {"lr_scheduler_type": scheduler, "warmup_ratio": float(warmup)}
                        for scheduler, warmup in STAGE2_SCHEDULERS
                    ],
                    "stage3_backward": [
                        {
                            "gif.quantizer_clip_backward": quantizer,
                            "replacement.outer_clip_backward": outer,
                        }
                        for quantizer, outer in (
                            ("hard_clip", "hard_clip"),
                            *STAGE3_BACKWARD_POLICIES,
                        )
                    ],
                    "stage4_duration_and_grad_clip": [
                        "epochs_2_lr_1.0x_max_grad_norm_1",
                        "epochs_2_lr_0.6x_max_grad_norm_1",
                        "epochs_1_lr_1.0x_max_grad_norm_2",
                    ],
                },
                "winners": winners or {},
            },
        )

    def persist(self) -> None:
        atomic_json(self.status_path, self.statuses)
        ordered = sorted(
            self.records,
            key=lambda item: (
                item.get("mean") is None,
                -(item.get("mean") or -math.inf),
                item["stage"],
                item["label"],
            ),
        )
        atomic_json(
            self.leaderboard_json,
            {
                "baselines": {"vanilla": 0.5851, "unaware": 0.5907},
                "ranking": ordered,
            },
        )
        self.leaderboard_csv.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.leaderboard_csv.with_name(
            f".{self.leaderboard_csv.name}.tmp"
        )
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "stage",
                    "label",
                    "status",
                    "mean",
                    *TASK_NAMES,
                    "learning_rate",
                    "lr_scheduler_type",
                    "warmup_ratio",
                    "num_train_epochs",
                    "max_grad_norm",
                    "gif_quantizer_clip_backward",
                    "outer_clip_backward",
                    "ann_dir",
                ],
            )
            writer.writeheader()
            for item in ordered:
                training = item.get("training", {})
                backward = item.get("backward", {})
                row = {
                    "stage": item["stage"],
                    "label": item["label"],
                    "status": item["status"],
                    "mean": item.get("mean"),
                    **{
                        name: item.get("task_metrics", {}).get(name)
                        for name in TASK_NAMES
                    },
                    **{
                        key: training.get(key)
                        for key in (
                            "learning_rate",
                            "lr_scheduler_type",
                            "warmup_ratio",
                            "num_train_epochs",
                            "max_grad_norm",
                        )
                    },
                    **backward,
                    "ann_dir": item.get("ann_dir"),
                }
                writer.writerow(row)
        os.replace(temporary, self.leaderboard_csv)

    def _temporary_config(self, cfg: dict[str, Any]) -> Path:
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".yaml",
            prefix="snn-tulu3-gif-tuning-",
            dir="/tmp",
            delete=False,
        )
        try:
            yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
        finally:
            handle.close()
        return Path(handle.name)

    def run_candidate(
        self,
        *,
        stage: str,
        label: str,
        cfg: dict[str, Any],
        legacy_ann_dir: Path | None = None,
    ) -> dict[str, Any]:
        ann_dir = ann_dir_for(cfg)
        existing = completed_summary(ann_dir)
        if existing is None and legacy_ann_dir is not None:
            legacy = completed_summary(legacy_ann_dir)
            if legacy is not None:
                path, summary = legacy
                record = record_from_summary(
                    stage=stage,
                    label=label,
                    cfg=cfg,
                    ann_dir=legacy_ann_dir,
                    summary_path=path,
                    summary=summary,
                    status="reused_legacy",
                )
                self.records.append(record)
                self.statuses[label] = {
                    "status": "reused_legacy",
                    "ann_dir": str(legacy_ann_dir),
                    "updated_at_unix": time.time(),
                }
                self.persist()
                print(f"[REUSE] {stage} {label}: mean={record['mean']:.6f}", flush=True)
                return record
        if existing is not None:
            path, summary = existing
            record = record_from_summary(
                stage=stage,
                label=label,
                cfg=cfg,
                ann_dir=ann_dir,
                summary_path=path,
                summary=summary,
                status="reused",
            )
            self.records.append(record)
            self.statuses[label] = {
                "status": "reused",
                "ann_dir": str(ann_dir),
                "updated_at_unix": time.time(),
            }
            self.persist()
            print(f"[REUSE] {stage} {label}: mean={record['mean']:.6f}", flush=True)
            return record

        run_config = self._temporary_config(cfg)
        try:
            training_is_complete = (
                (ann_dir / "training_result.json").is_file()
                and (ann_dir / "final").is_dir()
            )
            if training_is_complete:
                print(
                    f"[RESUME-EVAL] {stage} {label}: existing training is complete",
                    flush=True,
                )
            else:
                self.statuses[label] = {
                    "status": "training",
                    "ann_dir": str(ann_dir),
                    "updated_at_unix": time.time(),
                }
                self.persist()
                print(f"[TRAIN] {stage} {label}", flush=True)
                train = subprocess.run(
                    [
                        "torchrun",
                        "--standalone",
                        f"--nproc_per_node={self.num_processes}",
                        "scripts/train_ann.py",
                        "--config",
                        str(run_config),
                    ],
                    cwd=PROJECT_ROOT,
                    check=False,
                )
                if train.returncode != 0:
                    raise RuntimeError(
                        f"training exited with status {train.returncode}"
                    )

            self.statuses[label]["status"] = "evaluating"
            self.statuses[label]["updated_at_unix"] = time.time()
            self.persist()
            print(f"[EVAL] {stage} {label}", flush=True)
            evaluation = subprocess.run(
                [
                    "accelerate",
                    "launch",
                    "--num_processes",
                    str(self.num_processes),
                    "scripts/evaluate_lm_harness.py",
                    "--config",
                    str(run_config),
                    "--neuron",
                    "ann",
                ],
                cwd=PROJECT_ROOT,
                check=False,
            )
            if evaluation.returncode != 0:
                raise RuntimeError(
                    f"lm-eval exited with status {evaluation.returncode}"
                )
            completed = completed_summary(ann_dir)
            if completed is None:
                raise RuntimeError(
                    "lm-eval returned success but the six-task summary is incomplete"
                )
            path, summary = completed
            record = record_from_summary(
                stage=stage,
                label=label,
                cfg=cfg,
                ann_dir=ann_dir,
                summary_path=path,
                summary=summary,
                status="completed",
            )
            self.records.append(record)
            self.statuses[label] = {
                "status": "completed",
                "ann_dir": str(ann_dir),
                "mean": record["mean"],
                "updated_at_unix": time.time(),
            }
            self.persist()
            print(f"[DONE] {stage} {label}: mean={record['mean']:.6f}", flush=True)
            return record
        except Exception as exc:
            record = {
                "stage": stage,
                "label": label,
                "status": "failed",
                "mean": None,
                "error": str(exc),
                "ann_dir": str(ann_dir),
            }
            self.records.append(record)
            self.statuses[label] = {
                "status": "failed",
                "ann_dir": str(ann_dir),
                "error": str(exc),
                "updated_at_unix": time.time(),
            }
            self.persist()
            print(f"[FAILED] {stage} {label}: {exc}; continuing.", flush=True)
            return record
        finally:
            run_config.unlink(missing_ok=True)


def best_completed(records: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    completed = [record for record in records if record.get("mean") is not None]
    if not completed:
        raise RuntimeError(f"No successful candidates remain after {stage}")
    return max(completed, key=lambda record: float(record["mean"]))


def candidate_config(
    base: dict[str, Any],
    *,
    tuning_id: str,
    learning_rate: Decimal,
    scheduler: str,
    warmup: Decimal,
    quantizer_backward: str = "hard_clip",
    outer_backward: str = "hard_clip",
    epochs: int = 1,
    max_grad_norm: float = 1.0,
) -> dict[str, Any]:
    cfg = deepcopy(base)
    cfg["training"].update(
        {
            "tuning_run_id": tuning_id,
            "learning_rate": float(learning_rate),
            "lr_scheduler_type": scheduler,
            "warmup_ratio": float(warmup),
            "num_train_epochs": epochs,
            "max_grad_norm": max_grad_norm,
        }
    )
    cfg["gif"]["quantizer_clip_backward"] = quantizer_backward
    cfg["replacement"]["outer_clip_backward"] = outer_backward
    return cfg


def optimizer_from_record(record: dict[str, Any]) -> tuple[Decimal, str, Decimal]:
    training = record["training"]
    return (
        Decimal(str(training["learning_rate"])),
        str(training["lr_scheduler_type"]),
        Decimal(str(training["warmup_ratio"])),
    )


def main() -> int:
    args = parse_args()
    if args.num_processes <= 0:
        raise ValueError("--num-processes must be positive")
    source_path = Path(args.config)
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path
    source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    validate_source(source)
    base = fixed_base_config(source)
    run = TuningRun(source_path, base, args.num_processes)
    winners: dict[str, Any] = {}
    run.write_manifest(winners=winners)

    stage1: list[dict[str, Any]] = []
    for learning_rate in STAGE1_LEARNING_RATES:
        lr_text = decimal_text(learning_rate)
        label = f"s1_lr{lr_text}"
        cfg = candidate_config(
            base,
            tuning_id=label,
            learning_rate=learning_rate,
            scheduler="cosine",
            warmup=Decimal("0"),
        )
        legacy_ann_dir = (
            ann_dir_for(cfg, include_tuning_id=False)
            if learning_rate == Decimal("5e-6")
            else None
        )
        stage1.append(
            run.run_candidate(
                stage="stage1_learning_rate",
                label=label,
                cfg=cfg,
                legacy_ann_dir=legacy_ann_dir,
            )
        )
    winner1 = best_completed(stage1, "stage1_learning_rate")
    winners["stage1"] = {
        "label": winner1["label"],
        "mean": winner1["mean"],
        "training": winner1["training"],
    }
    run.write_manifest(winners=winners)
    best_lr, _, _ = optimizer_from_record(winner1)

    stage2 = [winner1]
    for scheduler, warmup in STAGE2_SCHEDULERS:
        label = f"s2_lr{decimal_text(best_lr)}_{scheduler}_wr{decimal_text(warmup)}"
        cfg = candidate_config(
            base,
            tuning_id=label,
            learning_rate=best_lr,
            scheduler=scheduler,
            warmup=warmup,
        )
        stage2.append(
            run.run_candidate(
                stage="stage2_scheduler_warmup", label=label, cfg=cfg
            )
        )
    winner2 = best_completed(stage2, "stage2_scheduler_warmup")
    winners["stage2"] = {
        "label": winner2["label"],
        "mean": winner2["mean"],
        "training": winner2["training"],
    }
    run.write_manifest(winners=winners)
    best_lr, best_scheduler, best_warmup = optimizer_from_record(winner2)

    stage3 = [winner2]
    for quantizer_backward, outer_backward in STAGE3_BACKWARD_POLICIES:
        label = (
            f"s3_lr{decimal_text(best_lr)}_{best_scheduler}_"
            f"wr{decimal_text(best_warmup)}_q{quantizer_backward}_c{outer_backward}"
        )
        cfg = candidate_config(
            base,
            tuning_id=label,
            learning_rate=best_lr,
            scheduler=best_scheduler,
            warmup=best_warmup,
            quantizer_backward=quantizer_backward,
            outer_backward=outer_backward,
        )
        stage3.append(
            run.run_candidate(stage="stage3_backward", label=label, cfg=cfg)
        )
    winner3 = best_completed(stage3, "stage3_backward")
    winners["stage3"] = {
        "label": winner3["label"],
        "mean": winner3["mean"],
        "training": winner3["training"],
        "backward": winner3["backward"],
    }
    run.write_manifest(winners=winners)
    quantizer_backward = winner3["backward"]["gif_quantizer_clip_backward"]
    outer_backward = winner3["backward"]["outer_clip_backward"]

    stage4_specs = (
        ("ep2_lr1x_gn1", best_lr, 2, 1.0),
        ("ep2_lr0.6x_gn1", best_lr * Decimal("0.6"), 2, 1.0),
        ("ep1_lr1x_gn2", best_lr, 1, 2.0),
    )
    stage4: list[dict[str, Any]] = [winner3]
    for suffix, learning_rate, epochs, max_grad_norm in stage4_specs:
        label = f"s4_{suffix}_{quantizer_backward}_{outer_backward}"
        cfg = candidate_config(
            base,
            tuning_id=label,
            learning_rate=learning_rate,
            scheduler=best_scheduler,
            warmup=best_warmup,
            quantizer_backward=quantizer_backward,
            outer_backward=outer_backward,
            epochs=epochs,
            max_grad_norm=max_grad_norm,
        )
        stage4.append(
            run.run_candidate(
                stage="stage4_duration_and_grad_clip", label=label, cfg=cfg
            )
        )
    winner4 = best_completed(stage4, "stage4_duration_and_grad_clip")
    winners["final"] = {
        "label": winner4["label"],
        "mean": winner4["mean"],
        "training": winner4["training"],
        "backward": winner4["backward"],
        "ann_dir": winner4["ann_dir"],
    }
    run.write_manifest(winners=winners)
    run.persist()

    failed = [record for record in run.records if record["status"] == "failed"]
    print(
        f"Tuning finished: {len(run.records) - len(failed)} usable records, "
        f"{len(failed)} failed. Final best={winner4['label']} "
        f"mean={winner4['mean']:.6f}",
        flush=True,
    )
    print(f"Leaderboard: {run.leaderboard_json}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
