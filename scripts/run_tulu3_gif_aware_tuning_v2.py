#!/usr/bin/env python3
"""Tulu-3 GIF-aware calibration/ratio grid for formal ANN fine-tuning.

The source YAML is read-only.  Every candidate uses a temporary YAML and owns
its Stage A/B calibration artifacts and ANN run directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
PROJECT_ROOT = Path(__file__).resolve().parents[1]
from snn2.artifacts import ArtifactLayout, safe_name
from snn2.data import validate_prefix_discovery_state
from snn2.state_validation import validate_clip_profile, validate_site_state_bundle
from snn2.training import validate_recorded_training_artifact_provenance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_NAMES = (
    "truthfulqa_mc1",
    "agieval",
    "arc_challenge",
    "piqa",
    "winogrande",
    "boolq",
)
CALIBRATION_NUM_SAMPLES = (128, 512, 1024)
GIF_RATIO_PAIRS = (
    (Decimal("0.9"), Decimal("0.1")),
    (Decimal("0.8"), Decimal("0.2")),
    (Decimal("0.7"), Decimal("0.3")),
    (Decimal("0.5"), Decimal("0.5")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the formal Tulu-3 GIF-aware calibration/ratio grid"
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decimal_text(value: Decimal | float | str) -> str:
    decimal = value if isinstance(value, Decimal) else Decimal(str(value))
    if not decimal.is_finite():
        raise ValueError(f"Non-finite decimal: {value}")
    return str(float(decimal))


def validate_source(cfg: dict[str, Any]) -> None:
    checks = {
        "experiment.task": cfg["experiment"].get("task") == "tulu3",
        "experiment.ann_mode": cfg["experiment"].get("ann_mode") == "gif_aware",
        "phase.T": cfg["phase"].get("T") == 4,
        "mtn.T": cfg["mtn"].get("T") == 4,
        "calibration.group_size": cfg["calibration"].get("group_size") == 128,
        "ann_training.prefix_enabled": (
            cfg["ann_training"].get("prefix_enabled") is True
        ),
        "replacement.common_clip_enabled": (
            cfg["replacement"].get("common_clip_enabled") is True
        ),
    }
    invalid = [key for key, valid in checks.items() if not valid]
    if invalid:
        raise ValueError(f"Source config violates fixed constraints: {invalid}")


def fixed_base_config(source: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(source)
    cfg["training"].update(
        {
            "train_samples": 10000,
            "train_seed": 42,
            "num_train_epochs": 1,
            "gradient_accumulation_steps": 16,
            "learning_rate": 5e-6,
            "weight_decay": 0.0,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.01,
            "max_grad_norm": 1.0,
            "eval_strategy": "epoch",
            "save_strategy": "no",
            "load_best_model_at_end": False,
            "resume_from_checkpoint": None,
            "replacement_diagnostics_max_calls_per_site": 1,
            "tuning_run_id": "v2_calibration_ratio_grid",
        }
    )
    cfg["ann_training_memory"].update(
        {
            "attention_core_checkpoint": True,
            "mlp_checkpoint": True,
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


def candidate_config(
    base: dict[str, Any],
    *,
    num_samples: int,
    low_ratio: Decimal,
    salient_ratio: Decimal,
) -> dict[str, Any]:
    if low_ratio + salient_ratio != Decimal("1.0"):
        raise ValueError("GIF low/salient ratios must sum to 1")
    cfg = deepcopy(base)
    cfg["calibration"]["num_samples"] = int(num_samples)
    cfg["gif"]["low_ratio"] = float(low_ratio)
    cfg["gif"]["salient_ratio"] = float(salient_ratio)
    return cfg


def candidate_label(
    num_samples: int, low_ratio: Decimal, salient_ratio: Decimal
) -> str:
    return (
        f"num_samples_{num_samples}_low_{decimal_text(low_ratio)}_"
        f"salient_{decimal_text(salient_ratio)}"
    )


def sweep_root(cfg: dict[str, Any]) -> Path:
    output_root = Path(cfg["experiment"]["output_root"])
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    return (
        output_root
        / str(cfg["experiment"]["id"])
        / "tulu3"
        / safe_name(str(cfg["experiment"]["model_name"]))
        / "gif_aware"
        / "_sweeps"
        / "tulu3_gif_aware_tuning_v2"
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
            summary = load_json(path)
            metrics = summary["task_metrics"]
            if set(metrics) == set(TASK_NAMES) and all(
                math.isfinite(float(metrics[name])) for name in TASK_NAMES
            ):
                return path, summary
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return None


def validate_training_result(
    cfg: dict[str, Any], layout: ArtifactLayout
) -> dict[str, Any]:
    path = layout.ann_dir / "training_result.json"
    if not path.is_file() or not layout.ann_checkpoint_dir.is_dir():
        raise FileNotFoundError("Candidate training is incomplete")
    result = load_json(path)
    expected = {
        "world_size": 3,
        "effective_global_batch_size": 48,
        "attention_core_checkpoint": True,
        "mlp_checkpoint": True,
        "ann_training_calibration_num_samples": int(
            cfg["calibration"]["num_samples"]
        ),
        "ann_training_gif_low_ratio": float(cfg["gif"]["low_ratio"]),
        "ann_training_gif_salient_ratio": float(cfg["gif"]["salient_ratio"]),
    }
    mismatched = {
        key: {"expected": value, "actual": result.get(key)}
        for key, value in expected.items()
        if result.get(key) != value
    }
    history = result.get("eval_loss_history")
    validate_recorded_training_artifact_provenance(cfg, layout)
    if (
        not isinstance(history, list)
        or not history
        or any(not math.isfinite(float(value)) for value in history)
        or not math.isfinite(float(result.get("eval_loss", math.nan)))
    ):
        mismatched["eval_loss_history"] = {
            "expected": "one or more finite values",
            "actual": history,
        }
    if mismatched:
        raise ValueError(f"Completed training provenance mismatch: {mismatched}")
    return result


def calibration_is_valid(cfg: dict[str, Any], layout: ArtifactLayout) -> bool:
    try:
        validation = validate_site_state_bundle(
            layout.ann_training_site_dir, clip_policy="forbid_all"
        )
        manifest = validation["manifest"]
        expected = {
            "purpose": "ann_training_calibration",
            "calibration_group_size": 128,
            "calibration_num_samples": int(cfg["calibration"]["num_samples"]),
            "gif_low_ratio": float(cfg["gif"]["low_ratio"]),
            "gif_salient_ratio": float(cfg["gif"]["salient_ratio"]),
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            return False
        validate_clip_profile(
            layout.ann_training_site_dir,
            layout.ann_training_clip_profile_dir,
            phase_T=4,
            mtn_T=4,
            group_size=128,
            num_samples=int(cfg["calibration"]["num_samples"]),
            low_ratio=float(cfg["gif"]["low_ratio"]),
            salient_ratio=float(cfg["gif"]["salient_ratio"]),
        )
        return True
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return False


def record_from_summary(
    *,
    label: str,
    cfg: dict[str, Any],
    layout: ArtifactLayout,
    summary_path: Path,
    summary: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    metrics = {name: float(summary["task_metrics"][name]) for name in TASK_NAMES}
    training_result = validate_training_result(cfg, layout)
    return {
        "label": label,
        "status": status,
        "mean": sum(metrics.values()) / len(TASK_NAMES),
        "task_metrics": metrics,
        "calibration_num_samples": int(cfg["calibration"]["num_samples"]),
        "gif_low_ratio": float(cfg["gif"]["low_ratio"]),
        "gif_salient_ratio": float(cfg["gif"]["salient_ratio"]),
        "eval_loss": float(training_result["eval_loss"]),
        "eval_loss_history": [
            float(value) for value in training_result["eval_loss_history"]
        ],
        "train_loss": training_result.get("train_loss"),
        "effective_global_batch_size": training_result[
            "effective_global_batch_size"
        ],
        "ann_dir": str(layout.ann_dir),
        "calibration_stage_a": str(layout.ann_training_site_dir),
        "calibration_stage_b": str(layout.ann_training_clip_profile_dir),
        "evaluation_summary": str(summary_path),
        "tokenization_metadata": str(
            layout.ann_dir / "tokenization_metadata.json"
        ),
        "training_curve": str(layout.ann_dir / "trainer_log_history.json"),
        "training_diagnostics": str(
            layout.ann_dir / "training_replacement_diagnostics.json"
        ),
        "evaluation_diagnostics": str(
            summary_path.parent / "replacement_diagnostics_rank0.json"
        ),
    }


class GridRun:
    def __init__(
        self,
        source_path: Path,
        source_hash: str,
        base: dict[str, Any],
        num_processes: int,
    ):
        self.source_path = source_path
        self.source_hash = source_hash
        self.base = base
        self.num_processes = num_processes
        self.root = sweep_root(base)
        self.manifest_path = self.root / "manifest.json"
        self.status_path = self.root / "run_status.json"
        self.leaderboard_json = self.root / "leaderboard.json"
        self.leaderboard_csv = self.root / "leaderboard.csv"
        self.statuses: dict[str, Any] = {}
        self.records: list[dict[str, Any]] = []
        if self.status_path.is_file():
            try:
                self.statuses = load_json(self.status_path)
            except json.JSONDecodeError:
                self.statuses = {}

    def temporary_config(self, cfg: dict[str, Any]) -> Path:
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".yaml",
            prefix="snn-tulu3-gif-tuning-v2-",
            dir="/tmp",
            delete=False,
        )
        try:
            yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
        finally:
            handle.close()
        return Path(handle.name)

    def command(self, argv: list[str], *, description: str) -> None:
        print(f"[{description}] {' '.join(argv)}", flush=True)
        completed = subprocess.run(argv, cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"{description} exited with status {completed.returncode}"
            )

    def write_manifest(self, winner: dict[str, Any] | None = None) -> None:
        atomic_json(
            self.manifest_path,
            {
                "plan_version": "tulu3_gif_aware_tuning_v2",
                "source_config": str(self.source_path),
                "source_config_sha256": self.source_hash,
                "selection_metric": "unweighted_mean_of_six_task_metrics",
                "task_names": list(TASK_NAMES),
                "num_processes": self.num_processes,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "grid": {
                    "calibration.num_samples": list(CALIBRATION_NUM_SAMPLES),
                    "gif_ratio_pairs": [
                        {
                            "gif.low_ratio": float(low),
                            "gif.salient_ratio": float(salient),
                        }
                        for low, salient in GIF_RATIO_PAIRS
                    ],
                    "candidate_count": (
                        len(CALIBRATION_NUM_SAMPLES) * len(GIF_RATIO_PAIRS)
                    ),
                },
                "fixed_constraints": {
                    "training.train_samples": 10000,
                    "training.learning_rate": 5e-6,
                    "training.lr_scheduler_type": "cosine",
                    "training.warmup_ratio": 0.01,
                    "training.num_train_epochs": 1,
                    "training.gradient_accumulation_steps": 16,
                    "training.max_grad_norm": 1.0,
                    "gif.quantizer_clip_backward": "hard_clip",
                    "replacement.outer_clip_backward": "hard_clip",
                    "phase.T": 4,
                    "mtn.T": 4,
                    "calibration.group_size": 128,
                    "ann_training.prefix_enabled": True,
                    "replacement.common_clip_enabled": True,
                    "ann_training_memory.attention_core_checkpoint": True,
                    "ann_training_memory.mlp_checkpoint": True,
                },
                "workflow_per_candidate": [
                    "prepare calibration manifest (once per num_samples)",
                    "validate or generate Pre-finetuning Prefix (once per num_samples)",
                    "ANN-training calibration Stage A",
                    "ANN-training calibration Stage B",
                    "ANN fine-tuning",
                    "six-task lm-eval",
                ],
                "winner": winner,
            },
        )

    def persist(self) -> None:
        atomic_json(self.status_path, self.statuses)
        ordered = sorted(
            self.records,
            key=lambda item: (
                item.get("mean") is None,
                -(item.get("mean") or -math.inf),
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
            fieldnames = [
                "label",
                "status",
                "mean",
                "eval_loss",
                "calibration_num_samples",
                "gif_low_ratio",
                "gif_salient_ratio",
                *TASK_NAMES,
                "ann_dir",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for item in ordered:
                writer.writerow(
                    {
                        key: (
                            item.get("task_metrics", {}).get(key)
                            if key in TASK_NAMES
                            else item.get(key)
                        )
                        for key in fieldnames
                    }
                )
        os.replace(temporary, self.leaderboard_csv)

    def update_status(
        self, label: str, status: str, layout: ArtifactLayout, **extra: Any
    ) -> None:
        self.statuses[label] = {
            "status": status,
            "ann_dir": str(layout.ann_dir),
            "calibration_stage_a": str(layout.ann_training_site_dir),
            "calibration_stage_b": str(layout.ann_training_clip_profile_dir),
            "updated_at_unix": time.time(),
            **extra,
        }
        self.persist()

    def ensure_prefix(
        self, cfg: dict[str, Any], layout: ArtifactLayout, config_path: Path
    ) -> None:
        try:
            validate_prefix_discovery_state(
                cfg, layout, layout.ann_training_prefix_dir
            )
            print(
                f"[REUSE-PREFIX] {layout.ann_training_prefix_dir}", flush=True
            )
            return
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            pass
        self.command(
            [
                sys.executable,
                "scripts/discover_prefix.py",
                "--config",
                str(config_path),
                "--stage",
                "pre_finetuning",
            ],
            description="PREFIX",
        )
        validate_prefix_discovery_state(
            cfg, layout, layout.ann_training_prefix_dir
        )

    def ensure_calibration(
        self, cfg: dict[str, Any], layout: ArtifactLayout, config_path: Path
    ) -> None:
        if calibration_is_valid(cfg, layout):
            print(
                f"[REUSE-CALIBRATION] {layout.ann_training_calibration_dir}",
                flush=True,
            )
            return
        if (layout.ann_dir / "training_result.json").exists():
            raise RuntimeError(
                "Training exists but its Stage A/B calibration is missing or invalid; "
                "refusing to replace provenance in place"
            )
        for phase in ("A", "B"):
            self.command(
                [
                    sys.executable,
                    "scripts/calibrate_sites.py",
                    "--config",
                    str(config_path),
                    "--stage",
                    "ann_training",
                    "--calibration-phase",
                    phase,
                ],
                description=f"CALIBRATION-{phase}",
            )
        if not calibration_is_valid(cfg, layout):
            raise RuntimeError("Generated ANN-training Stage A/B failed validation")

    def run_candidate(
        self, label: str, cfg: dict[str, Any], *, prepare_manifest: bool
    ) -> dict[str, Any]:
        layout = ArtifactLayout(cfg)
        run_config = self.temporary_config(cfg)
        try:
            existing = completed_summary(layout.ann_dir)
            if existing is not None:
                path, summary = existing
                record = record_from_summary(
                    label=label,
                    cfg=cfg,
                    layout=layout,
                    summary_path=path,
                    summary=summary,
                    status="reused_v2",
                )
                self.records.append(record)
                self.update_status(
                    label, "reused_v2", layout, mean=record["mean"]
                )
                print(f"[REUSE] {label}: mean={record['mean']:.6f}", flush=True)
                return record

            self.update_status(label, "preparing", layout)
            if prepare_manifest:
                self.command(
                    [
                        sys.executable,
                        "scripts/prepare_data.py",
                        "--config",
                        str(run_config),
                        "--calibration-only",
                    ],
                    description="DATA",
                )
            if not layout.calibration_data_manifest_path.is_file():
                raise FileNotFoundError(layout.calibration_data_manifest_path)
            self.ensure_prefix(cfg, layout, run_config)

            self.update_status(label, "calibrating", layout)
            self.ensure_calibration(cfg, layout, run_config)

            try:
                validate_training_result(cfg, layout)
                training_complete = True
            except FileNotFoundError:
                training_complete = False
            if training_complete:
                print(f"[RESUME-EVAL] {label}: valid training found", flush=True)
            else:
                self.update_status(label, "training", layout)
                self.command(
                    [
                        "torchrun",
                        "--standalone",
                        f"--nproc_per_node={self.num_processes}",
                        "scripts/train_ann.py",
                        "--config",
                        str(run_config),
                    ],
                    description="TRAIN",
                )
                validate_training_result(cfg, layout)

            self.update_status(label, "evaluating", layout)
            self.command(
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
                description="EVAL",
            )
            completed = completed_summary(layout.ann_dir)
            if completed is None:
                raise RuntimeError(
                    "lm-eval succeeded but no complete six-task summary was found"
                )
            path, summary = completed
            record = record_from_summary(
                label=label,
                cfg=cfg,
                layout=layout,
                summary_path=path,
                summary=summary,
                status="completed",
            )
            self.records.append(record)
            self.update_status(label, "completed", layout, mean=record["mean"])
            print(f"[DONE] {label}: mean={record['mean']:.6f}", flush=True)
            return record
        except Exception as exc:
            record = {
                "label": label,
                "status": "failed",
                "mean": None,
                "calibration_num_samples": int(
                    cfg["calibration"]["num_samples"]
                ),
                "gif_low_ratio": float(cfg["gif"]["low_ratio"]),
                "gif_salient_ratio": float(cfg["gif"]["salient_ratio"]),
                "ann_dir": str(layout.ann_dir),
                "error": str(exc),
            }
            self.records.append(record)
            self.update_status(label, "failed", layout, error=str(exc))
            print(f"[FAILED] {label}: {exc}; continuing.", flush=True)
            return record
        finally:
            run_config.unlink(missing_ok=True)
            if sha256(self.source_path) != self.source_hash:
                raise RuntimeError("Source YAML changed while the sweep was running")


def main() -> int:
    args = parse_args()
    if args.num_processes != 3:
        raise ValueError("v2 requires exactly --num-processes 3")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2":
        raise ValueError("v2 requires CUDA_VISIBLE_DEVICES=0,1,2")
    source_path = Path(args.config)
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path
    source_hash = sha256(source_path)
    source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    validate_source(source)
    base = fixed_base_config(source)
    run = GridRun(source_path, source_hash, base, args.num_processes)
    run.write_manifest()

    first_for_num_samples = {value: True for value in CALIBRATION_NUM_SAMPLES}
    for num_samples in CALIBRATION_NUM_SAMPLES:
        for low_ratio, salient_ratio in GIF_RATIO_PAIRS:
            label = candidate_label(num_samples, low_ratio, salient_ratio)
            cfg = candidate_config(
                base,
                num_samples=num_samples,
                low_ratio=low_ratio,
                salient_ratio=salient_ratio,
            )
            record = run.run_candidate(
                label,
                cfg,
                prepare_manifest=first_for_num_samples[num_samples],
            )
            if (
                record["status"] != "failed"
                or ArtifactLayout(cfg).calibration_data_manifest_path.is_file()
            ):
                first_for_num_samples[num_samples] = False

    usable = [record for record in run.records if record.get("mean") is not None]
    winner = (
        max(usable, key=lambda record: float(record["mean"])) if usable else None
    )
    run.write_manifest(winner=winner)
    run.persist()
    failed = [record for record in run.records if record["status"] == "failed"]
    if winner is None:
        print("Grid finished with no usable candidates.", flush=True)
    else:
        print(
            f"Grid finished: best={winner['label']} "
            f"mean={winner['mean']:.6f}; failed={len(failed)}",
            flush=True,
        )
    print(f"Leaderboard: {run.leaderboard_json}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
