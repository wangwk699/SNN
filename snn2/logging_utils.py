from __future__ import annotations

import json
import logging
import os
import platform
import socket
import subprocess
import time
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def build_logger(name: str, log_path: str | Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def software_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }
    for name in ("torch", "transformers", "datasets", "trl", "accelerate", "deepspeed"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # environment audit should not mask the real stage
            versions[name] = f"unavailable: {type(exc).__name__}"
    try:
        import torch

        versions["cuda_runtime"] = torch.version.cuda
        versions["cuda_device_count"] = torch.cuda.device_count()
        versions["cuda_devices"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    except Exception:
        versions["cuda_devices"] = []
    try:
        versions["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        versions["git_commit"] = "unknown"
    return versions


class StageRun(AbstractContextManager["StageRun"]):
    def __init__(self, stage: str, logs_dir: str | Path, metadata: dict[str, Any] | None = None):
        self.stage = stage
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = rank
        self.world_size = world_size
        file_stage = f"{stage}_rank{rank}" if world_size > 1 and not stage.endswith(f"rank{rank}") else stage
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata or {}
        self.started = 0.0
        self.started_at = ""
        self.logger = build_logger(file_stage, self.logs_dir / f"{file_stage}.log")
        self.events = self.logs_dir / f"{file_stage}.jsonl"
        self.result_path = self.logs_dir / f"{file_stage}_result.json"

    def __enter__(self) -> "StageRun":
        self.started = time.perf_counter()
        self.started_at = utc_now()
        try:
            import torch

            if torch.cuda.is_available():
                for index in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(index)
        except Exception:
            pass
        append_jsonl(
            self.events,
            {"event": "stage_start", "stage": self.stage, "time": self.started_at, **self.metadata},
        )
        self.logger.info("stage_start %s", self.stage)
        return self

    def event(self, name: str, **payload: Any) -> None:
        append_jsonl(self.events, {"event": name, "time": utc_now(), **payload})

    def __exit__(self, exc_type, exc, traceback) -> bool:
        elapsed = time.perf_counter() - self.started
        peak_bytes = None
        peak_by_device = None
        try:
            import torch

            if torch.cuda.is_available():
                peak_by_device = {
                    str(index): int(torch.cuda.max_memory_allocated(index))
                    for index in range(torch.cuda.device_count())
                }
                peak_bytes = max(peak_by_device.values(), default=0)
        except Exception:
            pass
        result = {
            "stage": self.stage,
            "status": "failed" if exc is not None else "completed",
            "started_at": self.started_at,
            "finished_at": utc_now(),
            "elapsed_seconds": elapsed,
            "peak_cuda_memory_bytes": peak_bytes,
            "peak_cuda_memory_by_device_bytes": peak_by_device,
            "rank": self.rank,
            "world_size": self.world_size,
            "pid": os.getpid(),
            "metadata": self.metadata,
            "software": software_versions(),
        }
        if exc is not None:
            result["error_type"] = type(exc).__name__
            result["error"] = str(exc)
        write_json(self.result_path, result)
        append_jsonl(self.events, {"event": "stage_end", "time": utc_now(), **result})
        self.logger.info("stage_end status=%s elapsed=%.3fs", result["status"], elapsed)
        return False
