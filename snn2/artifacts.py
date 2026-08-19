from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .config import save_yaml


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


class ArtifactLayout:
    def __init__(self, cfg: dict[str, Any]):
        exp = cfg["experiment"]
        model = safe_name(exp["model_name"])
        experiment_root = Path(exp["output_root"]) / exp["id"]
        task_root = experiment_root / exp["task"]
        model_root = task_root / model
        seed = f"seed{int(exp['seed'])}"
        learning_rate = f"lr{cfg['training']['learning_rate']}"

        self.model_root = model_root
        self.seed_name = seed
        self.root = model_root / exp["ann_mode"] / learning_rate / seed
        # 原始 Base 模型独立目录：
        # 不依赖 ann_mode，也不依赖 learning_rate
        self.base_root = model_root / "base" / seed

        self.shared_task_root = task_root / "_shared" / seed
        self.shared_model_root = model_root / "_shared" / seed
        policy = "rotated_prefix" if cfg["rotation"]["enabled"] else "vanilla_original"
        self.policy_root = self.shared_model_root / policy

    @property
    def base_dir(self) -> Path:
        return self.base_root

    @property
    def base_config_dir(self) -> Path:
        return self.base_root / "config"

    @property
    def base_logs_dir(self) -> Path:
        return self.base_root / "logs"

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def data_dir(self) -> Path:
        return self.shared_task_root / "data"

    @property
    def shared_task_logs_dir(self) -> Path:
        return self.shared_task_root / "logs"

    @property
    def policy_logs_dir(self) -> Path:
        return self.policy_root / "logs"

    @property
    def shared_task_config_dir(self) -> Path:
        return self.shared_task_root / "config"

    @property
    def policy_config_dir(self) -> Path:
        return self.policy_root / "config"

    @property
    def rotation_dir(self) -> Path:
        return self.shared_model_root / "rotated_prefix" / "rotation"

    @property
    def ann_training_prefix_dir(self) -> Path:
        return self.shared_model_root / "rotated_prefix" / "ann_training_prefix"

    @property
    def ann_training_calibration_dir(self) -> Path:
        return self.shared_model_root / "rotated_prefix" / "ann_training_calibration"

    @property
    def ann_training_site_dir(self) -> Path:
        return self.ann_training_calibration_dir / "sites"

    @property
    def vanilla_analysis_calibration_dir(self) -> Path:
        return self.shared_model_root / "vanilla_original" / "vanilla_analysis_calibration"

    @property
    def vanilla_analysis_site_dir(self) -> Path:
        return self.vanilla_analysis_calibration_dir / "sites"

    @property
    def prefix_dir(self) -> Path:
        """Legacy alias; new code must select an explicit prefix stage."""
        return self.ann_training_prefix_dir

    @property
    def calibration_dir(self) -> Path:
        """Legacy alias; new code must select an explicit calibration stage."""
        return self.ann_training_calibration_dir

    @property
    def site_dir(self) -> Path:
        return self.ann_training_site_dir

    @property
    def ann_dir(self) -> Path:
        return self.root / "ann"

    @property
    def ann_checkpoint_dir(self) -> Path:
        """Canonical final fine-tuned ANN checkpoint for all downstream stages."""
        return self.ann_dir / "final"

    @property
    def post_finetuning_dir(self) -> Path:
        return self.root / "post_finetuning"

    @property
    def post_finetuning_prefix_dir(self) -> Path:
        return self.post_finetuning_dir / "prefix"

    @property
    def post_finetuning_conversion_calibration_dir(self) -> Path:
        return self.post_finetuning_dir / "conversion_calibration"

    @property
    def post_finetuning_site_dir(self) -> Path:
        return self.post_finetuning_conversion_calibration_dir / "sites"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def snn_dir(self, neuron: str) -> Path:
        return self.root / "snn" / neuron

    def ensure(self) -> None:
        for path in (
            self.config_dir,
            self.data_dir,
            self.shared_task_logs_dir,
            self.rotation_dir,
            self.ann_training_prefix_dir,
            self.ann_training_calibration_dir,
            self.ann_training_site_dir,
            self.vanilla_analysis_calibration_dir,
            self.vanilla_analysis_site_dir,
            self.post_finetuning_prefix_dir,
            self.post_finetuning_conversion_calibration_dir,
            self.post_finetuning_site_dir,
            self.policy_logs_dir,
            self.ann_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def write_resolved_config(self, cfg: dict[str, Any], config_dir: Path | None = None) -> Path:
        config_dir = config_dir or self.config_dir
        config_dir.mkdir(parents=True, exist_ok=True)
        path = config_dir / "resolved_config.yaml"
        save_yaml(cfg, path)
        return path

def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        handle.write("\n")
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            pass
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
