from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .config import (
    conversion_prefix_enabled,
    is_aware_ann_mode,
    save_yaml,
    training_common_clip_enabled,
    use_post_finetuning_artifacts,
)


def prefix_enabled_dirname(enabled: bool) -> str:
    """Stable artifact suffix (the historical ``ture`` spelling is intentional)."""
    return "prefix_enabled_ture" if enabled else "prefix_enabled_false"


def conversion_artifact_source_dirname(use_post: bool) -> str:
    return f"use_post_finetuning_artifacts_{str(bool(use_post)).lower()}"


def ann_run_variant_dirname(
    *, prefix_enabled: bool, common_clip_enabled: bool, aware_mode: bool
) -> str:
    result = prefix_enabled_dirname(prefix_enabled)
    if aware_mode:
        result += (
            "_common_clip_enabled_true"
            if common_clip_enabled
            else "_common_clip_enabled_false"
        )
    return result


def phase_training_dirname(*, phase_T: Any, mtn_T: Any, surrogate_slope: Any, warmup_ratio: Any) -> str:
    return (
        f"phase_T_{int(phase_T)}_mtn_T_{int(mtn_T)}_"
        f"surrogate_slope_{float(surrogate_slope)}"
        f"_warmup_ratio_{float(warmup_ratio)}"
    )


def gif_training_dirname(*, phase_T: Any, mtn_T: Any, warmup_ratio: Any) -> str:
    return f"phase_T_{int(phase_T)}_mtn_T_{int(mtn_T)}_warmup_ratio_{float(warmup_ratio)}"


def clip_profile_dirname(phase_T: Any, mtn_T: Any) -> str:
    return f"phase_T_{int(phase_T)}_mtn_T_{int(mtn_T)}"


def phase_snn_dirname(phase_T: Any) -> str:
    return f"phase_T_{int(phase_T)}"


def mtn_snn_dirname(mtn_T: Any, mtn_K: Any) -> str:
    return f"mtn_T_{int(mtn_T)}_mtn_K_{int(mtn_K)}"


def calibration_group_dirname(group_size: Any) -> str:
    value = int(group_size)
    if value != -1 and value <= 0:
        raise ValueError("calibration.group_size must be -1 or a positive integer")
    return f"calibration_group_size_{value}"


def calibration_variant_dirname(group_size: Any, num_samples: Any) -> str:
    samples = int(num_samples)
    if samples <= 0:
        raise ValueError("calibration.num_samples must be a positive integer")
    return f"{calibration_group_dirname(group_size)}_num_samples_{samples}"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


class ArtifactLayout:
    def __init__(self, cfg: dict[str, Any]):
        self._cfg = cfg
        exp = cfg["experiment"]
        model = safe_name(exp["model_name"])
        experiment_root = Path(exp["output_root"]) / exp["id"]
        task_root = experiment_root / exp["task"]
        model_root = task_root / model
        seed = f"seed{int(exp['seed'])}"
        learning_rate = f"lr{cfg['training']['learning_rate']}"
        if exp["task"] == "tldr":
            configured_train_samples = cfg["training"].get("tldr_train_samples")
            if configured_train_samples is None:
                train_samples = "full"
            else:
                configured_train_samples = int(configured_train_samples)
                if configured_train_samples <= 0:
                    raise ValueError(
                        "training.tldr_train_samples must be a positive integer or null"
                    )
                train_samples = str(configured_train_samples)
            learning_rate = f"{learning_rate}_train_samples_{train_samples}"
        if is_aware_ann_mode(cfg):
            learning_rate = (
                f"num_samples_{int(cfg['calibration']['num_samples'])}_"
                f"{learning_rate}_"
                f"{calibration_group_dirname(cfg['calibration']['group_size'])}"
            )

        self.model_root = model_root
        self.seed_name = seed
        ann_prefix = bool(
            cfg.get("ann_training", {}).get(
                "prefix_enabled", cfg.get("prefix", {}).get("enabled", False)
            )
        )
        run_variant = ann_run_variant_dirname(
            prefix_enabled=ann_prefix,
            common_clip_enabled=training_common_clip_enabled(cfg),
            aware_mode=is_aware_ann_mode(cfg),
        )
        run_root = (
            model_root
            / exp["ann_mode"]
            / learning_rate
            / run_variant
        )
        if exp["ann_mode"] == "phase_aware":
            run_root = run_root / phase_training_dirname(
                phase_T=cfg["phase"]["T"], mtn_T=cfg["mtn"]["T"],
                surrogate_slope=cfg["phase"]["surrogate_slope"],
                warmup_ratio=cfg["training"]["warmup_ratio"],
            )
        elif exp["ann_mode"] == "gif_aware":
            run_root = run_root / gif_training_dirname(
                phase_T=cfg["phase"]["T"], mtn_T=cfg["mtn"]["T"],
                warmup_ratio=cfg["training"]["warmup_ratio"],
            )
        self.root = run_root / seed
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
    def calibration_data_dir(self) -> Path:
        return self.data_dir / "calibration" / f"num_samples_{int(self._cfg['calibration']['num_samples'])}"

    @property
    def calibration_data_manifest_path(self) -> Path:
        return self.calibration_data_dir / "calibration_manifest.json"

    @property
    def canonical_preprocessing_calibration_dir(self) -> Path:
        return self.data_dir / "canonical_preprocessing" / "num_samples_128"

    @property
    def canonical_preprocessing_calibration_manifest_path(self) -> Path:
        return self.canonical_preprocessing_calibration_dir / "calibration_manifest.json"
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
    def ann_training_prefix_base_dir(self) -> Path:
        return self.policy_root / "pre_finetuning_prefix"

    @property
    def ann_training_prefix_dir(self) -> Path:
        return self.ann_training_prefix_base_dir / f'num_samples_{int(self._cfg["calibration"]["num_samples"])}'

    @property
    def ann_training_prefix_config_dir(self) -> Path:
        return self.ann_training_prefix_dir / "config"

    @property
    def ann_training_prefix_logs_dir(self) -> Path:
        return self.ann_training_prefix_dir / "logs"
    @property
    def rotated_pre_finetuning_dir(self) -> Path:
        """Shared artifacts for evaluating the rotated Base before ANN fine-tuning."""
        return self.shared_model_root / "rotated_prefix" / "rotated_pre_finetuning"

    @property
    def rotated_pre_finetuning_config_dir(self) -> Path:
        return self.rotated_pre_finetuning_dir / "config"

    @property
    def rotated_pre_finetuning_logs_dir(self) -> Path:
        return self.rotated_pre_finetuning_dir / "logs"

    @property
    def rotated_pre_finetuning_prefix_dir(self) -> Path:
        """Alias of the Prefix used by ANN training; these are one object."""
        return self.ann_training_prefix_dir

    @property
    def rotated_pre_finetuning_evaluation_dir(self) -> Path:
        return self.rotated_pre_finetuning_dir / "evaluation"

    @property
    def ann_training_calibration_dir(self) -> Path:
        enabled = bool(
            self._cfg.get("ann_training", {}).get(
                "prefix_enabled", self._cfg.get("prefix", {}).get("enabled", False)
            )
        )
        return (
            self.shared_model_root
            / "rotated_prefix"
            / "ann_training_calibration"
            / prefix_enabled_dirname(enabled)
            / calibration_variant_dirname(
                self._cfg["calibration"]["group_size"],
                self._cfg["calibration"]["num_samples"],
            )
        )

    @property
    def ann_training_site_dir(self) -> Path:
        return self.ann_training_calibration_dir / "sites"

    @property
    def ann_training_clip_profiles_dir(self) -> Path:
        return self.ann_training_calibration_dir / "clip_profiles"

    @property
    def ann_training_clip_profile_dir(self) -> Path:
        return self.ann_training_clip_profiles_dir / clip_profile_dirname(
            self._cfg["phase"]["T"], self._cfg["mtn"]["T"]
        )

    @property
    def ann_training_clip_profile_config_dir(self) -> Path:
        return self.ann_training_clip_profile_dir / "config"

    @property
    def ann_training_clip_profile_logs_dir(self) -> Path:
        return self.ann_training_clip_profile_dir / "logs"

    @property
    def ann_training_calibration_config_dir(self) -> Path:
        return self.ann_training_calibration_dir / "config"

    @property
    def ann_training_calibration_logs_dir(self) -> Path:
        return self.ann_training_calibration_dir / "logs"

    @property
    def vanilla_analysis_calibration_dir(self) -> Path:
        return (
            self.shared_model_root
            / "vanilla_original"
            / "vanilla_analysis_calibration"
            / calibration_variant_dirname(self._cfg["calibration"]["group_size"], self._cfg["calibration"]["num_samples"])
        )

    @property
    def vanilla_analysis_site_dir(self) -> Path:
        return self.vanilla_analysis_calibration_dir / "sites"

    @property
    def vanilla_analysis_calibration_config_dir(self) -> Path:
        return self.vanilla_analysis_calibration_dir / "config"

    @property
    def vanilla_analysis_calibration_logs_dir(self) -> Path:
        return self.vanilla_analysis_calibration_dir / "logs"

    @property
    def prefix_dir(self) -> Path:
        return self.ann_training_prefix_dir

    @property
    def calibration_dir(self) -> Path:
        return self.ann_training_calibration_dir

    @property
    def site_dir(self) -> Path:
        return self.ann_training_site_dir

    @property
    def ann_dir(self) -> Path:
        return self.root / "ann"

    @property
    def ann_checkpoint_dir(self) -> Path:
        return self.ann_dir / "final"

    @property
    def post_finetuning_dir(self) -> Path:
        return self.root / "post_finetuning"

    @property
    def post_finetuning_prefix_config_dir(self) -> Path:
        return self.post_finetuning_prefix_dir / "config"

    @property
    def post_finetuning_prefix_logs_dir(self) -> Path:
        return self.post_finetuning_prefix_dir / "logs"

    @property
    def post_finetuning_prefix_base_dir(self) -> Path:
        return self.post_finetuning_dir / "prefix"

    @property
    def post_finetuning_prefix_dir(self) -> Path:
        return self.post_finetuning_prefix_base_dir / f'num_samples_{int(self._cfg["calibration"]["num_samples"])}'
    @property
    def post_finetuning_conversion_calibration_dir(self) -> Path:
        enabled = bool(
            self._cfg.get("post_finetuning", {}).get("prefix_enabled", True)
        )
        return (
            self.post_finetuning_dir
            / "conversion_calibration"
            / prefix_enabled_dirname(enabled)
            / calibration_variant_dirname(self._cfg["calibration"]["group_size"], self._cfg["calibration"]["num_samples"])
        )

    @property
    def post_finetuning_site_dir(self) -> Path:
        return self.post_finetuning_conversion_calibration_dir / "sites"

    @property
    def post_finetuning_clip_profiles_dir(self) -> Path:
        return self.post_finetuning_conversion_calibration_dir / "clip_profiles"

    @property
    def post_finetuning_clip_profile_dir(self) -> Path:
        return self.post_finetuning_clip_profiles_dir / clip_profile_dirname(self._cfg["phase"]["T"], self._cfg["mtn"]["T"])

    @property
    def post_finetuning_clip_profile_config_dir(self) -> Path:
        return self.post_finetuning_clip_profile_dir / "config"

    @property
    def post_finetuning_clip_profile_logs_dir(self) -> Path:
        return self.post_finetuning_clip_profile_dir / "logs"

    @property
    def post_finetuning_conversion_calibration_config_dir(self) -> Path:
        return self.post_finetuning_conversion_calibration_dir / "config"

    @property
    def post_finetuning_conversion_calibration_logs_dir(self) -> Path:
        return self.post_finetuning_conversion_calibration_dir / "logs"

    @property
    def conversion_prefix_dir(self) -> Path:
        return (
            self.post_finetuning_prefix_dir
            if use_post_finetuning_artifacts(self._cfg)
            else self.ann_training_prefix_dir
        )

    @property
    def conversion_calibration_dir(self) -> Path:
        return (
            self.post_finetuning_conversion_calibration_dir
            if use_post_finetuning_artifacts(self._cfg)
            else self.ann_training_calibration_dir
        )

    @property
    def conversion_site_dir(self) -> Path:
        return self.conversion_calibration_dir / "sites"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def snn_dir(self, neuron: str) -> Path:
        base = self.root / "snn" / conversion_artifact_source_dirname(
            use_post_finetuning_artifacts(self._cfg)
        )
        if is_aware_ann_mode(self._cfg):
            result = base / neuron
        else:
            result = base / calibration_variant_dirname(
                self._cfg["calibration"]["group_size"],
                self._cfg["calibration"]["num_samples"],
            ) / neuron
        if neuron == "phase":
            return result / phase_snn_dirname(self._cfg["phase"]["T"])
        if neuron == "mtn":
            return result / mtn_snn_dirname(self._cfg["mtn"]["T"], self._cfg["mtn"]["K"])
        return result

    def snn_conversion_dir(self, neuron: str) -> Path:
        enabled = conversion_prefix_enabled(self._cfg)
        return self.snn_dir(neuron) / "conversion" / prefix_enabled_dirname(enabled)

    def ensure(self) -> None:
        for path in (
            self.config_dir,
            self.data_dir,
            self.canonical_preprocessing_calibration_dir,
            self.post_finetuning_prefix_config_dir,
            self.post_finetuning_prefix_logs_dir,
            self.shared_task_logs_dir,
            self.rotation_dir,
            self.ann_training_prefix_dir,
            self.rotated_pre_finetuning_config_dir,
            self.rotated_pre_finetuning_logs_dir,
            self.rotated_pre_finetuning_prefix_dir,
            self.rotated_pre_finetuning_evaluation_dir,
            self.ann_training_calibration_dir,
            self.ann_training_calibration_config_dir,
            self.ann_training_calibration_logs_dir,
            self.ann_training_clip_profile_config_dir,
            self.ann_training_clip_profile_logs_dir,
            self.ann_training_site_dir,
            self.vanilla_analysis_calibration_dir,
            self.vanilla_analysis_calibration_config_dir,
            self.vanilla_analysis_calibration_logs_dir,
            self.vanilla_analysis_site_dir,
            self.post_finetuning_prefix_dir,
            self.post_finetuning_conversion_calibration_dir,
            self.post_finetuning_clip_profile_config_dir,
            self.post_finetuning_clip_profile_logs_dir,
            self.post_finetuning_conversion_calibration_config_dir,
            self.post_finetuning_conversion_calibration_logs_dir,
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
