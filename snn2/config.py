from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from .sites import SITE_COUNT
from .temporal_ops import (
    EMBEDDING_TEMPORAL_POLICY,
    GIF_ADD_BITS,
    GIF_BASE_BITS,
    GIF_HIGH_QMAX,
    GIF_LOCAL_STEPS,
    GIF_STEP_QMAX,
    PREFIX_TEMPORAL_POLICY,
    PHASE_FINAL_NORM_POLICY,
    PHASE_TAU_CALIBRATION,
    PHASE_TAU_EMA_FACTOR,
    PHASE_TAU_ACCUMULATOR_DTYPE,
    SOFTMAX_PREFIX_NEURON_POLICY,
    TEMPORAL_IMPLEMENTATION,
    TEMPORAL_LAYOUT,
    TEMPORAL_LINEAR_BIAS_POLICY,
)


ANN_MODES = {"vanilla", "unaware", "phase_aware", "gif_aware"}
AWARE_ANN_MODES = {"phase_aware", "gif_aware"}
SNN_NEURONS = {"phase", "gif", "mtn"}


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    cfg = resolve_config(cfg)
    validate_config(cfg)
    cfg.setdefault("_meta", {})
    cfg["_meta"].update(
        {
            "source_config": str(path.resolve()),
            "config_sha256": config_hash(cfg),
        }
    )
    return cfg


def resolve_config(raw: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(raw)
    cfg["replacement"].setdefault("common_clip_enabled", True)
    cfg.setdefault("ann_training", {})
    cfg.setdefault("rotated_pre_finetuning", {})
    cfg["rotated_pre_finetuning"].setdefault("prefix_enabled", True)
    cfg["ann_training"].setdefault(
        "prefix_enabled", bool(cfg.get("prefix", {}).get("enabled", True))
    )
    cfg.setdefault("evaluation", {})
    cfg["evaluation"].setdefault(
        "prefix_enabled",
        bool(cfg.get("post_finetuning", {}).get("prefix_enabled", True)),
    )
    mode = cfg["experiment"]["ann_mode"]
    if mode == "vanilla":
        cfg["rotation"]["enabled"] = False
        cfg["ann_training"]["prefix_enabled"] = False
        cfg["prefix"]["enabled"] = False
        cfg["replacement"]["train_mode"] = "none"
        cfg["replacement"]["common_clip_enabled"] = False
    elif mode == "unaware":
        cfg["rotation"]["enabled"] = True
        cfg["prefix"]["enabled"] = bool(cfg["ann_training"]["prefix_enabled"])
        cfg["replacement"]["train_mode"] = "none"
        cfg["replacement"]["common_clip_enabled"] = False
    elif mode == "phase_aware":
        cfg["rotation"]["enabled"] = True
        cfg["prefix"]["enabled"] = bool(cfg["ann_training"]["prefix_enabled"])
        cfg["replacement"]["train_mode"] = "phase"
        cfg["post_finetuning"].update({
            "rediscover_prefix": False,
            "recalibrate_sites": False,
            "post_finetuning_recalibration": False,
            "prefix_enabled": False,
        })
    elif mode == "gif_aware":
        cfg["rotation"]["enabled"] = True
        cfg["prefix"]["enabled"] = bool(cfg["ann_training"]["prefix_enabled"])
        cfg["replacement"]["train_mode"] = "gif"
        cfg["post_finetuning"].update({
            "rediscover_prefix": False,
            "recalibrate_sites": False,
            "post_finetuning_recalibration": False,
            "prefix_enabled": False,
        })
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    required = {
        "experiment",
        "data",
        "rotation",
        "prefix",
        "calibration",
        "deployment",
        "phase",
        "mtn",
        "gif",
        "replacement",
        "training",
        "evaluation",
        "post_finetuning",
    }
    missing = required - cfg.keys()
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")
    mode = cfg["experiment"].get("ann_mode")
    if mode not in ANN_MODES:
        raise ValueError(f"ann_mode must be one of {sorted(ANN_MODES)}, got {mode}")
    if int(cfg["calibration"]["num_samples"]) != 128:
        raise ValueError("Main experiments require exactly 128 calibration draws")
    expected_sites = int(cfg["calibration"]["expected_sites_per_layer"])
    if expected_sites != SITE_COUNT:
        raise ValueError(
            "calibration.expected_sites_per_layer must match "
            f"the code topology: config={expected_sites}, code={SITE_COUNT}"
        )
    if bool(cfg["calibration"].get("with_replacement", False)):
        raise ValueError("Calibration sampling must be done without replacement")
    group_size = cfg["calibration"].get("group_size")
    if (
        not isinstance(group_size, int)
        or isinstance(group_size, bool)
        or (group_size != -1 and group_size <= 0)
    ):
        raise ValueError("calibration.group_size must be -1 or a positive integer")
    if int(cfg["data"]["max_seq_length"]) != 2048:
        raise ValueError("Main experiments require max_seq_length=2048")
    try:
        surrogate_slope = float(cfg["phase"]["surrogate_slope"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "phase.surrogate_slope must be a positive finite number"
        ) from exc
    if not math.isfinite(surrogate_slope) or surrogate_slope <= 0.0:
        raise ValueError(
            "phase.surrogate_slope must be a positive finite number"
        )
    if bool(cfg["data"].get("packing", True)):
        raise ValueError("Packing must be disabled")
    if not bool(cfg["data"].get("truncation", False)):
        raise ValueError("Truncation must be enabled")
    if cfg["experiment"].get("task") == "tldr":
        configured_train_samples = cfg["training"].get("tldr_train_samples")
        if configured_train_samples is not None and int(configured_train_samples) <= 0:
            raise ValueError(
                "training.tldr_train_samples must be a positive integer or null"
            )
        int(cfg["training"].get("tldr_train_seed", 42))

    deployment = cfg["deployment"]
    expected_deployment = {
        "temporal_implementation": TEMPORAL_IMPLEMENTATION,
        "temporal_layout": TEMPORAL_LAYOUT,
        "linear_bias_policy": TEMPORAL_LINEAR_BIAS_POLICY,
        "prefix_temporal_policy": PREFIX_TEMPORAL_POLICY,
        "embedding_temporal_policy": EMBEDDING_TEMPORAL_POLICY,
        "softmax_prefix_neuron_policy": SOFTMAX_PREFIX_NEURON_POLICY,
        "phase_final_norm_policy": PHASE_FINAL_NORM_POLICY,
        "phase_tau_calibration": PHASE_TAU_CALIBRATION,
        "phase_tau_ema_factor": PHASE_TAU_EMA_FACTOR,
        "phase_tau_accumulator_dtype": PHASE_TAU_ACCUMULATOR_DTYPE,
    }
    unexpected_deployment = sorted(set(deployment) - set(expected_deployment))
    if unexpected_deployment:
        raise ValueError(
            "Unsupported temporal deployment keys; re-run materialize_configs.py: "
            f"{unexpected_deployment}"
        )
    mismatched_deployment = {
        key: (expected, deployment.get(key))
        for key, expected in expected_deployment.items()
        if deployment.get(key) != expected
    }
    if mismatched_deployment:
        raise ValueError(
            f"Unsupported temporal deployment policy: {mismatched_deployment}"
        )
    gif_policy = {
        "base_bits": GIF_BASE_BITS,
        "add_bits": GIF_ADD_BITS,
        "high_qmax": GIF_HIGH_QMAX,
        "temporal_steps": GIF_LOCAL_STEPS,
        "per_step_qmax": GIF_STEP_QMAX,
    }
    mismatched_gif = {
        key: (expected, cfg["gif"].get(key))
        for key, expected in gif_policy.items()
        if cfg["gif"].get(key) != expected
    }
    if mismatched_gif or 2 * int(cfg["gif"].get("per_step_qmax", -1)) != GIF_HIGH_QMAX:
        raise ValueError(f"Unsupported GIF qmax/chunk policy: {mismatched_gif}")

    if int(cfg["phase"]["T"]) <= 0 or int(cfg["mtn"]["T"]) <= 0:
        raise ValueError("Neuron timesteps must be positive")
    if int(cfg["mtn"]["K"]) <= 0:
        raise ValueError("MTN K must be positive")
    if int(cfg["gif"]["base_bits"]) < 2 or int(cfg["gif"]["add_bits"]) < 0:
        raise ValueError("Invalid GIF bit widths")
    if not 0.0 < float(cfg["gif"]["low_ratio"]) <= 1.0:
        raise ValueError("GIF low_ratio must be in (0, 1]")
    salient = float(cfg["gif"].get("salient_ratio", 1.0 - float(cfg["gif"]["low_ratio"])))
    if abs(float(cfg["gif"]["low_ratio"]) + salient - 1.0) > 1e-8:
        raise ValueError("GIF low_ratio + salient_ratio must equal 1")
    if cfg["gif"].get("runtime_quantization") != "static":
        raise ValueError("Main experiments require static GIF runtime quantization")
    if cfg["gif"].get("scale_initialization") != "direct_min_max":
        raise ValueError("Main experiments require direct min-max GIF initialization")
    if bool(cfg["gif"].get("mse_scale_refinement", True)):
        raise ValueError("Main experiments disable GIF MSE scale refinement")
    if mode != "vanilla" and not bool(cfg["rotation"].get("fused_weights_are_finetuned", False)):
        raise ValueError("Rotated modes must fine-tune the fused rotation weights")
    if float(cfg["rotation"].get("regression_relative_l2_threshold", 0.05)) <= 0.0:
        raise ValueError("rotation.regression_relative_l2_threshold must be positive")
    top1_threshold = float(
        cfg["rotation"].get("regression_top1_agreement_threshold", 0.95)
    )
    if not 0.0 <= top1_threshold < 1.0:
        raise ValueError(
            "rotation.regression_top1_agreement_threshold must be in [0, 1)"
        )
    expected_post = not is_aware_ann_mode(cfg)
    for key in ("rediscover_prefix", "recalibrate_sites", "post_finetuning_recalibration"):
        if bool(cfg["post_finetuning"].get(key, False)) != expected_post:
            raise ValueError(
                f"{mode} requires post_finetuning.{key}={str(expected_post).lower()}"
            )
    if mode == "vanilla" and training_prefix_enabled(cfg):
        raise ValueError("vanilla must not use a Pre-finetuning Prefix")
    if mode != "vanilla" and not training_prefix_enabled(cfg):
        raise ValueError(f"{mode} requires the shared Pre-finetuning Prefix")
    common_clip_enabled = cfg["replacement"].get("common_clip_enabled")
    if not isinstance(common_clip_enabled, bool):
        raise ValueError("replacement.common_clip_enabled must be true or false")
    if not is_aware_ann_mode(cfg) and common_clip_enabled:
        raise ValueError(f"{mode} requires replacement.common_clip_enabled=false")
    for section in ("ann_training", "rotated_pre_finetuning", "post_finetuning", "evaluation"):
        value = cfg[section].get("prefix_enabled")
        if not isinstance(value, bool):
            raise ValueError(f"{section}.prefix_enabled must be true or false")


def training_prefix_enabled(cfg: dict[str, Any]) -> bool:
    return bool(
        cfg.get("ann_training", {}).get(
            "prefix_enabled", cfg["prefix"].get("enabled", False)
        )
    )


def is_aware_ann_mode(cfg: dict[str, Any]) -> bool:
    return cfg["experiment"]["ann_mode"] in AWARE_ANN_MODES


def training_common_clip_enabled(cfg: dict[str, Any]) -> bool:
    return is_aware_ann_mode(cfg) and bool(
        cfg.get("replacement", {}).get("common_clip_enabled", True)
    )


def requires_pre_finetuning_prefix(cfg: dict[str, Any]) -> bool:
    return cfg["experiment"]["ann_mode"] != "vanilla"


def requires_ann_training_calibration(cfg: dict[str, Any]) -> bool:
    return is_aware_ann_mode(cfg)


def requires_post_finetuning_artifacts(cfg: dict[str, Any]) -> bool:
    return not is_aware_ann_mode(cfg)


def conversion_reuses_ann_training_artifacts(cfg: dict[str, Any]) -> bool:
    return is_aware_ann_mode(cfg)


def conversion_prefix_enabled(cfg: dict[str, Any]) -> bool:
    return training_prefix_enabled(cfg) if is_aware_ann_mode(cfg) else post_finetuning_prefix_enabled(cfg)


def conversion_calibration_stage(cfg: dict[str, Any]) -> str:
    return "ann_training" if is_aware_ann_mode(cfg) else "post_finetuning"


def final_evaluation_prefix_artifact_stage(cfg: dict[str, Any]) -> str:
    return "pre_finetuning" if is_aware_ann_mode(cfg) else "post_finetuning"


def post_finetuning_prefix_enabled(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("post_finetuning", {}).get("prefix_enabled", True))

def rotated_pre_finetuning_prefix_enabled(cfg: dict[str, Any]) -> bool:
    return bool(
        cfg.get("rotated_pre_finetuning", {}).get("prefix_enabled", True)
    )


def evaluation_prefix_enabled(cfg: dict[str, Any]) -> bool:
    return bool(
        cfg.get("evaluation", {}).get(
            "prefix_enabled", post_finetuning_prefix_enabled(cfg)
        )
    )


def post_finetuning_recalibration_enabled(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("post_finetuning", {}).get("post_finetuning_recalibration", True))


def config_hash(cfg: dict[str, Any]) -> str:
    payload = copy.deepcopy(cfg)
    payload.pop("_meta", None)
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)


def deep_get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node
