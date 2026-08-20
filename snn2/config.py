from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .sites import SITE_COUNT


ANN_MODES = {"vanilla", "unaware", "phase_aware", "gif_aware"}
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
    mode = cfg["experiment"]["ann_mode"]
    if mode == "vanilla":
        cfg["rotation"]["enabled"] = False
        cfg["prefix"]["enabled"] = False
        cfg["replacement"]["train_mode"] = "none"
    elif mode == "unaware":
        cfg["rotation"]["enabled"] = True
        cfg["prefix"]["enabled"] = True
        cfg["replacement"]["train_mode"] = "none"
    elif mode == "phase_aware":
        cfg["rotation"]["enabled"] = True
        cfg["prefix"]["enabled"] = True
        cfg["replacement"]["train_mode"] = "phase"
    elif mode == "gif_aware":
        cfg["rotation"]["enabled"] = True
        cfg["prefix"]["enabled"] = True
        cfg["replacement"]["train_mode"] = "gif"
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    required = {
        "experiment",
        "data",
        "rotation",
        "prefix",
        "calibration",
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
    if int(cfg["data"]["max_seq_length"]) != 2048:
        raise ValueError("Main experiments require max_seq_length=2048")
    if bool(cfg["data"].get("packing", True)):
        raise ValueError("Packing must be disabled")
    if not bool(cfg["data"].get("truncation", False)):
        raise ValueError("Truncation must be enabled")
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
    for key in ("rediscover_prefix", "recalibrate_sites", "prefix_enabled", "post_finetuning_recalibration"):
        if not bool(cfg["post_finetuning"].get(key, False)):
            raise ValueError(f"Main experiments require post_finetuning.{key}=true")


def training_prefix_enabled(cfg: dict[str, Any]) -> bool:
    return cfg["experiment"]["ann_mode"] != "vanilla" and bool(cfg["prefix"].get("enabled", False))


def post_finetuning_prefix_enabled(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("post_finetuning", {}).get("prefix_enabled", True))


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
