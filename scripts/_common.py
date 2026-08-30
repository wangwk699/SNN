from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from snn2.artifacts import ArtifactLayout
from snn2.config import load_config


def parser(
    description: str,
    neuron: bool = False,
    allow_ann: bool = False,
    deployment_overrides: bool = False,
) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--config", required=True, help="YAML experiment config")
    if neuron:
        choices = ("ann", "phase", "gif", "mtn") if allow_ann else ("phase", "gif", "mtn")
        result.add_argument("--neuron", required=True, choices=choices)
    if neuron or deployment_overrides:
        result.add_argument("--phase-T", type=int, default=None)
        result.add_argument("--mtn-T", type=int, default=None)
        result.add_argument("--mtn-K", type=int, default=None)
    return result


def apply_deployment_overrides(args, cfg):
    """Apply deployment-only T/K after ArtifactLayout fixed the source ANN run."""
    phase_T = getattr(args, "phase_T", None)
    mtn_T = getattr(args, "mtn_T", None)
    mtn_K = getattr(args, "mtn_K", None)
    neuron = getattr(args, "neuron", None)
    if neuron in {"ann", "gif"} and any(value is not None for value in (phase_T, mtn_T, mtn_K)):
        raise ValueError(f"Deployment T/K overrides do not apply to neuron={neuron}")
    if neuron == "phase" and any(value is not None for value in (mtn_T, mtn_K)):
        raise ValueError("Phase deployment accepts only --phase-T")
    if neuron == "mtn" and phase_T is not None:
        raise ValueError("MTN deployment accepts only --mtn-T/--mtn-K")
    for name, value in (("phase.T", phase_T), ("mtn.T", mtn_T), ("mtn.K", mtn_K)):
        if value is not None and value <= 0:
            raise ValueError(f"{name} deployment override must be a positive integer")
    if phase_T is not None:
        cfg["phase"]["T"] = phase_T
    if mtn_T is not None:
        cfg["mtn"]["T"] = mtn_T
    if mtn_K is not None:
        cfg["mtn"]["K"] = mtn_K
    return cfg


def setup(config_path: str, config_scope: str = "run"):
    cfg = load_config(config_path)
    layout = ArtifactLayout(cfg)

    if config_scope == "task_shared":
        config_dir = layout.shared_task_config_dir
    elif config_scope == "policy_shared":
        config_dir = layout.policy_config_dir
    elif config_scope == "rotated_pre_finetuning":
        config_dir = layout.rotated_pre_finetuning_config_dir
    elif config_scope == "ann_training_calibration":
        config_dir = layout.ann_training_calibration_config_dir
    elif config_scope == "vanilla_analysis_calibration":
        config_dir = layout.vanilla_analysis_calibration_config_dir
    elif config_scope == "post_finetuning_calibration":
        config_dir = layout.post_finetuning_conversion_calibration_config_dir
    elif config_scope == "run":
        config_dir = layout.config_dir
    elif config_scope == "base":
        config_dir = layout.base_config_dir
    else:
        raise ValueError(f"Unknown config scope: {config_scope}")

    if int(os.environ.get("RANK", "0")) == 0:
        layout.write_resolved_config(cfg, config_dir)

    seed = int(cfg["experiment"]["seed"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return cfg, layout
