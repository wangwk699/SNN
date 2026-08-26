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
    description: str, neuron: bool = False, allow_ann: bool = False
) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--config", required=True, help="YAML experiment config")
    if neuron:
        choices = ("ann", "phase", "gif", "mtn") if allow_ann else ("phase", "gif", "mtn")
        result.add_argument("--neuron", required=True, choices=choices)
    return result


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
