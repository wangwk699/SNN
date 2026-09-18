import copy
from pathlib import Path

import pytest
import torch
import yaml

from snn2.artifacts import ArtifactLayout, calibration_trajectory_dirname
from snn2.config import resolve_config, validate_config
from snn2.controller import SiteController
from snn2.temporal_ops import from_temporal


ROOT = Path(__file__).resolve().parents[1]


def _raw():
    value = yaml.safe_load((ROOT / "configs" / "experiment_matrix.yaml").read_text())["experiments"][0]["config"]
    value["experiment"]["ann_mode"] = "gif_aware"
    return value


def test_previous_layers_snn_defaults_and_path_isolation():
    paths = set()
    for phase in (False, True):
        for gif in (False, True):
            for mtn in (False, True):
                cfg = resolve_config(copy.deepcopy(_raw()))
                cfg["calibration"].update({
                    "phase_previous_layers_snn": phase,
                    "gif_previous_layers_snn": gif,
                    "mtn_previous_layers_snn": mtn,
                })
                validate_config(cfg)
                paths.add(str(ArtifactLayout(cfg).ann_training_calibration_dir))
                assert calibration_trajectory_dirname(cfg).count("previous_layers_snn") == 3
    assert len(paths) == 8


@pytest.mark.parametrize("value", ["true", 1, None])
def test_previous_layers_snn_requires_yaml_boolean(value):
    cfg = resolve_config(copy.deepcopy(_raw()))
    cfg["calibration"]["gif_previous_layers_snn"] = value
    with pytest.raises(ValueError, match="gif_previous_layers_snn"):
        validate_config(cfg)


def test_sequential_collection_uses_one_logical_activation_per_sample():
    controller = SiteController(mode="collect", phase_T=2, phase_base=2.0, mtn_T=2, mtn_K=2, mtn_threshold_factor=0.75)
    controller.begin_sequential_calibration("phase", 0)
    temporal = torch.tensor([[[[1.0, 2.0]]], [[[3.0, 4.0]]]])
    controller.apply(0, 1, from_temporal(temporal))
    stored = next(iter(controller.statistics.items.values()))
    assert torch.equal(stored.value_max, torch.tensor([4.0, 6.0], dtype=torch.float64))
    assert int(stored.phase_ema_updates.max()) == 1
