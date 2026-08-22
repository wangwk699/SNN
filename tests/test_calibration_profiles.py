import json

import torch

from snn2.calibration import build_site_states, materialize_calibration_states
from snn2.sites import SITE_IDS, SITE_NAMES


def _statistics():
    return {
        "channels": 4,
        "value_min": torch.full((4,), -1.0),
        "value_max": torch.full((4,), 1.0),
        "saliency_row_count": torch.ones(4, dtype=torch.long),
        "saliency_sum": torch.arange(4, dtype=torch.float64),
    }


def _cfg():
    return {
        "calibration": {"group_size": -1, "expected_sites_per_layer": 10},
        "phase": {"T": 4, "base": 2.0, "surrogate_slope": 4.0, "max_spikes": 4},
        "gif": {"base_bits": 4, "add_bits": 1, "low_ratio": 0.5},
        "mtn": {"T": 4, "K": 6, "threshold_factor": 0.75},
    }


def _write_statistics(root):
    directories = []
    for index in SITE_IDS:
        directory = root / "layer_000" / f"site_{index:02d}_{SITE_NAMES[index]}"
        directory.mkdir(parents=True)
        torch.save(_statistics(), directory / "statistics.pt")
        directories.append(directory)
    return directories


def test_build_site_states_with_common_clip():
    states = build_site_states(_statistics(), _cfg(), include_clip=True)

    assert set(states) == {"phase", "gif", "mtn", "clip"}


def test_build_site_states_without_common_clip():
    states = build_site_states(_statistics(), _cfg(), include_clip=False)

    assert set(states) == {"phase", "gif", "mtn"}


def test_conversion_materialization_keeps_temporal_common_clip(tmp_path):
    directories = _write_statistics(tmp_path)
    for directory in directories:
        torch.save({}, directory / "clip_state.pt")

    manifest = materialize_calibration_states(
        tmp_path,
        _cfg(),
        {
            "state_profile": "snn_conversion_with_common_clip",
            "common_clip_required": True,
        },
        include_clip=True,
    )

    assert manifest["state_profile"] == "snn_conversion_with_common_clip"
    assert all((directory / "clip_state.pt").exists() for directory in directories)
    summary = json.loads(
        (directories[0] / "calibration_summary.json").read_text(encoding="utf-8")
    )
    assert summary["clip_state_present"] is True
    assert summary["clip_valid"] is True


def test_ann_training_materialization_keeps_common_clip(tmp_path):
    directories = _write_statistics(tmp_path)

    manifest = materialize_calibration_states(
        tmp_path,
        _cfg(),
        {
            "state_profile": "ann_training_with_common_clip",
            "common_clip_required": True,
        },
        include_clip=True,
    )

    assert manifest["state_profile"] == "ann_training_with_common_clip"
    assert all((directory / "clip_state.pt").exists() for directory in directories)
    summary = json.loads(
        (directories[0] / "calibration_summary.json").read_text(encoding="utf-8")
    )
    assert summary["clip_state_present"] is True
    assert summary["clip_valid"] is True
