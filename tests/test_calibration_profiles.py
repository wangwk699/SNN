import json

import torch
import pytest

from snn2.calibration import build_site_states, materialize_calibration_states
from snn2.sites import SITE_IDS, SITE_NAMES
from snn2.phase_statistics import (
    PHASE_TAU_ACCUMULATOR_DTYPE, PHASE_TAU_CALIBRATION,
    PHASE_TAU_CHANNEL_POLICY, PHASE_TAU_EMA_FACTOR, PHASE_TAU_REDUCTION_POLICY,
)
from snn2.temporal_ops import STATISTICS_FORMAT_VERSION


def _statistics(site_index=1):
    if site_index in {2, 3, 4, 6}:
        shape, layout, heads, width, channels = (1, 4), "attention_head", 1, 4, 4
    elif site_index == 5:
        shape, layout, heads, width, channels = (1,), "attention_softmax", 1, None, 1
    else:
        shape, layout, heads, width, channels = (4,), "last_dim", None, None, 4
    saliency_shape = (0,) if site_index == 5 else shape
    return {
        "format_version": STATISTICS_FORMAT_VERSION, "site_index": site_index,
        "layout_kind": layout, "num_heads": heads, "channels_per_head": width,
        "channels": channels, "value_min": torch.full(shape, -1.0),
        "value_max": torch.full(shape, 1.0),
        "saliency_row_count": torch.ones(saliency_shape, dtype=torch.long),
        "saliency_sum": torch.arange(torch.tensor(saliency_shape).prod().item(), dtype=torch.float64).reshape(saliency_shape),
        "phase_ema_abs_max": torch.ones(shape), "phase_ema_updates": torch.ones(shape, dtype=torch.long),
        "phase_tau_calibration": PHASE_TAU_CALIBRATION,
        "phase_tau_ema_factor": PHASE_TAU_EMA_FACTOR,
        "phase_tau_accumulator_dtype": PHASE_TAU_ACCUMULATOR_DTYPE,
        "phase_tau_channel_policy": PHASE_TAU_CHANNEL_POLICY,
        "phase_tau_reduction_policy": PHASE_TAU_REDUCTION_POLICY,
    }


def _cfg():
    return {
        "calibration": {"group_size": -1, "expected_sites_per_layer": 10},
        "phase": {"T": 4, "base": 2.0, "surrogate_slope": 1.0, "max_spikes": 4},
        "gif": {"base_bits": 4, "add_bits": 1, "low_ratio": 0.5},
        "mtn": {"T": 4, "K": 6, "threshold_factor": 0.75},
    }


def _write_statistics(root):
    directories = []
    for index in SITE_IDS:
        directory = root / "layer_000" / f"site_{index:02d}_{SITE_NAMES[index]}"
        directory.mkdir(parents=True)
        torch.save(_statistics(index), directory / "statistics.pt")
        directories.append(directory)
    global_directory = root / "_global" / "final_rmsnorm"
    global_directory.mkdir(parents=True)
    torch.save(_statistics(None), global_directory / "statistics.pt")
    return directories


def test_build_site_states_with_common_clip():
    states = build_site_states(_statistics(), _cfg(), include_clip=True)

    assert set(states) == {"phase", "gif", "mtn", "clip"}


def test_build_site_states_without_common_clip():
    states = build_site_states(_statistics(), _cfg(), include_clip=False)

    assert set(states) == {"phase", "gif", "mtn"}
    assert states["phase"]["tau"].dtype == torch.float32
    assert states["phase"]["tau"].numel() == 1
    assert states["phase"]["group_size"] == 4
    assert states["phase"]["tau_accumulator_dtype"] == "float32"
    assert states["phase"]["tau_reduction_policy"] == "within_group_max_after_channel_ema"


def test_non_divisible_ordinary_group_fails_but_site5_ignores_global_group():
    cfg = _cfg()
    cfg["calibration"]["group_size"] = 3
    with pytest.raises(ValueError, match="not divisible|divisible"):
        build_site_states(_statistics(1), cfg, include_clip=False)
    states = build_site_states(_statistics(5), cfg, include_clip=True)
    assert set(states) == {"phase", "gif", "mtn"}
    assert states["phase"]["tau"].shape == (1, 1)
    assert states["mtn"]["base_scale"].shape == (1, 1)
    assert states["gif"]["gif_policy"] == "softmax_fixed_range_u16"
    assert states["gif"]["qmax"] == 65535


def test_conversion_materialization_removes_common_clip(tmp_path):
    directories = _write_statistics(tmp_path)
    for directory in directories:
        torch.save({}, directory / "clip_state.pt")

    manifest = materialize_calibration_states(
        tmp_path,
        _cfg(),
        {
            "state_profile": "snn_conversion_without_clip",
            "common_clip_required": False,
        },
        include_clip=False,
        expected_num_hidden_layers=1,
    )

    assert manifest["state_profile"] == "snn_conversion_without_clip"
    assert all(not (directory / "clip_state.pt").exists() for directory in directories)
    summary = json.loads(
        (directories[0] / "calibration_summary.json").read_text(encoding="utf-8")
    )
    assert summary["clip_state_present"] is False
    assert "clip_valid" not in summary


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
        expected_num_hidden_layers=1,
    )

    assert manifest["state_profile"] == "ann_training_with_common_clip"
    assert (tmp_path / "_global" / "final_rmsnorm" / "phase_state.pt").exists()
    assert manifest["global_states"]["final_rmsnorm"]["phase_state_sha256"]
    assert all((directory / "clip_state.pt").exists() == (index != 5) for index, directory in zip(SITE_IDS, directories))
    summary = json.loads(
        (directories[0] / "calibration_summary.json").read_text(encoding="utf-8")
    )
    assert summary["clip_state_present"] is True
    assert summary["clip_valid"] is True


def test_ann_training_calibration_is_identical_for_common_clip_switch(tmp_path):
    roots = [tmp_path / "enabled", tmp_path / "disabled"]
    states_by_variant = []
    for root, enabled in zip(roots, (True, False)):
        directories = _write_statistics(root)
        cfg = _cfg()
        cfg["replacement"] = {"common_clip_enabled": enabled}
        manifest = materialize_calibration_states(
            root,
            cfg,
            {
                "common_clip_required": True,
                "common_clip_generated": True,
                "common_clip_application_control": "replacement.common_clip_enabled",
            },
            include_clip=True,
            expected_num_hidden_layers=1,
        )
        assert manifest["common_clip_generated"] is True
        assert all(
            (directory / "clip_state.pt").exists() == (index != 5)
            for index, directory in zip(SITE_IDS, directories)
        )
        states_by_variant.append(
            {
                name: torch.load(
                    directories[0] / f"{name}_state.pt", weights_only=False
                )
                for name in ("phase", "gif", "mtn", "clip")
            }
        )
    for name in states_by_variant[0]:
        left, right = states_by_variant[0][name], states_by_variant[1][name]
        assert left.keys() == right.keys()
        for key in left:
            if isinstance(left[key], torch.Tensor):
                torch.testing.assert_close(left[key], right[key])
            elif isinstance(left[key], tuple):
                for left_value, right_value in zip(left[key], right[key]):
                    torch.testing.assert_close(left_value, right_value)
            else:
                assert left[key] == right[key]
