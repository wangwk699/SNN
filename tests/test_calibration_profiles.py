import json

import torch
import pytest

from snn2.calibration import build_site_states, materialize_calibration_states
from snn2.sites import SITE_IDS, SITE_NAMES
from snn2.phase_statistics import (
    PHASE_TAU_ACCUMULATOR_DTYPE, PHASE_TAU_CALIBRATION,
    PHASE_TAU_CHANNEL_POLICY, PHASE_TAU_EMA_FACTOR, PHASE_TAU_REDUCTION_POLICY,
)
from snn2.temporal_ops import SOFTMAX_SITE5_GIF_POLICY, STATISTICS_FORMAT_VERSION


def _statistics(site_index=1):
    if site_index in {2, 3, 4}:
        shape, layout, heads, width, channels = (1, 4), "attention_head", 1, 4, 4
    elif site_index == 5:
        shape, layout, heads, width, channels = (1,), "attention_softmax", 1, None, 1
    else:
        shape, layout, heads, width, channels = (4,), "last_dim", None, None, 4
    saliency_shape = shape
    roles = (
        ("q", "k", "v") if site_index == 1 else
        (("gate", "up") if site_index == 7 else
         (("default",) if site_index in {3, 4, 6, 10} else ()))
    )
    return {
        "format_version": STATISTICS_FORMAT_VERSION, "site_index": site_index,
        "layout_kind": layout, "num_heads": heads, "channels_per_head": width,
        "channels": channels, "value_min": torch.full(shape, -1.0),
        "value_max": torch.full(shape, 1.0),
        "saliency_row_count_by_role": {role: torch.ones(saliency_shape, dtype=torch.long) for role in roles},
        "saliency_sum_by_role": {role: torch.zeros(saliency_shape, dtype=torch.float64 if site_index in {3, 4} else torch.float32) for role in roles},
        "saliency_rule_by_role": {role: ("spikellm_qk_k_fp64" if site_index == 3 else ("spikellm_pv_v_fp64" if site_index == 4 else "spikellm_linear_fp32")) for role in roles},
        "saliency_accumulator_dtype_by_role": {role: ("float64" if site_index in {3, 4} else "float32") for role in roles},
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
    site5_gif = states["gif"]
    assert site5_gif["gif_policy"] == SOFTMAX_SITE5_GIF_POLICY
    assert site5_gif["quantization_applied"] is False
    assert site5_gif["temporal_policy"] == "identity"
    for key in (
        "range_min", "range_max", "quantization_bits", "qmin", "qmax",
        "scale", "zero_point",
    ):
        assert key not in site5_gif


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
            elif isinstance(left[key], dict):
                assert left[key].keys() == right[key].keys()
                for nested in left[key]:
                    if isinstance(left[key][nested], torch.Tensor):
                        torch.testing.assert_close(left[key][nested], right[key][nested])
                    else:
                        assert left[key][nested] == right[key][nested]
            elif isinstance(left[key], tuple):
                for left_value, right_value in zip(left[key], right[key]):
                    torch.testing.assert_close(left_value, right_value)
            else:
                assert left[key] == right[key]


def test_site2_all_low_state_has_strict_low_only_contract():
    state = build_site_states(_statistics(2), _cfg(), include_clip=False)["gif"]
    assert state["gif_policy"] == "all_low_static_qmax15"
    assert state["base_bits"] == 4
    assert state["add_bits"] == 1
    assert state["low_qmin"] == 0
    assert state["low_qmax"] == 15
    assert state["temporal_steps"] == 2
    assert state["per_step_qmin"] == 0
    assert state["per_step_qmax"] == 15
    assert state["quantization_path"] == "low_only"
    assert state["quantization_applied"] is True
    assert state["saliency_enabled"] is False
    assert state["temporal_policy"] == "low_at_t0_zero_at_t1"
    assert state["parameter_layout"] == "attention_head_grouped"
    forbidden = {
        "high_qmin", "high_qmax", "high_scale", "high_zero",
        "mask_low", "mask_low_by_role", "mask_roles", "integer_decomposition",
    }
    assert not forbidden.intersection(state)


def test_calibration_manifest_records_saliency_and_mask_provenance(tmp_path):
    _write_statistics(tmp_path)
    manifest = materialize_calibration_states(
        tmp_path, _cfg(), include_clip=False, expected_num_hidden_layers=1
    )
    assert manifest["gif_saliency_selection_policy"] == "spikellm_global_per_channel_threshold_leq"
    assert manifest["gif_saliency_tie_policy"] == "mask_low_equals_score_le_threshold"
    assert manifest["gif_linear_saliency_dtype"] == "float32"
    assert manifest["gif_matmul_saliency_dtype"] == "float64"

    sites = manifest["sites"]
    by_index = {entry["site_index"]: entry for entry in sites.values()}
    assert by_index[1]["saliency_roles"] == ["q", "k", "v"]
    assert by_index[1]["saliency_rule_by_role"] == {
        "q": "spikellm_linear_fp32", "k": "spikellm_linear_fp32",
        "v": "spikellm_linear_fp32",
    }
    assert by_index[1]["saliency_accumulator_dtype_by_role"] == {
        "q": "float32", "k": "float32", "v": "float32",
    }
    assert by_index[1]["gif_mask_policy"] == "multi_role"
    assert by_index[1]["gif_mask_roles"] == ["q", "k", "v"]

    assert by_index[3]["saliency_rule_by_role"] == {"default": "spikellm_qk_k_fp64"}
    assert by_index[3]["saliency_accumulator_dtype_by_role"] == {"default": "float64"}
    assert by_index[3]["gif_mask_policy"] == "single"
    assert by_index[3]["gif_mask_roles"] == []
    assert by_index[4]["saliency_rule_by_role"] == {"default": "spikellm_pv_v_fp64"}
    assert by_index[6]["saliency_accumulator_dtype_by_role"] == {"default": "float32"}
    assert by_index[7]["saliency_roles"] == ["gate", "up"]
    assert by_index[7]["gif_mask_roles"] == ["gate", "up"]

    for site in (2, 5, 8, 9):
        assert by_index[site]["saliency_enabled"] is False
        assert by_index[site]["saliency_roles"] == []
        assert by_index[site]["saliency_rule_by_role"] == {}
        assert by_index[site]["saliency_accumulator_dtype_by_role"] == {}
        assert by_index[site]["gif_mask_policy"] is None
        assert by_index[site]["gif_mask_roles"] == []
