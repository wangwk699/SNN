import json

import torch
import pytest

from snn2.calibration import (
    build_clip_state, build_site_states, materialize_calibration_states,
    materialize_clip_profile,
)
from snn2.sites import SITE_IDS, SITE_NAMES
from snn2.state_validation import validate_clip_profile, validate_site_state_bundle
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
        "calibration": {"group_size": -1, "num_samples": 128, "expected_sites_per_layer": 10},
        "phase": {"T": 4, "base": 2.0, "surrogate_slope": 1.0},
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


def test_build_site_states_is_always_stage_a_only():
    states = build_site_states(_statistics(), _cfg())

    assert set(states) == {"phase", "gif", "mtn"}

def test_build_site_states_without_common_clip():
    states = build_site_states(_statistics(), _cfg())

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
        build_site_states(_statistics(1), cfg)
    states = build_site_states(_statistics(5), cfg)
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

def test_stage_a_materialization_is_clip_free_and_runtime_independent(tmp_path):
    directories = _write_statistics(tmp_path)
    cfg = _cfg()
    cfg["phase"]["T"] = 2
    cfg["mtn"].update({"T": 2, "K": 4})
    manifest = materialize_calibration_states(
        tmp_path, cfg, expected_num_hidden_layers=1
    )
    assert manifest["calibration_architecture"] == "two_stage_A_common_B_clip_profiles"
    assert all(not (directory / "clip_state.pt").exists() for directory in directories)
    assert manifest["calibration_phase"] == "A"
    assert all(not (directory / "calibration_summary.json").exists() for directory in directories)
    phase = torch.load(directories[0] / "phase_state.pt", weights_only=False)
    mtn = torch.load(directories[0] / "mtn_state.pt", weights_only=False)
    assert not {"T", "base", "max_spikes", "v0", "surrogate_slope"}.intersection(phase)
    assert not {"T", "K", "threshold_factor"}.intersection(mtn)
    assert set(manifest["sites"][directories[0].relative_to(tmp_path).as_posix()]["state_sha256"]) == {"phase", "gif", "mtn"}


def test_two_stage_b_profiles_reuse_unchanged_stage_a(tmp_path):
    directories = _write_statistics(tmp_path / "sites")
    cfg = _cfg()
    materialize_calibration_states(
        tmp_path / "sites", cfg, expected_num_hidden_layers=1
    )
    stage_a_hashes = {
        path.relative_to(tmp_path / "sites").as_posix(): path.read_bytes()
        for path in (tmp_path / "sites").glob("**/*_state.pt")
    }
    first = tmp_path / "clip_profiles" / "phase_T_2_mtn_T_2"
    cfg["phase"]["T"], cfg["mtn"]["T"] = 2, 2
    materialize_clip_profile(tmp_path / "sites", first, cfg)
    second = tmp_path / "clip_profiles" / "phase_T_4_mtn_T_8"
    cfg["phase"]["T"], cfg["mtn"]["T"] = 4, 8
    materialize_clip_profile(tmp_path / "sites", second, cfg)
    assert (first / "clip_profile_manifest.json").exists()
    assert (second / "clip_profile_manifest.json").exists()
    assert all(
        path.read_bytes() == stage_a_hashes[path.relative_to(tmp_path / "sites").as_posix()]
        for path in (tmp_path / "sites").glob("**/*_state.pt")
    )
    for index, directory in zip(SITE_IDS, directories):
        relative = directory.relative_to(tmp_path / "sites")
        assert (first / relative / "clip_state.pt").exists() == (index != 5)
        assert (second / relative / "clip_state.pt").exists() == (index != 5)


def test_mask_aware_role_specific_clip_classifies_site1_roles():
    cfg = _cfg()
    cfg["calibration"]["group_size"] = -1
    states = build_site_states(_statistics(1), cfg)
    states["gif"]["mask_low_by_role"] = {
        "q": torch.ones(4, dtype=torch.bool),
        "k": torch.zeros(4, dtype=torch.bool),
        "v": torch.tensor([True, False, True, False]),
    }
    clip = build_clip_state(
        states["phase"], states["gif"], states["mtn"], phase_T=4, mtn_T=8
    )
    assert clip["clip_role_policy"] == "role_specific"
    assert clip["clip_roles"] == ["q", "k", "v"]
    assert clip["gif_group_classification_by_role"]["q"].item() == 0
    assert clip["gif_group_classification_by_role"]["k"].item() == 1
    assert clip["gif_group_classification_by_role"]["v"].item() == 2


def test_site2_all_low_state_has_strict_low_only_contract():
    state = build_site_states(_statistics(2), _cfg())["gif"]
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
        tmp_path, _cfg(), expected_num_hidden_layers=1
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


def test_stage_a_validator_rejects_runtime_field_in_manifest(tmp_path):
    _write_statistics(tmp_path)
    materialize_calibration_states(
        tmp_path, _cfg(), expected_num_hidden_layers=1
    )
    path = tmp_path / "calibration_state_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["phase_T"] = 4
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime-dependent fields"):
        validate_site_state_bundle(tmp_path, clip_policy="forbid_all")


def test_clip_profile_validator_rejects_tampered_site_summary(tmp_path):
    cfg = _cfg()
    site_root = tmp_path / "sites"
    _write_statistics(site_root)
    materialize_calibration_states(
        site_root, cfg, expected_num_hidden_layers=1
    )
    profile_root = tmp_path / "phase_T_4_mtn_T_4"
    materialize_clip_profile(site_root, profile_root, cfg)
    summary_path = next(profile_root.glob("layer_*/site_*/calibration_summary.json"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["clip_rule"] = "tampered"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest/site summary mismatch"):
        validate_clip_profile(
            site_root,
            profile_root,
            phase_T=4,
            mtn_T=4,
            group_size=-1,
            num_samples=128,
        )
