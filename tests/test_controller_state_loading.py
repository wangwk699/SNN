import pytest
import torch

from snn2.controller import SiteController
from snn2.calibration import materialize_calibration_states
from snn2.sites import SITE_IDS, SITE_NAMES, site_key
from snn2.temporal_ops import (
    GIF_INTEGER_DECOMPOSITION,
    PHASE_TAU_CALIBRATION,
    PHASE_TAU_EMA_FACTOR,
    SITE_STATE_FORMAT_VERSION,
    TEMPORAL_IMPLEMENTATION_VERSION,
    temporal_policy_metadata,
)


def _header(kind):
    return {
        "state_kind": kind,
        "format_version": SITE_STATE_FORMAT_VERSION,
        "temporal_implementation_version": TEMPORAL_IMPLEMENTATION_VERSION,
    }


def _statistics():
    return {
        "channels": 3,
        "value_min": torch.full((3,), -1.0),
        "value_max": torch.full((3,), 1.0),
        "saliency_row_count": torch.ones(3, dtype=torch.long),
        "saliency_sum": torch.arange(3, dtype=torch.float64),
        "phase_ema_abs_max": torch.ones(3),
        "phase_ema_updates": torch.ones(3, dtype=torch.long),
        "phase_tau_statistic": PHASE_TAU_CALIBRATION,
        "phase_tau_ema_factor": PHASE_TAU_EMA_FACTOR,
        "phase_statistical_view": "spikingllm_identity_input_layout",
        "phase_statistical_view_version": 1,
    }


def _write_bundle(root):
    for index in SITE_IDS:
        directory = root / "layer_000" / f"site_{index:02d}_{SITE_NAMES[index]}"
        directory.mkdir(parents=True)
        torch.save(_statistics(), directory / "statistics.pt")
    global_directory = root / "_global" / "final_rmsnorm"
    global_directory.mkdir(parents=True)
    torch.save(_statistics(), global_directory / "statistics.pt")
    cfg = {
        "calibration": {"group_size": -1, "expected_sites_per_layer": 10},
        "phase": {"T": 2, "base": 2.0, "surrogate_slope": 1.0, "max_spikes": 2},
        "gif": {"base_bits": 4, "add_bits": 1, "low_ratio": 0.5},
        "mtn": {"T": 2, "K": 2, "threshold_factor": 0.75},
    }
    materialize_calibration_states(root, cfg, include_clip=False, expected_num_hidden_layers=1)



def _site_directory(root):
    directory = root / site_key(0, 1)
    directory.mkdir(parents=True)
    return directory


def _phase_state():
    return {
        **_header("phase"),
        "T": 2,
        "base": 2.0,
        "group_size": -1,
        "surrogate_slope": 1.0,
        "max_spikes": 2,
        "tau": torch.tensor([1.0]),
        "v0": torch.tensor([0.125]),
        "tau_calibration": PHASE_TAU_CALIBRATION,
        "tau_ema_factor": PHASE_TAU_EMA_FACTOR,
        "tau_accumulator_dtype": "float32",
        "tau_channel_policy": "spikingllm_flatten_attention_heads_before_channel_ema",
        "tau_reduction_policy": "per_channel_ema_then_global_max",
        "phase_statistical_view": "spikingllm_identity_input_layout",
        "phase_statistical_view_version": 1,
    }


def _gif_state():
    return {
        **_header("gif"),
        "base_bits": 4,
        "add_bits": 1,
        "low_qmin": 0,
        "low_qmax": 15,
        "high_qmin": 0,
        "high_qmax": 30,
        "temporal_steps": 2,
        "per_step_qmin": 0,
        "per_step_qmax": 15,
        "integer_decomposition": GIF_INTEGER_DECOMPOSITION,
        "group_size": -1,
        "low_scale": torch.tensor([0.1]),
        "low_zero": torch.tensor([7.0]),
        "high_scale": torch.tensor([0.05]),
        "high_zero": torch.tensor([13.0]),
        "mask_low": torch.tensor([True, False, True]),
    }


def _mtn_state():
    return {
        **_header("mtn"),
        "T": 2,
        "K": 2,
        "group_size": -1,
        "threshold_factor": 0.75,
        "base_scale": torch.tensor([1.0]),
    }


def _clip_state():
    return {
        **_header("clip"),
        "gif_high_qmax": 30,
        "gif_per_step_qmax": 15,
        "gif_integer_decomposition": GIF_INTEGER_DECOMPOSITION,
        "group_size": -1,
        "lower": torch.tensor([-1.0]),
        "upper": torch.tensor([1.0]),
        "gif_low_range": (torch.tensor([-1.0]), torch.tensor([1.0])),
        "gif_high_range": (torch.tensor([-1.0]), torch.tensor([1.0])),
    }


@pytest.mark.parametrize(
    ("neuron", "state", "expected_steps"),
    [
        ("phase", _phase_state(), 2),
        ("gif", _gif_state(), 2),
        ("mtn", _mtn_state(), 2),
    ],
)
def test_deployment_loads_only_selected_neuron_state(
    tmp_path, neuron, state, expected_steps
):
    _write_bundle(tmp_path)
    controller = SiteController(site_root=tmp_path)

    assert controller.set_deployment(neuron) == expected_steps
    output = controller.apply(0, 1, torch.zeros(expected_steps, 1, 3))

    assert output.shape == (expected_steps, 1, 3)
    assert set(controller._modules[site_key(0, 1)]) == {neuron}


@pytest.mark.parametrize(
    ("mode", "state_name", "state"),
    [
        ("phase", "phase_state.pt", _phase_state()),
        ("gif", "gif_state.pt", _gif_state()),
    ],
)
def test_ann_replacement_requires_common_clip(tmp_path, mode, state_name, state):
    directory = _site_directory(tmp_path)
    torch.save(state, directory / state_name)
    controller = SiteController(mode=mode, site_root=tmp_path)

    with pytest.raises(FileNotFoundError, match="clip_state.pt"):
        controller.apply(0, 1, torch.zeros(1, 3))


def test_ann_phase_still_applies_common_clip(tmp_path):
    directory = _site_directory(tmp_path)
    torch.save(_phase_state(), directory / "phase_state.pt")
    torch.save(_clip_state(), directory / "clip_state.pt")
    controller = SiteController(mode="phase", site_root=tmp_path)

    output = controller.apply(0, 1, torch.tensor([[100.0, 0.0, -100.0]]))

    assert torch.all(output <= 1.0)
    assert torch.all(output >= -1.0)
    assert set(controller._modules[site_key(0, 1)]) == {"phase", "clip"}


def test_ann_gif_still_applies_common_clip(tmp_path):
    directory = _site_directory(tmp_path)
    torch.save(_gif_state(), directory / "gif_state.pt")
    torch.save(_clip_state(), directory / "clip_state.pt")
    controller = SiteController(mode="gif", site_root=tmp_path)

    output = controller.apply(0, 1, torch.tensor([[100.0, 0.0, -100.0]]))

    assert torch.all(output <= 1.0)
    assert torch.all(output >= -1.0)
    assert set(controller._modules[site_key(0, 1)]) == {"gif", "clip"}

def test_deployment_rejects_cross_site_temporal_step_mismatch(tmp_path):
    _write_bundle(tmp_path)
    path = tmp_path / site_key(0, 2) / "phase_state.pt"
    state = torch.load(path, weights_only=False)
    state["T"] = 3
    torch.save(state, path)
    import json
    from snn2.artifacts import sha256_file
    manifest_path = tmp_path / "calibration_state_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sites"][site_key(0, 2)]["state_sha256"]["phase"] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Inconsistent temporal steps"):
        SiteController(site_root=tmp_path).set_deployment("phase")


def test_deployment_rejects_manifest_policy_mismatch(tmp_path):
    import json

    _write_bundle(tmp_path)
    path = tmp_path / "calibration_state_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["prefix_temporal_policy"] = "full_prefix_each_timestep"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible legacy temporal"):
        SiteController(site_root=tmp_path).set_deployment("phase")

def test_final_rmsnorm_phase_is_phase_only_and_not_part_of_sites(tmp_path):
    _write_bundle(tmp_path)
    phase = SiteController(site_root=tmp_path)
    phase.set_deployment("phase")
    value = torch.randn(2, 1, 3)
    output = phase.apply_final_norm_phase(value)
    assert output.shape == value.shape
    assert phase._final_norm_phase is not None
    assert set(phase._modules) == set()

    for neuron in ("gif", "mtn"):
        controller = SiteController(site_root=tmp_path)
        controller.set_deployment(neuron)
        assert controller.apply_final_norm_phase(value) is value
        assert controller._final_norm_phase is None
    assert not (tmp_path / "_global" / "final_rmsnorm" / "clip_state.pt").exists()
