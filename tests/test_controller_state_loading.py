import pytest
import torch

from snn2.controller import SiteController
from snn2.sites import site_key


def _site_directory(root):
    directory = root / site_key(0, 1)
    directory.mkdir(parents=True)
    return directory


def _phase_state():
    return {
        "T": 2,
        "base": 2.0,
        "group_size": -1,
        "surrogate_slope": 4.0,
        "max_spikes": 2,
        "tau": torch.tensor([1.0]),
        "v0": torch.tensor([0.125]),
    }


def _gif_state():
    return {
        "base_bits": 4,
        "add_bits": 1,
        "group_size": -1,
        "low_scale": torch.tensor([0.1]),
        "low_zero": torch.tensor([7.0]),
        "high_scale": torch.tensor([0.05]),
        "high_zero": torch.tensor([13.0]),
        "mask_low": torch.tensor([True, False, True]),
    }


def _mtn_state():
    return {
        "T": 2,
        "K": 2,
        "group_size": -1,
        "threshold_factor": 0.75,
        "base_scale": torch.tensor([1.0]),
    }


def _clip_state():
    return {
        "group_size": -1,
        "lower": torch.tensor([-1.0]),
        "upper": torch.tensor([1.0]),
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
    directory = _site_directory(tmp_path)
    torch.save(state, directory / f"{neuron}_state.pt")
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


def test_ann_gif_still_applies_common_clip(tmp_path):
    directory = _site_directory(tmp_path)
    torch.save(_gif_state(), directory / "gif_state.pt")
    torch.save(_clip_state(), directory / "clip_state.pt")
    controller = SiteController(mode="gif", site_root=tmp_path)

    output = controller.apply(0, 1, torch.tensor([[100.0, 0.0, -100.0]]))

    assert torch.all(output <= 1.0)
    assert torch.all(output >= -1.0)
    assert set(controller._modules[site_key(0, 1)]) == {"gif", "clip"}
