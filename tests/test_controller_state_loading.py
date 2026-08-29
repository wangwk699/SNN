import pytest
import torch

from snn2.controller import SiteController
from snn2.calibration import materialize_calibration_states
from snn2.neurons import SoftmaxIdentityGIF
from snn2.sites import SITE_IDS, SITE_NAMES, site_key
from snn2.temporal_ops import (
    GIF_INTEGER_DECOMPOSITION,
    PHASE_TAU_CALIBRATION,
    PHASE_TAU_EMA_FACTOR,
    SITE_STATE_FORMAT_VERSION,
    TEMPORAL_IMPLEMENTATION_VERSION,
    temporal_policy_metadata,
    STATISTICS_FORMAT_VERSION,
)
from snn2.phase_statistics import (
    PHASE_TAU_ACCUMULATOR_DTYPE, PHASE_TAU_CHANNEL_POLICY,
    PHASE_TAU_REDUCTION_POLICY,
)
from snn2.state_validation import validate_site_state_bundle


def _header(kind):
    return {
        "state_kind": kind,
        "format_version": SITE_STATE_FORMAT_VERSION,
        "temporal_implementation_version": TEMPORAL_IMPLEMENTATION_VERSION,
    }


def _statistics(site_index=1):
    if site_index in {2, 3, 4}:
        shape, layout, heads, width, channels = (1, 3), "attention_head", 1, 3, 3
    elif site_index == 5:
        shape, layout, heads, width, channels = (1,), "attention_softmax", 1, None, 1
    else:
        shape, layout, heads, width, channels = (3,), "last_dim", None, None, 3
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
        "saliency_rule_by_role": {role: ("spikellm_matmul_fp64" if site_index in {3, 4} else "spikellm_linear_fp32") for role in roles},
        "saliency_accumulator_dtype_by_role": {role: ("float64" if site_index in {3, 4} else "float32") for role in roles},
        "phase_ema_abs_max": torch.ones(shape), "phase_ema_updates": torch.ones(shape, dtype=torch.long),
        "phase_tau_calibration": PHASE_TAU_CALIBRATION,
        "phase_tau_ema_factor": PHASE_TAU_EMA_FACTOR,
        "phase_tau_accumulator_dtype": PHASE_TAU_ACCUMULATOR_DTYPE,
        "phase_tau_channel_policy": PHASE_TAU_CHANNEL_POLICY,
        "phase_tau_reduction_policy": PHASE_TAU_REDUCTION_POLICY,
    }


def _write_bundle(root, *, include_clip=False):
    for index in SITE_IDS:
        directory = root / "layer_000" / f"site_{index:02d}_{SITE_NAMES[index]}"
        directory.mkdir(parents=True)
        torch.save(_statistics(index), directory / "statistics.pt")
    global_directory = root / "_global" / "final_rmsnorm"
    global_directory.mkdir(parents=True)
    torch.save(_statistics(None), global_directory / "statistics.pt")
    cfg = {
        "calibration": {"group_size": -1, "expected_sites_per_layer": 10},
        "phase": {"T": 2, "base": 2.0, "surrogate_slope": 1.0, "max_spikes": 2},
        "gif": {"base_bits": 4, "add_bits": 1, "low_ratio": 0.5},
        "mtn": {"T": 2, "K": 2, "threshold_factor": 0.75},
    }
    materialize_calibration_states(root, cfg, include_clip=include_clip, expected_num_hidden_layers=1)


def test_site5_common_clip_uses_identity_gif_without_loading_clipper(tmp_path):
    _write_bundle(tmp_path, include_clip=True)
    site5 = tmp_path / "layer_000" / f"site_05_{SITE_NAMES[5]}"
    assert not (site5 / "clip_state.pt").exists()
    controller = SiteController(mode="gif", site_root=tmp_path, common_clip_enabled=True)
    x = torch.rand(1, 1, 2, 3)
    output = controller.apply(0, 5, x)
    assert output is x
    assert torch.equal(output, x)
    assert set(controller._modules[site_key(0, 5)]) == {"gif"}
    assert isinstance(
        controller._modules[site_key(0, 5)]["gif"], SoftmaxIdentityGIF
    )


def test_deploy_gif_site5_is_exact_identity(tmp_path):
    _write_bundle(tmp_path)
    controller = SiteController(site_root=tmp_path)
    controller.set_deployment("gif", clip_bundle_policy="forbid_all")
    incoming = torch.randn(2, 1, 1, 2, 3)
    output = controller.apply(0, 5, incoming.reshape(2, 1, 2, 3))
    assert torch.equal(output, incoming.reshape(2, 1, 2, 3))
    assert set(controller._modules[site_key(0, 5)]) == {"gif"}



def _site_directory(root):
    directory = root / site_key(0, 1)
    directory.mkdir(parents=True)
    return directory


def _phase_state():
    return {
        **_header("phase"),
        "T": 2,
        "base": 2.0,
        "parameter_layout": "last_dim_grouped", "configured_group_size": -1,
        "group_size": 3, "num_heads": None, "channels_per_head": 3,
        "groups_per_head": 1,
        "max_spikes": 2,
        "tau": torch.tensor([1.0]),
        "v0": torch.tensor([0.125]),
        "tau_calibration": PHASE_TAU_CALIBRATION,
        "tau_ema_factor": PHASE_TAU_EMA_FACTOR,
        "tau_accumulator_dtype": "float32",
        "tau_channel_policy": PHASE_TAU_CHANNEL_POLICY,
        "tau_reduction_policy": PHASE_TAU_REDUCTION_POLICY,
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
        "gif_policy": "ordinary_salient_static_qmax30",
        "parameter_layout": "last_dim_grouped", "configured_group_size": -1,
        "group_size": 3, "num_heads": None, "channels_per_head": 3,
        "groups_per_head": 1,
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
        "parameter_layout": "last_dim_grouped", "configured_group_size": -1,
        "group_size": 3, "num_heads": None, "channels_per_head": 3,
        "groups_per_head": 1,
        "threshold_factor": 0.75,
        "base_scale": torch.tensor([1.0]),
    }


def _clip_state():
    return {
        **_header("clip"),
        "ordinary_gif_high_qmax": 30,
        "ordinary_gif_per_step_qmax": 15,
        "gif_integer_decomposition": GIF_INTEGER_DECOMPOSITION,
        "parameter_layout": "last_dim_grouped", "configured_group_size": -1,
        "group_size": 3, "num_heads": None, "channels_per_head": 3,
        "groups_per_head": 1,
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
    _write_bundle(tmp_path, include_clip=True)
    controller = SiteController(site_root=tmp_path)

    assert controller.set_deployment(
        neuron, clip_bundle_policy="allow_eligible"
    ) == expected_steps
    output = controller.apply(
        0, 1, torch.zeros(expected_steps, 1, 3),
        **({"gif_role": "q"} if neuron == "gif" else {}),
    )

    assert output.shape == (expected_steps, 1, 3)
    assert set(controller._modules[site_key(0, 1)]) == {neuron}


def test_allow_eligible_still_rejects_site5_clip(tmp_path):
    _write_bundle(tmp_path, include_clip=True)
    torch.save(_clip_state(), tmp_path / site_key(0, 5) / "clip_state.pt")

    with pytest.raises(ValueError, match="Site 5 permanently forbids"):
        SiteController(site_root=tmp_path).set_deployment(
            "phase", clip_bundle_policy="allow_eligible"
        )


def test_require_eligible_rejects_missing_ordinary_clip(tmp_path):
    _write_bundle(tmp_path, include_clip=True)
    (tmp_path / site_key(0, 1) / "clip_state.pt").unlink()

    with pytest.raises((FileNotFoundError, ValueError), match="clip_state.pt"):
        validate_site_state_bundle(
            tmp_path, clip_policy="require_eligible"
        )


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
    controller = SiteController(
        mode=mode,
        site_root=tmp_path,
        common_clip_enabled=True,
        phase_surrogate_slope=1.0 if mode == "phase" else None,
    )

    with pytest.raises(FileNotFoundError, match="clip_state.pt"):
        controller.apply(0, 1, torch.zeros(1, 3))


def test_ann_phase_still_applies_common_clip(tmp_path):
    directory = _site_directory(tmp_path)
    torch.save(_phase_state(), directory / "phase_state.pt")
    torch.save(_clip_state(), directory / "clip_state.pt")
    controller = SiteController(
        mode="phase",
        site_root=tmp_path,
        common_clip_enabled=True,
        phase_surrogate_slope=1.0,
    )

    output = controller.apply(0, 1, torch.tensor([[100.0, 0.0, -100.0]]))

    assert torch.all(output <= 1.0)
    assert torch.all(output >= -1.0)
    assert set(controller._modules[site_key(0, 1)]) == {"phase", "clip"}


def test_ann_phase_uses_runtime_surrogate_slope_override(tmp_path):
    directory = _site_directory(tmp_path)
    torch.save(_phase_state(), directory / "phase_state.pt")
    controller = SiteController(
        mode="phase",
        site_root=tmp_path,
        common_clip_enabled=False,
        phase_surrogate_slope=2.0,
    )

    controller.apply(0, 1, torch.tensor([[0.5, 0.5, 0.5]]))

    module = controller._modules[site_key(0, 1)]["phase"]
    assert module.slope == 2.0
    assert "surrogate_slope" not in torch.load(
        directory / "phase_state.pt", weights_only=False
    )


def test_phase_rejects_legacy_state_with_surrogate_slope():
    state = _phase_state()
    state["surrogate_slope"] = 1.0
    with pytest.raises(ValueError, match="Legacy Phase state"):
        from snn2.neurons import PhaseSurrogate

        PhaseSurrogate(state, surrogate_slope=2.0)


def test_ann_phase_controller_requires_explicit_surrogate_slope(tmp_path):
    with pytest.raises(ValueError, match="requires an explicit"):
        SiteController(mode="phase", site_root=tmp_path)


def test_ann_gif_still_applies_common_clip(tmp_path):
    directory = _site_directory(tmp_path)
    torch.save(_gif_state(), directory / "gif_state.pt")
    torch.save(_clip_state(), directory / "clip_state.pt")
    controller = SiteController(
        mode="gif", site_root=tmp_path, common_clip_enabled=True
    )

    output = controller.apply(0, 1, torch.tensor([[100.0, 0.0, -100.0]]))

    assert torch.all(output <= 1.0)
    assert torch.all(output >= -1.0)
    assert set(controller._modules[site_key(0, 1)]) == {"gif", "clip"}


@pytest.mark.parametrize(
    ("mode", "state_name", "state", "neuron_name"),
    [
        ("phase", "phase_state.pt", _phase_state(), "phase"),
        ("gif", "gif_state.pt", _gif_state(), "gif"),
    ],
)
def test_ann_replacement_can_disable_common_clip(
    tmp_path, mode, state_name, state, neuron_name
):
    directory = _site_directory(tmp_path)
    torch.save(state, directory / state_name)
    controller = SiteController(
        mode=mode,
        site_root=tmp_path,
        common_clip_enabled=False,
        phase_surrogate_slope=1.0 if mode == "phase" else None,
    )
    output = controller.apply(0, 1, torch.tensor([[2.0, 0.0, -2.0]]))
    direct = controller._modules[site_key(0, 1)][neuron_name](
        torch.tensor([[2.0, 0.0, -2.0]])
    )
    torch.testing.assert_close(output, direct)
    assert set(controller._modules[site_key(0, 1)]) == {neuron_name}


def test_deployment_rejects_common_clip_switch(tmp_path):
    with pytest.raises(ValueError, match="only applies"):
        SiteController(
            mode="deploy_phase", site_root=tmp_path, common_clip_enabled=True
        )
    controller = SiteController(
        mode="phase",
        site_root=tmp_path,
        common_clip_enabled=True,
        phase_surrogate_slope=1.0,
    )
    with pytest.raises(ValueError, match="cannot enable"):
        controller.set_deployment("phase", clip_bundle_policy="forbid_all")

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
        SiteController(site_root=tmp_path).set_deployment(
            "phase", clip_bundle_policy="forbid_all"
        )


def test_deployment_rejects_manifest_policy_mismatch(tmp_path):
    import json

    _write_bundle(tmp_path)
    path = tmp_path / "calibration_state_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["prefix_temporal_policy"] = "full_prefix_each_timestep"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible legacy temporal"):
        SiteController(site_root=tmp_path).set_deployment(
            "phase", clip_bundle_policy="forbid_all"
        )

def test_final_rmsnorm_phase_is_phase_only_and_not_part_of_sites(tmp_path):
    _write_bundle(tmp_path)
    phase = SiteController(site_root=tmp_path)
    phase.set_deployment("phase", clip_bundle_policy="forbid_all")
    value = torch.randn(2, 1, 3)
    output = phase.apply_final_norm_phase(value)
    assert output.shape == value.shape
    assert phase._final_norm_phase is not None
    assert set(phase._modules) == set()

    for neuron in ("gif", "mtn"):
        controller = SiteController(site_root=tmp_path)
        controller.set_deployment(neuron, clip_bundle_policy="forbid_all")
        assert controller.apply_final_norm_phase(value) is value
        assert controller._final_norm_phase is None
    assert not (tmp_path / "_global" / "final_rmsnorm" / "clip_state.pt").exists()


def test_gif_identity_sites_skip_clip_while_phase_uses_it(tmp_path):
    _write_bundle(tmp_path, include_clip=True)
    x = torch.full((1, 2, 3), 100.0)
    gif = SiteController(mode="gif", site_root=tmp_path, common_clip_enabled=True)
    assert torch.equal(gif.apply(0, 8, x), x)
    assert torch.equal(gif.apply(0, 9, x), x)

    phase = SiteController(
        mode="phase", site_root=tmp_path, common_clip_enabled=True,
        phase_surrogate_slope=1.0,
    )
    assert not torch.equal(phase.apply(0, 8, x), x)
