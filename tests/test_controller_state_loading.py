import pytest
import torch

from snn2.controller import SiteController
from snn2.neurons import Clipper, PhaseSurrogate
from snn2.phase_statistics import (
    PHASE_TAU_ACCUMULATOR_DTYPE,
    PHASE_TAU_CALIBRATION,
    PHASE_TAU_CHANNEL_POLICY,
    PHASE_TAU_EMA_FACTOR,
    PHASE_TAU_REDUCTION_POLICY,
)
from snn2.sites import site_key
from snn2.temporal_ops import (
    GIF_INTEGER_DECOMPOSITION,
    SITE_STATE_FORMAT_VERSION,
    TEMPORAL_IMPLEMENTATION_VERSION,
)


def _header(kind):
    return {
        "state_kind": kind,
        "format_version": SITE_STATE_FORMAT_VERSION,
        "temporal_implementation_version": TEMPORAL_IMPLEMENTATION_VERSION,
    }


def _layout():
    return {
        "parameter_layout": "last_dim_grouped",
        "configured_group_size": -1,
        "group_size": 4,
        "num_heads": None,
        "channels_per_head": 4,
        "groups_per_head": 1,
    }


def _phase_state():
    return {
        **_header("phase"), **_layout(), "tau": torch.tensor([2.0]),
        "tau_calibration": PHASE_TAU_CALIBRATION,
        "tau_ema_factor": PHASE_TAU_EMA_FACTOR,
        "tau_accumulator_dtype": PHASE_TAU_ACCUMULATOR_DTYPE,
        "tau_channel_policy": PHASE_TAU_CHANNEL_POLICY,
        "tau_reduction_policy": PHASE_TAU_REDUCTION_POLICY,
    }


def _clip_state(*, roles=None):
    common = {
        **_header("clip"), **_layout(), "phase_T": 4, "mtn_T": 4,
        "ordinary_gif_high_qmax": 30,
        "ordinary_gif_per_step_qmax": 15,
        "gif_integer_decomposition": GIF_INTEGER_DECOMPOSITION,
        "gif_constraint_policy": "identity",
        "gif_group_classification_enum": {"all_low": 0, "all_high": 1, "mixed": 2},
    }
    if roles is None:
        return {
            **common, "clip_role_policy": "single",
            "lower": torch.tensor([-0.25]), "upper": torch.tensor([0.25]),
            "rule": "intersection(phase, mtn)",
        }
    return {
        **common, "clip_role_policy": "role_specific", "clip_roles": list(roles),
        "lower_by_role": {role: torch.tensor([-index - 1.0]) for index, role in enumerate(roles)},
        "upper_by_role": {role: torch.tensor([index + 1.0]) for index, role in enumerate(roles)},
        "rule": "mask_aware_per_group_role_specific",
    }


def _write(root, site_index, name, state, *, clip=False):
    directory = root / site_key(0, site_index)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(state, directory / ("clip_state.pt" if clip else f"{name}_state.pt"))


def test_phase_state_is_runtime_t_independent_and_v0_is_derived():
    state = _phase_state()
    assert not {"T", "base", "v0", "max_spikes"}.intersection(state)
    for T in (1, 2, 4, 8):
        module = PhaseSurrogate(state, T=T)
        torch.testing.assert_close(module.v0, 0.5 * module.tau * 2.0 ** (-T))


def test_phase_controller_loads_stage_a_and_separate_stage_b(tmp_path):
    stage_a, stage_b = tmp_path / "sites", tmp_path / "profile"
    _write(stage_a, 6, "phase", _phase_state())
    _write(stage_b, 6, "clip", _clip_state(), clip=True)
    controller = SiteController(
        mode="phase", site_root=stage_a, clip_root=stage_b,
        common_clip_enabled=True, phase_T=4, phase_surrogate_slope=1.0,
    )
    output = controller.apply(0, 6, torch.full((1, 1, 4), 3.0))
    assert torch.all(output <= 0.25)


def test_role_specific_clipper_requires_valid_role():
    clipper = Clipper(_clip_state(roles=("q", "k", "v")))
    x = torch.full((1, 1, 4), 4.0)
    with pytest.raises(ValueError, match="role must be"):
        clipper(x)
    with pytest.raises(ValueError, match="role must be"):
        clipper(x, role="bad")
    assert torch.equal(clipper(x, role="q"), torch.ones_like(x))


def test_phase_site1_shared_apply_does_not_clip_then_branch_clips(tmp_path):
    stage_a, stage_b = tmp_path / "sites", tmp_path / "profile"
    _write(stage_a, 1, "phase", _phase_state())
    _write(stage_b, 1, "clip", _clip_state(roles=("q", "k", "v")), clip=True)
    controller = SiteController(
        mode="phase", site_root=stage_a, clip_root=stage_b,
        common_clip_enabled=True, phase_T=4, phase_surrogate_slope=1.0,
    )
    shared = controller.apply(0, 1, torch.full((1, 1, 4), 3.0))
    assert "clip" not in controller._modules[site_key(0, 1)]
    clipped = controller.apply_role_clip(0, 1, shared, role="q")
    assert torch.all(clipped <= 1.0)


@pytest.mark.parametrize(
    ("site_index", "roles", "role"),
    [(1, ("q", "k", "v"), "q"), (7, ("gate", "up"), "gate")],
)
def test_phase_multirole_shared_apply_never_reuses_role_clip(tmp_path, site_index, roles, role):
    stage_a, stage_b = tmp_path / "sites", tmp_path / "profile"
    _write(stage_a, site_index, "phase", _phase_state())
    _write(stage_b, site_index, "clip", _clip_state(roles=roles), clip=True)
    controller = SiteController(
        mode="phase", site_root=stage_a, clip_root=stage_b,
        common_clip_enabled=True, phase_T=4, phase_surrogate_slope=1.0,
    )
    x = torch.full((1, 1, 4), 3.0)
    shared_before = controller.apply(0, site_index, x)
    controller.apply_role_clip(0, site_index, shared_before, role=role)
    assert "clip" in controller._modules[site_key(0, site_index)]
    shared_after = controller.apply(0, site_index, x)
    torch.testing.assert_close(shared_after, shared_before)

def test_deployment_uses_explicit_runtime_parameters(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "snn2.controller.validate_site_state_bundle",
        lambda *_args, **_kwargs: {"temporal_steps": {"gif": 2}},
    )
    controller = SiteController(
        site_root=tmp_path, phase_T=8, mtn_T=6, mtn_K=4,
        mtn_threshold_factor=0.75,
    )
    assert controller.set_deployment("phase", clip_bundle_policy="forbid_all") == 8
    assert controller.set_deployment("mtn", clip_bundle_policy="forbid_all") == 6


def test_legacy_phase_and_mtn_runtime_fields_fail_fast():
    phase = _phase_state()
    phase["T"] = 4
    with pytest.raises(ValueError, match="pre-A/B"):
        PhaseSurrogate(phase, T=4)


def test_common_clip_requires_explicit_stage_b_root():
    with pytest.raises(ValueError, match="clip_root"):
        SiteController(mode="gif", site_root="sites", common_clip_enabled=True)


def test_snn_deployment_rejects_common_clip_switch(tmp_path):
    controller = SiteController(
        mode="gif", site_root=tmp_path, clip_root=tmp_path,
        common_clip_enabled=True,
    )
    with pytest.raises(ValueError, match="cannot enable common Clip"):
        controller.set_deployment("gif", clip_bundle_policy="forbid_all")
