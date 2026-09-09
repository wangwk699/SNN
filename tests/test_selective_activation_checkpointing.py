from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import snn2.model_integration as model_integration
from snn2.hadamard import make_spec
from snn2.model_integration import (
    _make_mlp_forward,
    _selective_checkpoint_allowed,
    snn2_eager_attention_forward,
)
from snn2.neurons import PhaseSurrogate, StaticGIF
from snn2.phase_statistics import (
    NEURON_PARAMETER_CLAMP_MAX,
    NEURON_PARAMETER_CLAMP_MIN,
    NEURON_PARAMETER_CLAMP_POLICY,
    PHASE_TAU_ACCUMULATOR_DTYPE,
    PHASE_TAU_CALIBRATION,
    PHASE_TAU_CHANNEL_POLICY,
    PHASE_TAU_EMA_FACTOR,
    PHASE_TAU_REDUCTION_POLICY,
)
from snn2.temporal_ops import (
    GIF_ADD_BITS,
    GIF_BASE_BITS,
    GIF_HIGH_QMAX,
    GIF_INTEGER_DECOMPOSITION,
    GIF_LOCAL_STEPS,
    GIF_LOW_QMAX,
    GIF_SALIENT_POLICY,
    GIF_STEP_QMAX,
    SITE_STATE_FORMAT_VERSION,
    TEMPORAL_IMPLEMENTATION_VERSION,
)


class _ReplacementController:
    def __init__(self, mode: str, *, attention: bool, mlp: bool):
        self.mode = mode
        self.checkpoint_attention_core = attention
        self.checkpoint_mlp = mlp
        self.regression_recorder = None
        self.gain = torch.nn.Parameter(torch.tensor(1.125))

    def apply(self, _layer: int, site: int, value: torch.Tensor, **_kwargs) -> torch.Tensor:
        # Sites 5/8/9/10 stand in for trainable Phase/GIF ANN replacements.
        return value * self.gain if site in {5, 8, 9, 10} else value


def _attention_run(*, mode: str, checkpoint_enabled: bool, prefix: int = 0, dropout: float = 0.0):
    torch.manual_seed(101)
    query = torch.randn(2, 4, 8, 16, requires_grad=True)
    key = torch.randn(2, 4, 8 + prefix, 16, requires_grad=True)
    value = torch.randn(2, 4, 8 + prefix, 16, requires_grad=True)
    mask = torch.zeros(2, 1, 8, 8 + prefix)
    controller = _ReplacementController(mode, attention=checkpoint_enabled, mlp=False)
    module = SimpleNamespace(
        _snn2_controller=controller,
        _snn2_layer_index=0,
        num_key_value_groups=1,
        scaling=0.25,
        training=True,
    )
    output, weights = snn2_eager_attention_forward(
        module, query, key, value, mask, dropout=dropout
    )
    (output.float().square().mean() + weights.float().square().mean()).backward()
    return output.detach(), weights.detach(), query.grad, key.grad, value.grad, controller.gain.grad


@pytest.mark.parametrize("mode", ["phase", "gif"])
def test_attention_checkpoint_forward_and_backward_match(mode):
    off = _attention_run(mode=mode, checkpoint_enabled=False)
    on = _attention_run(mode=mode, checkpoint_enabled=True)
    _assert_close_runs(on, off)


def test_attention_checkpoint_supports_prefix_key_value_length():
    off = _attention_run(mode="phase", checkpoint_enabled=False, prefix=3)
    on = _attention_run(mode="phase", checkpoint_enabled=True, prefix=3)
    assert off[1].shape == (2, 4, 8, 11)
    _assert_close_runs(on, off)


def test_attention_checkpoint_preserves_dropout_rng():
    torch.manual_seed(31)
    off = _attention_run(mode="phase", checkpoint_enabled=False, dropout=0.25)
    torch.manual_seed(31)
    on = _attention_run(mode="phase", checkpoint_enabled=True, dropout=0.25)
    _assert_close_runs(on, off)


def _state_header(kind: str) -> dict:
    return {
        "state_kind": kind,
        "format_version": SITE_STATE_FORMAT_VERSION,
        "temporal_implementation_version": TEMPORAL_IMPLEMENTATION_VERSION,
    }


def _real_phase_state(*, layout: str, channels: int | None = None, heads: int = 2) -> dict:
    if layout == "last_dim_grouped":
        if channels is None or channels % 2:
            raise ValueError("last_dim_grouped Phase test requires an even channel count")
        state_layout = {
            "parameter_layout": layout,
            "configured_group_size": 2,
            "group_size": 2,
            "num_heads": None,
            "channels_per_head": channels,
            "groups_per_head": channels // 2,
        }
        tau = torch.full((channels // 2,), 0.5)
    elif layout == "attention_head_scalar":
        state_layout = {
            "parameter_layout": layout,
            "configured_group_size": -1,
            "group_size": -1,
            "num_heads": heads,
            "channels_per_head": None,
            "groups_per_head": 1,
        }
        tau = torch.full((heads, 1), 0.5)
    else:
        raise ValueError(layout)
    return {
        **_state_header("phase"),
        **state_layout,
        "tau": tau,
        "tau_calibration": PHASE_TAU_CALIBRATION,
        "tau_ema_factor": PHASE_TAU_EMA_FACTOR,
        "tau_accumulator_dtype": PHASE_TAU_ACCUMULATOR_DTYPE,
        "tau_channel_policy": PHASE_TAU_CHANNEL_POLICY,
        "tau_reduction_policy": PHASE_TAU_REDUCTION_POLICY,
        "tau_clamp_min": NEURON_PARAMETER_CLAMP_MIN,
        "tau_clamp_max": NEURON_PARAMETER_CLAMP_MAX,
        "tau_clamp_policy": NEURON_PARAMETER_CLAMP_POLICY,
    }


def _real_gif_state(*, channels: int = 12) -> dict:
    if channels % 2:
        raise ValueError("last_dim_grouped GIF test requires an even channel count")
    groups = channels // 2
    return {
        **_state_header("gif"),
        "parameter_layout": "last_dim_grouped",
        "configured_group_size": 2,
        "group_size": 2,
        "num_heads": None,
        "channels_per_head": channels,
        "groups_per_head": groups,
        "gif_policy": GIF_SALIENT_POLICY,
        "base_bits": GIF_BASE_BITS,
        "add_bits": GIF_ADD_BITS,
        "low_qmin": 0,
        "low_qmax": GIF_LOW_QMAX,
        "high_qmin": 0,
        "high_qmax": GIF_HIGH_QMAX,
        "temporal_steps": GIF_LOCAL_STEPS,
        "per_step_qmin": 0,
        "per_step_qmax": GIF_STEP_QMAX,
        "integer_decomposition": GIF_INTEGER_DECOMPOSITION,
        "low_scale": torch.full((groups,), 0.1),
        "low_zero": torch.zeros(groups),
        "high_scale": torch.full((groups,), 0.05),
        "high_zero": torch.zeros(groups),
        "mask_low": torch.tensor(
            [True, False] * (channels // 2), dtype=torch.bool
        ),
    }


class _RealPhaseAttentionController:
    def __init__(self, checkpoint_enabled: bool):
        self.mode = "phase"
        self.checkpoint_attention_core = checkpoint_enabled
        self.checkpoint_mlp = False
        self.regression_recorder = None
        self.phase = PhaseSurrogate(
            _real_phase_state(layout="attention_head_scalar"),
            T=4,
            surrogate_slope=1.0,
        )

    def apply(self, _layer: int, site: int, value: torch.Tensor, **_kwargs) -> torch.Tensor:
        return self.phase(value) if site == 5 else value


class _RealPhaseMLPController:
    def __init__(self, checkpoint_enabled: bool):
        self.mode = "phase"
        self.checkpoint_attention_core = False
        self.checkpoint_mlp = checkpoint_enabled
        self.regression_recorder = None
        self.modules = {
            site: PhaseSurrogate(
                _real_phase_state(layout="last_dim_grouped", channels=12),
                T=4,
                surrogate_slope=1.0,
            )
            for site in (8, 9, 10)
        }

    def apply(self, _layer: int, site: int, value: torch.Tensor, **_kwargs) -> torch.Tensor:
        return self.modules[site](value) if site in self.modules else value


class _RealGIFMLPController:
    def __init__(self, checkpoint_enabled: bool):
        self.mode = "gif"
        self.checkpoint_attention_core = False
        self.checkpoint_mlp = checkpoint_enabled
        self.regression_recorder = None
        self.modules = {site: StaticGIF(_real_gif_state()) for site in (8, 9, 10)}

    def apply(self, _layer: int, site: int, value: torch.Tensor, **_kwargs) -> torch.Tensor:
        return self.modules[site](value) if site in self.modules else value


def _real_phase_attention_run(checkpoint_enabled: bool):
    torch.manual_seed(211)
    query = (torch.randn(1, 2, 4, 4) * 0.5).requires_grad_()
    key = (torch.randn(1, 2, 4, 4) * 0.5).requires_grad_()
    value = (torch.randn(1, 2, 4, 4) * 0.5).requires_grad_()
    controller = _RealPhaseAttentionController(checkpoint_enabled)
    module = SimpleNamespace(
        _snn2_controller=controller,
        _snn2_layer_index=0,
        num_key_value_groups=1,
        scaling=0.5,
        training=True,
    )
    output, weights = snn2_eager_attention_forward(
        module, query, key, value, torch.zeros(1, 1, 4, 4), dropout=0.0
    )
    output.float().square().mean().backward()
    return output.detach(), weights.detach(), query.grad, key.grad, value.grad


def test_real_phase_attention_checkpoint_forward_backward_match():
    off = _real_phase_attention_run(False)
    on = _real_phase_attention_run(True)
    _assert_close_runs(on, off)
    assert torch.count_nonzero(on[2]) > 0


class _MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(6, 12, bias=False)
        self.up_proj = torch.nn.Linear(6, 12, bias=False)
        self.down_proj = torch.nn.Linear(12, 6, bias=False)
        self.act_fn = torch.nn.SiLU()


def _mlp_run(*, checkpoint_enabled: bool, r4, monkeypatch):
    torch.manual_seed(37)
    mlp = _MLP()
    x = torch.randn(2, 5, 6, requires_grad=True)
    controller = _ReplacementController("phase", attention=False, mlp=checkpoint_enabled)
    monkeypatch.setattr(model_integration, "random_hadamard", lambda value, _spec: value)
    output = _make_mlp_forward(controller, 0, r4)(mlp, x)
    output.float().square().mean().backward()
    return output.detach(), x.grad, mlp.gate_proj.weight.grad, mlp.up_proj.weight.grad, mlp.down_proj.weight.grad, controller.gain.grad


def test_mlp_checkpoint_forward_and_backward_match(monkeypatch):
    off = _mlp_run(checkpoint_enabled=False, r4=None, monkeypatch=monkeypatch)
    on = _mlp_run(checkpoint_enabled=True, r4=None, monkeypatch=monkeypatch)
    _assert_close_runs(on, off)


def _real_mlp_run(controller_cls, *, checkpoint_enabled: bool, r4=None):
    torch.manual_seed(307)
    mlp = _MLP()
    x = (torch.randn(2, 5, 6) * 0.25 + 0.2).requires_grad_()
    controller = controller_cls(checkpoint_enabled)
    output = _make_mlp_forward(controller, 0, r4)(mlp, x)
    output.float().square().mean().backward()
    return (
        output.detach(),
        x.grad,
        mlp.gate_proj.weight.grad,
        mlp.up_proj.weight.grad,
        mlp.down_proj.weight.grad,
    )


def test_real_phase_mlp_checkpoint_forward_backward_match():
    off = _real_mlp_run(_RealPhaseMLPController, checkpoint_enabled=False)
    on = _real_mlp_run(_RealPhaseMLPController, checkpoint_enabled=True)
    _assert_close_runs(on, off)
    assert torch.count_nonzero(on[1]) > 0


def test_real_static_gif_mlp_checkpoint_forward_backward_match():
    off = _real_mlp_run(_RealGIFMLPController, checkpoint_enabled=False)
    on = _real_mlp_run(_RealGIFMLPController, checkpoint_enabled=True)
    _assert_close_runs(on, off)
    assert torch.isfinite(on[1]).all()
    assert torch.count_nonzero(on[1]) > 0


def test_real_phase_r4_mlp_checkpoint_forward_backward_match():
    r4 = make_spec("R4_test", dimension=12, seed=123)
    off = _real_mlp_run(_RealPhaseMLPController, checkpoint_enabled=False, r4=r4)
    on = _real_mlp_run(_RealPhaseMLPController, checkpoint_enabled=True, r4=r4)
    _assert_close_runs(on, off)
    assert torch.count_nonzero(on[2]) > 0


def _assert_close_runs(actual, expected) -> None:
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "mode", ["identity", "none", "collect", "deploy_phase", "deploy_gif", "deploy_mtn"]
)
def test_checkpoint_guard_rejects_non_aware_modes(mode):
    controller = _ReplacementController(mode, attention=True, mlp=True)
    module = torch.nn.Linear(2, 2).train()
    assert not _selective_checkpoint_allowed(module, controller, kind="attention")
    assert not _selective_checkpoint_allowed(module, controller, kind="mlp")


def test_checkpoint_guard_rejects_eval_and_regression_recording():
    controller = _ReplacementController("phase", attention=True, mlp=True)
    module = torch.nn.Linear(2, 2).eval()
    assert not _selective_checkpoint_allowed(module, controller, kind="attention")
    module.train()
    controller.regression_recorder = object()
    assert not _selective_checkpoint_allowed(module, controller, kind="mlp")


def test_checkpoint_guard_rejects_no_grad():
    controller = _ReplacementController("phase", attention=True, mlp=True)
    module = torch.nn.Linear(2, 2).train()
    with torch.no_grad():
        assert not _selective_checkpoint_allowed(module, controller, kind="attention")
        assert not _selective_checkpoint_allowed(module, controller, kind="mlp")


def test_checkpoint_guard_rejects_unknown_kind():
    controller = _ReplacementController("phase", attention=True, mlp=True)
    with pytest.raises(ValueError, match="unknown"):
        _selective_checkpoint_allowed(torch.nn.Linear(2, 2).train(), controller, kind="unknown")
