from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

import pytest
import torch

import snn2.model_integration as model_integration
from snn2.config import resolve_config, validate_config
from snn2.model_integration import (
    _make_mlp_forward,
    _selective_checkpoint_allowed,
    snn2_eager_attention_forward,
)


ROOT = Path(__file__).resolve().parents[1]


def _generated_config(mode: str) -> dict:
    return yaml.safe_load(
        (ROOT / "configs" / "generated" / f"exp2_llama3_8b_tulu3__{mode}.yaml").read_text(
            encoding="utf-8"
        )
    )

def test_memory_config_defaults_and_llama3_variants():
    legacy = _generated_config("vanilla")
    legacy.pop("ann_training_memory")
    resolved = resolve_config(legacy)
    assert resolved["ann_training_memory"] == {
        "attention_core_checkpoint": False,
        "mlp_checkpoint": False,
    }
    validate_config(resolved)
    for mode, expected in {
        "vanilla": False,
        "unaware": False,
        "phase_aware": True,
        "gif_aware": True,
    }.items():
        memory = _generated_config(mode)["ann_training_memory"]
        assert memory["attention_core_checkpoint"] is expected
        assert memory["mlp_checkpoint"] is expected


@pytest.mark.parametrize("mode", ["vanilla", "unaware"])
def test_memory_config_rejects_non_aware_modes(mode):
    cfg = _generated_config(mode)
    cfg["ann_training_memory"]["attention_core_checkpoint"] = True
    with pytest.raises(ValueError, match="only valid for aware ANN modes"):
        validate_config(cfg)


@pytest.mark.parametrize("mode", ["phase_aware", "gif_aware"])
def test_memory_config_allows_aware_modes(mode):
    cfg = _generated_config(mode)
    validate_config(cfg)


def test_memory_config_rejects_transformers_gradient_checkpointing():
    cfg = _generated_config("phase_aware")
    cfg["training"]["gradient_checkpointing"] = True
    with pytest.raises(ValueError, match="must not be combined"):
        validate_config(cfg)


def test_memory_config_rejects_unknown_or_non_boolean_keys():
    cfg = _generated_config("phase_aware")
    cfg["ann_training_memory"]["unknown"] = False
    with pytest.raises(ValueError, match="Unsupported"):
        validate_config(cfg)
    cfg = _generated_config("phase_aware")
    cfg["ann_training_memory"]["mlp_checkpoint"] = 1
    with pytest.raises(ValueError, match="must be true or false"):
        validate_config(cfg)


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
    for actual, expected in zip(on, off):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_attention_checkpoint_supports_prefix_key_value_length():
    off = _attention_run(mode="phase", checkpoint_enabled=False, prefix=3)
    on = _attention_run(mode="phase", checkpoint_enabled=True, prefix=3)
    assert off[1].shape == (2, 4, 8, 11)
    for actual, expected in zip(on, off):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_attention_checkpoint_preserves_dropout_rng():
    torch.manual_seed(31)
    off = _attention_run(mode="phase", checkpoint_enabled=False, dropout=0.25)
    torch.manual_seed(31)
    on = _attention_run(mode="phase", checkpoint_enabled=True, dropout=0.25)
    for actual, expected in zip(on, off):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


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


@pytest.mark.parametrize("r4", [None, object()])
def test_mlp_checkpoint_forward_and_backward_match(monkeypatch, r4):
    off = _mlp_run(checkpoint_enabled=False, r4=r4, monkeypatch=monkeypatch)
    on = _mlp_run(checkpoint_enabled=True, r4=r4, monkeypatch=monkeypatch)
    for actual, expected in zip(on, off):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("mode", ["identity", "none", "collect", "deploy_phase", "deploy_gif", "deploy_mtn"])
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


def test_checkpoint_guard_rejects_unknown_kind():
    controller = _ReplacementController("phase", attention=True, mlp=True)
    with pytest.raises(ValueError, match="unknown"):
        _selective_checkpoint_allowed(torch.nn.Linear(2, 2).train(), controller, kind="unknown")
