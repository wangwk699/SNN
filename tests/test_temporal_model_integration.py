from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from snn2.model_integration import (
    _make_mlp_forward,
    install_model_integration,
    repeat_kv,
    temporal_forward,
)
from snn2.temporal_model import deployment_attention_forward
from snn2.temporal_ops import from_temporal, to_temporal


class _BypassController:
    def __init__(self, steps=4, mode="deploy_phase"):
        self.temporal_steps = steps
        self.mode = mode
        self.applied = []

    def apply(self, layer, site, value):
        self.applied.append(site)
        return value

    def record_saliency(self, *args):
        pass

    def record_saliency_reduced(self, *args):
        pass


@pytest.mark.parametrize("with_prefix", [False, True])
def test_temporal_attention_sum_matches_ann_with_gqa_mask_and_batch(with_prefix):
    torch.manual_seed(23)
    steps, batch, heads, kv_heads, query_len, current_len, dim = 4, 2, 4, 2, 3, 3, 5
    prefix_len = 2 if with_prefix else 0
    query = torch.randn(steps, batch, heads, query_len, dim)
    current_key = torch.randn(steps, batch, kv_heads, current_len, dim)
    current_value = torch.randn_like(current_key)
    if with_prefix:
        prefix_key = torch.randn(batch, kv_heads, prefix_len, dim)
        prefix_value = torch.randn_like(prefix_key)
        temporal_prefix_key = prefix_key.unsqueeze(0).expand(steps, -1, -1, -1, -1) / steps
        temporal_prefix_value = prefix_value.unsqueeze(0).expand_as(temporal_prefix_key) / steps
        key = torch.cat((temporal_prefix_key, current_key), dim=-2)
        value = torch.cat((temporal_prefix_value, current_value), dim=-2)
    else:
        key, value = current_key, current_value
    key_len = prefix_len + current_len
    mask = torch.zeros(batch, 1, query_len, key_len)
    causal = torch.triu(torch.ones(query_len, current_len, dtype=torch.bool), diagonal=1)
    mask[..., prefix_len:] = mask[..., prefix_len:].masked_fill(causal, float("-inf"))
    mask[1, ..., -1] = float("-inf")
    flat_mask = mask.unsqueeze(0).expand(steps, -1, -1, -1, -1).reshape(
        steps * batch, 1, query_len, key_len
    )
    controller = _BypassController(steps)
    module = SimpleNamespace(
        training=False,
        num_key_value_groups=heads // kv_heads,
        scaling=dim**-0.5,
    )
    output, weights = deployment_attention_forward(
        module,
        from_temporal(query),
        from_temporal(key),
        from_temporal(value),
        flat_mask,
        scaling=None,
        dropout=0.0,
        controller=controller,
        layer_index=0,
        repeat_kv=repeat_kv,
        softcap=None,
    )
    summed_query = query.sum(0)
    summed_key = repeat_kv(key.sum(0), heads // kv_heads)
    summed_value = repeat_kv(value.sum(0), heads // kv_heads)
    reference_weights = torch.softmax(
        torch.matmul(summed_query, summed_key.transpose(-2, -1)) * dim**-0.5 + mask,
        dim=-1,
    )
    reference_output = torch.matmul(reference_weights, summed_value).transpose(1, 2)
    torch.testing.assert_close(
        to_temporal(output, steps).sum(0), reference_output, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        to_temporal(weights, steps).sum(0), reference_weights, rtol=1e-5, atol=1e-5
    )
    assert controller.applied == [2, 3, 4, 5, 6]


class _RMSNorm(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(0.7, 1.3, hidden))
        self.variance_epsilon = 1e-6

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.variance_epsilon)


class _MLP(torch.nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden, intermediate, bias=True)
        self.up_proj = torch.nn.Linear(hidden, intermediate, bias=True)
        self.down_proj = torch.nn.Linear(intermediate, hidden, bias=True)
        self.act_fn = torch.nn.functional.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _AttentionShell(torch.nn.Module):
    def __init__(self, hidden, with_qk_norm):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden, hidden, bias=True)
        self.k_proj = torch.nn.Linear(hidden, hidden, bias=True)
        self.v_proj = torch.nn.Linear(hidden, hidden, bias=True)
        self.o_proj = torch.nn.Linear(hidden, hidden, bias=True)
        self.num_heads = 2
        self.config = SimpleNamespace(num_attention_heads=2)
        if with_qk_norm:
            self.q_norm = _RMSNorm(hidden // 2)
            self.k_norm = _RMSNorm(hidden // 2)


class _Layer(torch.nn.Module):
    def __init__(self, hidden, with_qk_norm):
        super().__init__()
        self.input_layernorm = _RMSNorm(hidden)
        self.post_attention_layernorm = _RMSNorm(hidden)
        self.self_attn = _AttentionShell(hidden, with_qk_norm)
        self.mlp = _MLP(hidden, hidden * 2)


class _Backbone(torch.nn.Module):
    def __init__(self, vocab, hidden, with_qk_norm):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab, hidden)
        self.layers = torch.nn.ModuleList([_Layer(hidden, with_qk_norm)])
        self.norm = _RMSNorm(hidden)


class _TinyModel(torch.nn.Module):
    def __init__(self, with_qk_norm=True):
        super().__init__()
        self.model = _Backbone(17, 6, with_qk_norm)
        self.lm_head = torch.nn.Linear(6, 17, bias=True)
        self.config = SimpleNamespace()

    def forward(self, input_ids, attention_mask, use_cache=False):
        x = self.model.embed_tokens(input_ids)
        layer = self.model.layers[0]
        x = x + layer.input_layernorm(x)
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        return SimpleNamespace(logits=self.lm_head(self.model.norm(x)))


@pytest.mark.parametrize("with_qk_norm", [True, False])
def test_tiny_decoder_temporal_sum_matches_ann_and_wraps_optional_qk_norm(with_qk_norm):
    torch.manual_seed(29)
    model = _TinyModel(with_qk_norm).eval()
    reference = copy.deepcopy(model).eval()
    controller = _BypassController(steps=4)
    install_model_integration(model, controller, rotation_state=None)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mask = torch.ones_like(ids)
    expected = reference(ids, mask).logits
    actual = temporal_forward(model, controller, ids, mask)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert model.model.norm in model._snn2_wrapped_norms
    attention = model.model.layers[0].self_attn
    if with_qk_norm:
        assert attention.q_norm in model._snn2_wrapped_norms
        assert attention.k_norm in model._snn2_wrapped_norms
    else:
        assert not hasattr(attention, "q_norm")


def test_model_integration_keeps_non_deploy_forward_unchanged():
    torch.manual_seed(31)
    model = _TinyModel().eval()
    reference = copy.deepcopy(model).eval()
    controller = _BypassController(steps=4, mode="identity")
    install_model_integration(model, controller, rotation_state=None)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mask = torch.ones_like(ids)
    torch.testing.assert_close(model(ids, mask).logits, reference(ids, mask).logits)
