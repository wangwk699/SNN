from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from snn2.evaluation import position_ids_from_attention_mask
from snn2.model_integration import (
    _make_mlp_forward,
    install_model_integration,
    repeat_kv,
    snn2_eager_attention_forward,
    temporal_forward,
)
from snn2.prefix_cache import build_prefix_key_values, install_prefix_kv_forward
from snn2.sites import site_key
from snn2.temporal_model import deployment_attention_forward
from snn2.temporal_ops import from_temporal, to_temporal


class _BypassController:
    def __init__(self, steps=4, mode="deploy_phase"):
        self.temporal_steps = steps
        self.mode = mode
        self.applied = []
        self.applied_shapes = []

    def apply(self, layer, site, value, **kwargs):
        self.applied.append(site)
        self.applied_shapes.append(tuple(value.shape))
        return value

    def apply_final_norm_phase(self, value):
        return value

    def record_saliency(self, *args):
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
    reference_output = torch.matmul(reference_weights, summed_value).transpose(1, 2).contiguous().reshape(batch, query_len, heads * dim)
    torch.testing.assert_close(
        to_temporal(output, steps).sum(0), reference_output, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        to_temporal(weights, steps).sum(0), reference_weights, rtol=1e-5, atol=1e-5
    )
    assert controller.applied == [2, 3, 4, 5, 6]
    assert controller.applied_shapes[1][-2] == key_len
    assert controller.applied_shapes[2][-2] == key_len
    assert controller.applied_shapes[3][-1] == key_len


def test_static_attention_runtime_applies_sites_3_and_4_to_full_prefix_kv():
    controller = _BypassController(steps=1, mode="identity")
    module = SimpleNamespace(
        _snn2_controller=controller,
        _snn2_layer_index=0,
        training=False,
        num_key_value_groups=1,
        scaling=0.5,
    )
    query = torch.randn(1, 2, 2, 4)
    key = torch.randn(1, 2, 5, 4)
    value = torch.randn_like(key)
    snn2_eager_attention_forward(
        module, query, key, value, None, dropout=0.0
    )
    assert controller.applied[:3] == [2, 3, 4]
    assert controller.applied_shapes[1][-2] == 5
    assert controller.applied_shapes[2][-2] == 5


def test_collect_attention_excludes_prefix_from_site_3_and_4_statistics():
    from snn2.controller import SiteController

    controller = SiteController(mode="collect")
    module = SimpleNamespace(
        _snn2_controller=controller,
        _snn2_layer_index=0,
        training=False,
        num_key_value_groups=2,
        scaling=0.5,
    )
    query = torch.randn(1, 4, 2, 4)
    key = torch.randn(1, 2, 5, 4)
    value = torch.randn_like(key)
    snn2_eager_attention_forward(
        module, query, key, value, None, dropout=0.0
    )
    key_stats = controller.statistics.items[site_key(0, 3)]
    value_stats = controller.statistics.items[site_key(0, 4)]
    assert key_stats.row_count.item() == 2
    assert value_stats.row_count.item() == 2
    assert key_stats.channels == 16
    assert value_stats.channels == 16
    assert key_stats.num_heads == 4
    assert key_stats.channels_per_head == 4
    expected_key = repeat_kv(key, 2)[..., 3:, :].abs().amax(dim=(0, 2))
    expected_value = repeat_kv(value, 2)[..., 3:, :].abs().amax(dim=(0, 2))
    torch.testing.assert_close(key_stats.phase_ema_abs_max, expected_key)
    torch.testing.assert_close(value_stats.phase_ema_abs_max, expected_value)


def test_collect_attention_without_prefix_uses_repeated_query_heads():
    from snn2.controller import SiteController

    controller = SiteController(mode="collect")
    module = SimpleNamespace(
        _snn2_controller=controller,
        _snn2_layer_index=0,
        training=False,
        num_key_value_groups=2,
        scaling=0.5,
    )
    query = torch.randn(1, 4, 2, 4)
    key = torch.randn(1, 2, 2, 4)
    value = torch.randn_like(key)
    snn2_eager_attention_forward(module, query, key, value, None, dropout=0.0)
    for site in (3, 4):
        stats = controller.statistics.items[site_key(0, site)]
        assert stats.channels == 16
        assert stats.num_heads == 4
        assert stats.channels_per_head == 4
        assert stats.phase_ema_abs_max.shape == (4, 4)
        assert stats.saliency_sum_by_role["default"].shape == (4, 4)
    repeated_key = repeat_kv(key, 2)
    repeated_value = repeat_kv(value, 2)
    q64, k64 = query.double(), repeated_key.double()
    qk64 = torch.matmul(q64, k64.transpose(2, 3))
    expected_key = (k64 * torch.matmul(qk64.transpose(2, 3), q64)).sum(dim=(0, 2))
    weights = torch.softmax(torch.matmul(query, repeated_key.transpose(2, 3)) * 0.5, dim=-1)
    p64, v64 = weights.double(), repeated_value.double()
    pv64 = torch.matmul(p64, v64)
    expected_value = (v64 * torch.matmul(p64.transpose(2, 3), pv64)).sum(dim=(0, 2))
    key_stats = controller.statistics.items[site_key(0, 3)]
    value_stats = controller.statistics.items[site_key(0, 4)]
    torch.testing.assert_close(key_stats.saliency_sum_by_role["default"], expected_key)
    torch.testing.assert_close(value_stats.saliency_sum_by_role["default"], expected_value)
    assert key_stats.saliency_sum_by_role["default"].dtype == torch.float64
    assert value_stats.saliency_sum_by_role["default"].dtype == torch.float64


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


def test_deployment_embedding_is_uniformly_distributed_across_time():
    torch.manual_seed(30)
    model = _TinyModel().eval()
    controller = _BypassController(steps=4)
    install_model_integration(model, controller, rotation_state=None)
    ids = torch.tensor([[1, 2, 3]])
    original = model.model.embed_tokens.weight[ids].detach()
    repeated = ids.repeat(4, 1)
    frames = to_temporal(model.model.embed_tokens(repeated), 4)
    for frame in frames:
        torch.testing.assert_close(frame, original / 4)
    torch.testing.assert_close(frames.sum(0), original)
    assert torch.count_nonzero(frames[1:]).item() > 0


def test_model_integration_keeps_non_deploy_forward_unchanged():
    torch.manual_seed(31)
    model = _TinyModel().eval()
    reference = copy.deepcopy(model).eval()
    controller = _BypassController(steps=4, mode="identity")
    install_model_integration(model, controller, rotation_state=None)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mask = torch.ones_like(ids)
    torch.testing.assert_close(model(ids, mask).logits, reference(ids, mask).logits)


def _hf_tiny_model(kind):
    if kind == "qwen3":
        from transformers import Qwen3Config, Qwen3ForCausalLM

        config = Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=32,
            attention_dropout=0.0,
            pad_token_id=0,
        )
        return Qwen3ForCausalLM(config)
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        attention_dropout=0.0,
        pad_token_id=0,
    )
    return LlamaForCausalLM(config)


@pytest.mark.parametrize("kind", ["qwen3", "llama"])
@pytest.mark.parametrize("steps", [None, 2, 4])
def test_prefix_left_padding_batch_invariance(kind, steps):
    torch.manual_seed(37)
    model = _hf_tiny_model(kind).eval()
    prefix = build_prefix_key_values(model, [7, 8])
    controller = _BypassController(
        steps=steps or 1,
        mode="identity" if steps is None else "deploy_phase",
    )
    install_model_integration(model, controller, rotation_state=None)
    install_prefix_kv_forward(model, prefix, controller=controller)

    alone_ids = torch.tensor([[5, 6]])
    alone_mask = torch.ones_like(alone_ids)
    batch_ids = torch.tensor([[0, 0, 5, 6], [10, 11, 12, 13]])
    batch_mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])

    def run(ids, mask):
        positions = position_ids_from_attention_mask(mask)
        if steps is None:
            return model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=positions,
                use_cache=False,
            ).logits
        return temporal_forward(
            model, controller, ids, mask, position_ids=positions
        )

    alone = run(alone_ids, alone_mask)[0, -1]
    batched = run(batch_ids, batch_mask)[0, -1]
    torch.testing.assert_close(batched, alone, rtol=1e-5, atol=1e-5)
