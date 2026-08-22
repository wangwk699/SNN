from __future__ import annotations

import pytest
from types import SimpleNamespace
import torch

from snn2.prefix_cache import (
    _extend_attention_mask,
    _fresh_dynamic_cache,
    install_prefix_kv_forward,
)
from snn2.temporal_ops import to_temporal


def _prefix(layer_count=2, saved_batch=1):
    result = []
    for layer in range(layer_count):
        key = torch.arange(saved_batch * 2 * 3 * 4, dtype=torch.float32).reshape(
            saved_batch, 2, 3, 4
        ) + 100 * layer
        value = key + 1000
        result.append((key, value))
    return tuple(result)


@pytest.mark.parametrize("steps", [2, 4])
@pytest.mark.parametrize("batch", [1, 3])
def test_temporal_prefix_is_time_major_uniform_fraction(steps, batch):
    prefix = _prefix()
    cache = _fresh_dynamic_cache(
        prefix, logical_batch_size=batch, temporal_steps=steps
    ).to_legacy_cache()
    assert len(cache) == 2
    for layer, (key, value) in enumerate(cache):
        assert key.shape == (steps * batch, 2, 3, 4)
        expected_key = prefix[layer][0].expand(batch, -1, -1, -1)
        expected_value = prefix[layer][1].expand(batch, -1, -1, -1)
        temporal_key = to_temporal(key, steps)
        temporal_value = to_temporal(value, steps)
        for timestep in range(steps):
            torch.testing.assert_close(temporal_key[timestep], expected_key / steps)
            torch.testing.assert_close(temporal_value[timestep], expected_value / steps)
        torch.testing.assert_close(temporal_key.sum(0), expected_key)
        torch.testing.assert_close(temporal_value.sum(0), expected_value)


@pytest.mark.parametrize("batch", [1, 3])
def test_ann_prefix_keeps_one_complete_copy_per_sample(batch):
    prefix = _prefix(layer_count=1)
    key, value = _fresh_dynamic_cache(
        prefix, logical_batch_size=batch
    ).to_legacy_cache()[0]
    torch.testing.assert_close(key, prefix[0][0].expand(batch, -1, -1, -1))
    torch.testing.assert_close(value, prefix[0][1].expand(batch, -1, -1, -1))


def test_saved_multi_batch_prefix_must_match_logical_batch():
    with pytest.raises(ValueError, match="logical batch"):
        _fresh_dynamic_cache(_prefix(saved_batch=2), logical_batch_size=3)


def test_prefix_mask_is_full_visibility_in_every_temporal_frame():
    steps, batch, current, prefix = 4, 3, 5, 2
    mask = torch.tensor([[1, 1, 1, 0, 0]] * (steps * batch))
    extended = _extend_attention_mask(
        mask,
        batch_size=steps * batch,
        current_length=current,
        cached_prefix_length=prefix,
        device=mask.device,
    )
    temporal = to_temporal(extended, steps)
    assert torch.all(temporal[..., :prefix] == 1)
    for timestep in range(1, steps):
        torch.testing.assert_close(temporal[timestep], temporal[0])


class _ForwardEcho(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, **kwargs):
        return kwargs


def test_temporal_prefix_offsets_position_and_cache_position_once():
    steps, batch, current = 2, 3, 4
    model = _ForwardEcho()
    controller = SimpleNamespace(mode="deploy_phase", temporal_steps=steps)
    install_prefix_kv_forward(model, _prefix(layer_count=1), controller=controller)
    input_ids = torch.zeros(steps * batch, current, dtype=torch.long)
    position_ids = torch.arange(current).expand(steps * batch, -1)
    cache_position = torch.arange(current)
    result = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        position_ids=position_ids,
        cache_position=cache_position,
    )
    prefix_len = 3
    torch.testing.assert_close(result["position_ids"], position_ids + prefix_len)
    torch.testing.assert_close(result["cache_position"], cache_position + prefix_len)
    assert result["attention_mask"].shape[-1] == current + prefix_len
    assert torch.all(result["attention_mask"][:, :prefix_len] == 1)
