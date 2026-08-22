from __future__ import annotations

from typing import Any, Callable

import torch

from .controller import SiteController
from .temporal_ops import (
    from_temporal,
    temporal_seq_matmul,
    temporal_softmax,
    to_temporal,
)


def deployment_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    scaling: float | None,
    dropout: float,
    controller: SiteController,
    layer_index: int,
    repeat_kv: Callable[[torch.Tensor, int], torch.Tensor],
    softcap: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    steps = int(controller.temporal_steps or 0)
    if steps <= 0:
        raise RuntimeError("Deployment attention requires a positive temporal step count")
    if module.training or float(dropout) != 0.0:
        raise RuntimeError("Temporal deployment attention requires model.eval() and dropout=0")

    current_length = int(query.shape[-2])
    past_length = max(int(key.shape[-2]) - current_length, 0)
    query = controller.apply(layer_index, 2, query)
    if past_length:
        prefix_key, current_key = key[..., :past_length, :], key[..., past_length:, :]
        prefix_value, current_value = value[..., :past_length, :], value[..., past_length:, :]
        current_key = controller.apply(layer_index, 3, current_key)
        current_value = controller.apply(layer_index, 4, current_value)
        key = torch.cat((prefix_key, current_key), dim=-2)
        value = torch.cat((prefix_value, current_value), dim=-2)
    else:
        key = controller.apply(layer_index, 3, key)
        value = controller.apply(layer_index, 4, value)

    groups = int(getattr(module, "num_key_value_groups", 1))
    key = repeat_kv(key, groups)
    value = repeat_kv(value, groups)
    temporal_query = to_temporal(query, steps)
    temporal_key_t = to_temporal(key.transpose(2, 3), steps)
    qk_increment = temporal_seq_matmul(temporal_query, temporal_key_t)
    scale = float(
        scaling
        if scaling is not None
        else getattr(module, "scaling", query.shape[-1] ** -0.5)
    )
    score_increment = qk_increment * scale
    if attention_mask is not None:
        attention_mask = attention_mask[..., : key.shape[-2]]
    weight_increment = temporal_softmax(
        score_increment,
        attention_mask,
        softcap=softcap,
    )
    flat_weights = from_temporal(weight_increment)
    if past_length:
        prefix_weights = flat_weights[..., :past_length]
        current_weights = controller.apply(
            layer_index, 5, flat_weights[..., past_length:]
        )
        flat_weights = torch.cat((prefix_weights, current_weights), dim=-1)
    else:
        flat_weights = controller.apply(layer_index, 5, flat_weights)
    weight_increment = to_temporal(flat_weights, steps)
    output_increment = temporal_seq_matmul(
        weight_increment, to_temporal(value, steps)
    )
    output = controller.apply(layer_index, 6, from_temporal(output_increment))
    return output.transpose(1, 2).contiguous(), flat_weights
