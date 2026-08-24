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


def _record_regression(controller: SiteController, name: str, value: torch.Tensor) -> None:
    recorder = getattr(controller, "regression_recorder", None)
    if recorder is not None:
        recorder.record(name, value, temporal=True)


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

    query = controller.apply(layer_index, 2, query)
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
    _record_regression(
        controller,
        f"layer_{layer_index:03d}/attn/qk_scaled",
        from_temporal(score_increment),
    )
    if attention_mask is not None:
        attention_mask = attention_mask[..., : key.shape[-2]]
    weight_increment = temporal_softmax(
        score_increment,
        attention_mask,
        softcap=softcap,
    )
    flat_weights = from_temporal(weight_increment)
    _record_regression(
        controller,
        f"layer_{layer_index:03d}/attn/softmax_before_site5",
        flat_weights,
    )
    flat_weights = controller.apply(layer_index, 5, flat_weights)
    weight_increment = to_temporal(flat_weights, steps)
    output_increment = temporal_seq_matmul(
        weight_increment, to_temporal(value, steps)
    )
    flat_output = from_temporal(output_increment)
    _record_regression(
        controller, f"layer_{layer_index:03d}/attn/pv_before_site6", flat_output
    )
    output = controller.apply(layer_index, 6, flat_output)
    return output.transpose(1, 2).contiguous(), flat_weights
