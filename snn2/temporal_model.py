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
    """Temporal attention, preserving legacy statistics layouts while collecting."""
    steps = int(controller.temporal_steps or 0)
    if steps <= 0:
        raise RuntimeError("Deployment attention requires a positive temporal step count")
    if module.training or float(dropout) != 0.0:
        raise RuntimeError("Temporal deployment attention requires model.eval() and dropout=0")

    query = controller.apply(layer_index, 2, query)
    groups = int(getattr(module, "num_key_value_groups", 1))
    key, value = repeat_kv(key, groups), repeat_kv(value, groups)
    current_length = int(query.shape[-2])
    past_length = max(int(key.shape[-2]) - current_length, 0)
    num_heads, head_dim = int(key.shape[1]), int(key.shape[-1])
    collecting = getattr(
        controller, "collecting_statistics", getattr(controller, "mode", None) == "collect"
    )
    collecting_saliency = getattr(
        controller, "collecting_saliency", getattr(controller, "mode", None) in {"collect", "calibration_collect"}
    )

    if collecting:
        # Sites 3/4 are attention-head grouped: record native [T*B,H,L,D],
        # then let the controller sum temporal frames to [B,H,L,D].
        statistics_key = key[..., past_length:, :] if past_length else key
        statistics_value = value[..., past_length:, :] if past_length else value
        controller.record_activation(layer_index, 3, statistics_key)
        controller.record_activation(layer_index, 4, statistics_value)
        if collecting_saliency:
            q64 = controller.logical_activation_for_calibration(query).detach().to(torch.float64)
            k64 = controller.logical_activation_for_calibration(key).detach().to(torch.float64)
            qk64 = torch.matmul(q64, k64.transpose(-2, -1))
            key_score = k64 * torch.matmul(qk64.transpose(-2, -1), q64)
            if past_length:
                key_score = key_score[..., past_length:, :]
            controller.record_saliency(layer_index, 3, key_score, source="spikellm_qk_k_fp64")
    else:
        key_flat = key.transpose(1, 2).contiguous().reshape(
            key.shape[0], key.shape[2], num_heads * head_dim
        )
        value_flat = value.transpose(1, 2).contiguous().reshape(
            value.shape[0], value.shape[2], num_heads * head_dim
        )
        key_flat = controller.apply(layer_index, 3, key_flat)
        value_flat = controller.apply(layer_index, 4, value_flat)
        key = key_flat.reshape(key_flat.shape[0], key_flat.shape[1], num_heads, head_dim).transpose(1, 2).contiguous()
        value = value_flat.reshape(value_flat.shape[0], value_flat.shape[1], num_heads, head_dim).transpose(1, 2).contiguous()

    temporal_query = to_temporal(query, steps)
    temporal_key_t = to_temporal(key.transpose(2, 3), steps)
    qk_increment = temporal_seq_matmul(temporal_query, temporal_key_t)
    scale = float(scaling if scaling is not None else getattr(module, "scaling", query.shape[-1] ** -0.5))
    score_increment = qk_increment * scale
    _record_regression(controller, f"layer_{layer_index:03d}/attn/qk_scaled", from_temporal(score_increment))
    if attention_mask is not None:
        attention_mask = attention_mask[..., : key.shape[-2]]
    weight_increment = temporal_softmax(score_increment, attention_mask, softcap=softcap)
    flat_weights = from_temporal(weight_increment)

    if collecting_saliency:
        p64 = controller.logical_activation_for_calibration(flat_weights).detach().to(torch.float64)
        v64 = controller.logical_activation_for_calibration(value).detach().to(torch.float64)
        pv64 = torch.matmul(p64, v64)
        value_score = v64 * torch.matmul(p64.transpose(-2, -1), pv64)
        if past_length:
            value_score = value_score[..., past_length:, :]
        controller.record_saliency(layer_index, 4, value_score, source="spikellm_pv_v_fp64")

    _record_regression(controller, f"layer_{layer_index:03d}/attn/softmax_before_site5", flat_weights)
    if collecting:
        statistics_weights = flat_weights[..., past_length:] if past_length else flat_weights
        controller.record_activation(layer_index, 5, statistics_weights)
    else:
        flat_weights = controller.apply(layer_index, 5, flat_weights)
    weight_increment = to_temporal(flat_weights, steps)
    output_increment = temporal_seq_matmul(weight_increment, to_temporal(value, steps))
    flat_output_heads = from_temporal(output_increment)
    _record_regression(controller, f"layer_{layer_index:03d}/attn/pv_head_output_before_merge", flat_output_heads)
    flat_output = flat_output_heads.transpose(1, 2).contiguous().reshape(
        flat_output_heads.shape[0], flat_output_heads.shape[2], flat_output_heads.shape[1] * flat_output_heads.shape[3]
    )
    _record_regression(controller, f"layer_{layer_index:03d}/attn/pv_merged_before_site6", flat_output)
    output = controller.apply(layer_index, 6, flat_output)
    return output, flat_weights
