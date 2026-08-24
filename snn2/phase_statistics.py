from __future__ import annotations

import torch


PHASE_STATISTICAL_VIEW = "spikingllm_identity_input_layout"
PHASE_STATISTICAL_VIEW_VERSION = 1
PHASE_TAU_CHANNEL_POLICY = (
    "spikingllm_flatten_attention_heads_before_channel_ema"
)
PHASE_TAU_REDUCTION_POLICY = "per_channel_ema_then_global_max"


def phase_statistical_view(site_index: int, x: torch.Tensor) -> torch.Tensor:
    """Return the SpikingLLM Identity-input channel layout for Phase EMA only."""
    if site_index not in {2, 3, 4, 5, 6}:
        return x
    if x.ndim != 4:
        raise ValueError(
            f"Phase statistical view for site {site_index} expects a 4-D tensor, "
            f"got shape={tuple(x.shape)}"
        )
    batch, heads, first_length, last = x.shape
    if site_index == 5:
        return x.permute(0, 2, 3, 1).contiguous().reshape(
            batch, first_length * last, heads
        )
    return x.permute(0, 2, 1, 3).contiguous().reshape(
        batch, first_length, heads * last
    )
