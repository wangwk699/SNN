from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn.functional as F

from .phase_statistics import (
    MTN_BASE_SCALE_CALIBRATION,
    NEURON_PARAMETER_CLAMP_MAX,
    NEURON_PARAMETER_CLAMP_MIN,
    PARAMETER_ACCUMULATOR_DTYPE,
    PARAMETER_CALIBRATION,
    PARAMETER_CHANNEL_POLICY,
    PARAMETER_EMA_FACTOR,
    PARAMETER_REDUCTION_POLICY,
    PHASE_TAU_ACCUMULATOR_DTYPE,
    PHASE_TAU_CALIBRATION,
    PHASE_TAU_CHANNEL_POLICY,
    PHASE_TAU_EMA_FACTOR,
    PHASE_TAU_REDUCTION_POLICY,
)


TEMPORAL_IMPLEMENTATION_VERSION = 8
TEMPORAL_IMPLEMENTATION = "sparse_llm_temporal_v8_final_norm_phase_mtn_ema_clamp"
TEMPORAL_LAYOUT = "time_major_flattened_TB"
TEMPORAL_LINEAR_BIAS_POLICY = "first_timestep_once"
PREFIX_TEMPORAL_POLICY = "uniform_kv_divide_by_T"

EMBEDDING_TEMPORAL_POLICY = "uniform_embedding_divide_by_T"
SOFTMAX_PREFIX_NEURON_POLICY = "full_softmax_tensor_including_prefix"
FINAL_NORM_NEURON_POLICY = "phase_ann_surrogate_phase_snn_temporal_mtn_snn_temporal_gif_identity_clip_forbidden_v1"
SITE_STATE_FORMAT_VERSION = 10
STATISTICS_FORMAT_VERSION = 4
CALIBRATION_MANIFEST_FORMAT_VERSION = 12
CONVERSION_METADATA_FORMAT_VERSION = 13

CALIBRATION_GROUPING_POLICY = "site234_logical_per_head_site6_merged_last_dim_v2"
SOFTMAX_SITE5_GROUPING_POLICY = "per_head_full_variable_key_axis"
SOFTMAX_SITE5_GIF_POLICY = "spikellm_nbits16_sentinel_identity"
SOFTMAX_SITE5_CLIP_POLICY = "disabled"
GIF_SALIENT_POLICY = "ordinary_salient_static_qmax30"
GIF_ALL_LOW_POLICY = "all_low_static_qmax15"
GIF_IDENTITY_POLICY = "identity"
GIF_SALIENCY_SELECTION_POLICY = "spikellm_global_per_channel_threshold_leq"
GIF_SALIENCY_TIE_POLICY = "mask_low_equals_score_le_threshold"
GIF_LINEAR_SALIENCY_DTYPE = "float32"
GIF_MATMUL_SALIENCY_DTYPE = "float64"

GIF_BASE_BITS = 4
GIF_ADD_BITS = 1
GIF_LOCAL_STEPS = 2
GIF_LOW_QMIN = 0
GIF_LOW_QMAX = 15
GIF_HIGH_QMIN = 0
GIF_HIGH_QMAX = 30
GIF_STEP_QMIN = 0
GIF_STEP_QMAX = 15
GIF_INTEGER_DECOMPOSITION = (
    "two_unsigned_chunks_each_0_to_15_high_qmax_30"
)


def temporal_policy_metadata() -> dict[str, Any]:
    return {
        "temporal_implementation_version": TEMPORAL_IMPLEMENTATION_VERSION,
        "temporal_implementation": TEMPORAL_IMPLEMENTATION,
        "temporal_layout": TEMPORAL_LAYOUT,
        "temporal_linear_bias_policy": TEMPORAL_LINEAR_BIAS_POLICY,
        "prefix_temporal_policy": PREFIX_TEMPORAL_POLICY,
        "embedding_temporal_policy": EMBEDDING_TEMPORAL_POLICY,
        "softmax_prefix_neuron_policy": SOFTMAX_PREFIX_NEURON_POLICY,
        "final_norm_neuron_policy": FINAL_NORM_NEURON_POLICY,
        "phase_tau_calibration": PHASE_TAU_CALIBRATION,
        "phase_tau_ema_factor": PHASE_TAU_EMA_FACTOR,
        "phase_tau_accumulator_dtype": PHASE_TAU_ACCUMULATOR_DTYPE,
        "phase_tau_channel_policy": PHASE_TAU_CHANNEL_POLICY,
        "phase_tau_reduction_policy": PHASE_TAU_REDUCTION_POLICY,
        "calibration_grouping_policy": CALIBRATION_GROUPING_POLICY,
        "softmax_site5_grouping_policy": SOFTMAX_SITE5_GROUPING_POLICY,
        "softmax_site5_gif_policy": SOFTMAX_SITE5_GIF_POLICY,
        "softmax_site5_clip_policy": SOFTMAX_SITE5_CLIP_POLICY,
        "gif_saliency_selection_policy": GIF_SALIENCY_SELECTION_POLICY,
        "gif_saliency_tie_policy": GIF_SALIENCY_TIE_POLICY,
        "phase_parameter_calibration": PARAMETER_CALIBRATION,
        "mtn_parameter_calibration": MTN_BASE_SCALE_CALIBRATION,
        "parameter_ema_factor": PARAMETER_EMA_FACTOR,
        "parameter_accumulator_dtype": PARAMETER_ACCUMULATOR_DTYPE,
        "parameter_channel_policy": PARAMETER_CHANNEL_POLICY,
        "parameter_reduction_policy": PARAMETER_REDUCTION_POLICY,
        "parameter_clamp_min": NEURON_PARAMETER_CLAMP_MIN,
        "parameter_clamp_max": NEURON_PARAMETER_CLAMP_MAX,
        "gif_linear_saliency_dtype": GIF_LINEAR_SALIENCY_DTYPE,
        "gif_matmul_saliency_dtype": GIF_MATMUL_SALIENCY_DTYPE,
        "ordinary_gif_high_qmax": GIF_HIGH_QMAX,
        "ordinary_gif_local_decomposition_steps": GIF_LOCAL_STEPS,
        "ordinary_gif_per_step_qmax": GIF_STEP_QMAX,
        "ordinary_gif_integer_decomposition": GIF_INTEGER_DECOMPOSITION,
    }


def validate_temporal_policy(
    metadata: Mapping[str, Any], *, context: str
) -> None:
    expected = temporal_policy_metadata()
    missing = [key for key in expected if key not in metadata]
    mismatched = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if key in metadata and metadata[key] != value
    }
    if missing or mismatched:
        raise ValueError(
            f"{context} is an incompatible legacy temporal/GIF artifact; "
            "re-materialize calibration states and conversion descriptors before "
            f"SNN evaluation (missing={missing}, mismatched={mismatched})"
        )


def to_temporal(x: torch.Tensor, steps: int) -> torch.Tensor:
    steps = int(steps)
    if steps <= 0:
        raise ValueError(f"Temporal steps must be positive, got {steps}")
    if x.ndim == 0 or x.shape[0] % steps != 0:
        leading = None if x.ndim == 0 else int(x.shape[0])
        raise ValueError(
            f"Leading dimension {leading} is not divisible by temporal steps={steps}"
        )
    batch = x.shape[0] // steps
    return x.reshape(steps, batch, *x.shape[1:])


def from_temporal(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError("A temporal tensor must have leading [T, B] dimensions")
    return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])


def temporal_difference(
    x: torch.Tensor, op: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError("A temporal tensor must have leading [T, B] dimensions")
    cumulative = x.float().cumsum(dim=0)
    current = op(cumulative)
    if current.shape[:2] != x.shape[:2]:
        raise ValueError("Temporal unary operator changed the T/B dimensions")
    previous = torch.cat((torch.zeros_like(current[:1]), current[:-1]), dim=0)
    return (current - previous).to(dtype=x.dtype)


def _rmsnorm_epsilon(module: torch.nn.Module) -> float:
    for name in ("variance_epsilon", "eps"):
        value = getattr(module, name, None)
        if value is not None:
            return float(value)
    raise AttributeError(
        f"{type(module).__name__} has neither variance_epsilon nor eps"
    )


def temporal_rmsnorm(x: torch.Tensor, module: torch.nn.Module) -> torch.Tensor:
    epsilon = _rmsnorm_epsilon(module)
    weight = getattr(module, "weight", None)

    def rmsnorm(cumulative: torch.Tensor) -> torch.Tensor:
        variance = cumulative.square().mean(dim=-1, keepdim=True)
        normalized = cumulative * torch.rsqrt(variance + epsilon)
        # Qwen/Llama RMSNorm casts the normalized activation back to the
        # incoming dtype before applying the learned weight. Matching that
        # order matters at BF16 Phase thresholds.
        normalized = normalized.to(dtype=x.dtype)
        if weight is not None:
            normalized = normalized * weight.to(
                device=normalized.device, dtype=normalized.dtype
            )
        return normalized

    return temporal_difference(x, rmsnorm)


def temporal_silu(x: torch.Tensor) -> torch.Tensor:
    return temporal_difference(x, F.silu)


def temporal_seq_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape[:2] != b.shape[:2]:
        raise ValueError(
            f"Temporal matmul T/B mismatch: {tuple(a.shape[:2])} != {tuple(b.shape[:2])}"
        )
    a_float = a.float()
    b_float = b.float()
    sum_a = a_float.cumsum(dim=0)
    sum_b = b_float.cumsum(dim=0)
    output = (
        torch.matmul(sum_a, b_float)
        + torch.matmul(a_float, sum_b)
        - torch.matmul(a_float, b_float)
    )
    return output.to(dtype=a.dtype)


def _fixed_temporal_mask(
    attention_mask: torch.Tensor | None,
    *,
    steps: int,
    batch: int,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.shape[0] == steps * batch:
        temporal_mask = to_temporal(attention_mask, steps)
    elif attention_mask.shape[0] == steps and attention_mask.ndim >= 2:
        temporal_mask = attention_mask
        if temporal_mask.shape[1] != batch:
            raise ValueError("Temporal attention mask has an incompatible batch size")
    elif attention_mask.shape[0] in {1, batch}:
        return attention_mask
    else:
        raise ValueError(
            "Attention mask must be broadcastable from batch B or laid out as [T*B, ...]"
        )
    reference = temporal_mask[0]
    for timestep in range(1, steps):
        if not torch.equal(reference, temporal_mask[timestep]):
            raise ValueError("Attention mask must be identical in every temporal frame")
    return reference


def temporal_softmax(
    score_increment: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    softcap: float | None = None,
    dim: int = -1,
) -> torch.Tensor:
    if score_increment.ndim < 3:
        raise ValueError("Temporal softmax expects leading [T, B, ...] dimensions")
    steps, batch = int(score_increment.shape[0]), int(score_increment.shape[1])
    fixed_mask = _fixed_temporal_mask(
        attention_mask, steps=steps, batch=batch
    )

    def softmax(cumulative: torch.Tensor) -> torch.Tensor:
        values = cumulative
        if softcap is not None:
            cap = float(softcap)
            if cap <= 0:
                raise ValueError("softcap must be positive")
            values = torch.tanh(values / cap) * cap
        if fixed_mask is not None:
            values = values + fixed_mask.float()
        return F.softmax(values, dim=dim, dtype=torch.float32)

    return temporal_difference(score_increment, softmax)


def temporal_symmetric_hadamard(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(
            f"Temporal Hadamard inputs must have identical shapes: {a.shape} != {b.shape}"
        )
    a_float = a.float()
    b_float = b.float()
    sum_a = a_float.sum(dim=0, keepdim=True)
    sum_b = b_float.sum(dim=0, keepdim=True)
    return (0.5 * (a_float * sum_b + sum_a * b_float)).to(dtype=a.dtype)



def temporal_bias_once(
    output_with_repeated_bias: torch.Tensor,
    bias: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    temporal = to_temporal(output_with_repeated_bias, steps).clone()
    view = bias.to(device=temporal.device, dtype=temporal.dtype).view(
        *([1] * (temporal.ndim - 1)), -1
    )
    temporal[1:] = temporal[1:] - view
    return from_temporal(temporal)
