from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .phase_statistics import (
    PHASE_STATISTICAL_VIEW,
    PHASE_STATISTICAL_VIEW_VERSION,
    PHASE_TAU_CHANNEL_POLICY,
    PHASE_TAU_REDUCTION_POLICY,
)

from .temporal_ops import (
    GIF_ADD_BITS,
    GIF_BASE_BITS,
    GIF_HIGH_QMAX,
    GIF_INTEGER_DECOMPOSITION,
    GIF_LOCAL_STEPS,
    GIF_LOW_QMAX,
    GIF_STEP_QMAX,
    SITE_STATE_FORMAT_VERSION,
    TEMPORAL_IMPLEMENTATION_VERSION,
)


def gif_high_qmax(base_bits: int, add_bits: int) -> int:
    """Return the only GIF integer range supported by this experiment."""
    if int(base_bits) != GIF_BASE_BITS or int(add_bits) != GIF_ADD_BITS:
        raise ValueError(
            f"GIF requires base_bits={GIF_BASE_BITS}, add_bits={GIF_ADD_BITS}"
        )
    return GIF_HIGH_QMAX


def _validate_state_header(state: dict[str, Any], state_kind: str) -> None:
    if (
        state.get("state_kind") != state_kind
        or state.get("format_version") != SITE_STATE_FORMAT_VERSION
        or state.get("temporal_implementation_version")
        != TEMPORAL_IMPLEMENTATION_VERSION
    ):
        raise ValueError(
            f"Incompatible legacy {state_kind} state; expected state_kind={state_kind}, "
            f"format_version={SITE_STATE_FORMAT_VERSION}, "
            f"temporal_implementation_version={TEMPORAL_IMPLEMENTATION_VERSION}. "
            "Re-materialize calibration states before training/conversion/evaluation."
        )


def _channel_values(x: torch.Tensor, values: torch.Tensor, group_size: int) -> torch.Tensor:
    if group_size <= 0 and values.numel() == 1:
        return values.to(device=x.device, dtype=x.dtype).view(*([1] * x.ndim))
    if x.shape[-1] % group_size != 0:
        raise ValueError(f"Last dimension {x.shape[-1]} is not divisible by group_size={group_size}")
    groups = x.shape[-1] // group_size
    if values.numel() != groups:
        raise ValueError(f"Expected {groups} group values, got {values.numel()}")
    expanded = values.to(device=x.device, dtype=x.dtype).repeat_interleave(group_size)
    return expanded.view(*([1] * (x.ndim - 1)), x.shape[-1])


class HeavisideSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, slope: float):
        ctx.save_for_backward(x)
        ctx.slope = float(slope)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        slope = ctx.slope
        sigma = torch.sigmoid(slope * x)
        return grad_output * slope * sigma * (1.0 - sigma), None


def hard_clip(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.minimum(x, upper), lower)


class PhaseSurrogate(nn.Module):
    def __init__(
        self,
        state: dict[str, Any],
        *,
        surrogate_slope: float | None = None,
    ):
        super().__init__()
        _validate_state_header(state, "phase")
        if "surrogate_slope" in state:
            raise ValueError(
                "Legacy Phase state contains surrogate_slope; re-run calibration"
            )
        if (
            state.get("tau_calibration") != "spikingllm_ema_channel_abs_max"
            or float(state.get("tau_ema_factor", -1.0)) != 0.99
            or state.get("tau_accumulator_dtype") != "float32"
            or state.get("tau_channel_policy") != PHASE_TAU_CHANNEL_POLICY
            or state.get("tau_reduction_policy") != PHASE_TAU_REDUCTION_POLICY
            or state.get("phase_statistical_view") != PHASE_STATISTICAL_VIEW
            or state.get("phase_statistical_view_version")
            != PHASE_STATISTICAL_VIEW_VERSION
        ):
            raise ValueError(
                "Incompatible Phase tau calibration; SpikingLLM EMA factor 0.99 is required"
            )
        self.T = int(state["T"])
        self.base = float(state["base"])
        self.group_size = int(state["group_size"])
        if self.group_size != -1 or state["tau"].numel() != 1:
            raise ValueError("SpikingLLM-aligned Phase requires scalar tau and group_size=-1")
        self.slope = (
            None if surrogate_slope is None else float(surrogate_slope)
        )
        if self.slope is not None and (
            not math.isfinite(self.slope) or self.slope <= 0.0
        ):
            raise ValueError("Phase surrogate_slope must be a positive finite number")
        self.max_spikes = int(state.get("max_spikes", 2))
        self.register_buffer("tau", state["tau"].float())
        self.register_buffer("v0", state["v0"].float())

    def encode(self, x: torch.Tensor, return_temporal: bool) -> torch.Tensor:
        sign = x.sign().detach()
        tau = _channel_values(x, self.tau, self.group_size)
        v0 = _channel_values(x, self.v0, self.group_size)
        membrane = x.abs() + v0
        spike_count = torch.zeros_like(x)
        outputs = []
        for timestep in range(self.T):
            amplitude = tau * (self.base ** (-(timestep + 1)))
            distance = membrane - amplitude
            spike = (
                (distance > 0).to(distance.dtype)
                if self.slope is None
                else HeavisideSigmoid.apply(distance, self.slope)
            )
            if self.max_spikes > 0:
                spike = spike * (spike_count < self.max_spikes).to(spike.dtype)
            spike_count = spike_count + spike.detach()
            outputs.append(sign * amplitude * spike)
            membrane = membrane - amplitude * spike
        temporal = torch.stack(outputs, dim=0)
        return temporal if return_temporal else temporal.sum(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x, return_temporal=False)


    def temporal(self, incoming: torch.Tensor) -> torch.Tensor:
        if incoming.shape[0] != self.T:
            raise ValueError(f"Phase expects T={self.T}, got {incoming.shape[0]}")
        return self.encode(incoming.sum(dim=0), return_temporal=True)


class StaticGIF(nn.Module):
    def __init__(self, state: dict[str, Any]):
        super().__init__()
        _validate_state_header(state, "gif")
        self.base_bits = int(state["base_bits"])
        self.add_bits = int(state["add_bits"])
        self.group_size = int(state["group_size"])
        self.high_qmax = gif_high_qmax(self.base_bits, self.add_bits)
        expected_policy = {
            "low_qmin": 0,
            "low_qmax": GIF_LOW_QMAX,
            "high_qmin": 0,
            "high_qmax": GIF_HIGH_QMAX,
            "temporal_steps": GIF_LOCAL_STEPS,
            "per_step_qmin": 0,
            "per_step_qmax": GIF_STEP_QMAX,
            "integer_decomposition": GIF_INTEGER_DECOMPOSITION,
        }
        mismatched = {
            key: (expected, state.get(key))
            for key, expected in expected_policy.items()
            if key not in state or state[key] != expected
        }
        if mismatched:
            raise ValueError(
                "Incompatible legacy GIF qmax/chunk policy; re-materialize calibration "
                f"states (mismatched={mismatched})"
            )
        self.register_buffer("low_scale", state["low_scale"].float())
        self.register_buffer("low_zero", state["low_zero"].float())
        self.register_buffer("high_scale", state["high_scale"].float())
        self.register_buffer("high_zero", state["high_zero"].float())
        self.register_buffer("mask_low", state["mask_low"].bool())

    @staticmethod
    def round_ste(x: torch.Tensor) -> torch.Tensor:
        return (x.round() - x).detach() + x

    def _quantize(
        self,
        x: torch.Tensor,
        scale_values: torch.Tensor,
        zero_values: torch.Tensor,
        *,
        qmin: int,
        qmax: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = _channel_values(x, scale_values, self.group_size).clamp_min(1e-8)
        zero = _channel_values(x, zero_values, self.group_size)
        qmin, qmax = int(qmin), int(qmax)
        if qmin != 0 or qmax <= qmin:
            raise ValueError(f"Invalid unsigned GIF range [{qmin}, {qmax}]")
        q = (self.round_ste(x.float() / scale.float()) + zero.float()).clamp(
            qmin, qmax
        )
        dequantized = (q - zero.float()) * scale.float()
        return dequantized.to(x.dtype), q, zero.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low, _, _ = self._quantize(
            x, self.low_scale, self.low_zero, qmin=0, qmax=GIF_LOW_QMAX
        )
        high, _, _ = self._quantize(
            x,
            self.high_scale,
            self.high_zero,
            qmin=0,
            qmax=self.high_qmax,
        )
        mask = self.mask_low.to(device=x.device)
        if mask.numel() != x.shape[-1]:
            if not bool(self.mask_low.numel() >= x.shape[-1]):
                padding = torch.ones(
                    x.shape[-1] - mask.numel(), device=x.device, dtype=torch.bool
                )
                mask = torch.cat((mask, padding))
            else:
                mask = mask[: x.shape[-1]]
        mask = mask.view(*([1] * (x.ndim - 1)), x.shape[-1])
        return torch.where(mask, low, high)

    @property
    def temporal_steps(self) -> int:
        return GIF_LOCAL_STEPS

    @staticmethod
    def integer_chunks(q_high: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.any(q_high < 0) or torch.any(q_high > GIF_HIGH_QMAX):
            raise ValueError(f"GIF high integer code must be in [0, {GIF_HIGH_QMAX}]")
        chunk0 = q_high.clamp(0, GIF_STEP_QMAX)
        chunk1 = q_high - chunk0
        if (
            torch.any(chunk0 < 0)
            or torch.any(chunk0 > GIF_STEP_QMAX)
            or torch.any(chunk1 < 0)
            or torch.any(chunk1 > GIF_STEP_QMAX)
            or torch.any(chunk0 + chunk1 != q_high)
        ):
            raise RuntimeError("GIF two-step integer decomposition invariant failed")
        return chunk0, chunk1

    def temporal(self, incoming: torch.Tensor) -> torch.Tensor:
        if incoming.shape[0] != self.temporal_steps:
            raise ValueError(
                f"GIF expects T={self.temporal_steps}, got {incoming.shape[0]}"
            )
        x = incoming.sum(dim=0)
        _, low_q, low_zero = self._quantize(
            x, self.low_scale, self.low_zero, qmin=0, qmax=GIF_LOW_QMAX
        )
        _, high_q, high_zero = self._quantize(
            x,
            self.high_scale,
            self.high_zero,
            qmin=0,
            qmax=self.high_qmax,
        )
        mask = self.mask_low.to(device=x.device)
        if mask.numel() != x.shape[-1]:
            if mask.numel() < x.shape[-1]:
                mask = torch.cat(
                    (
                        mask,
                        torch.ones(
                            x.shape[-1] - mask.numel(), device=x.device, dtype=torch.bool
                        ),
                    )
                )
            else:
                mask = mask[: x.shape[-1]]
        mask = mask.view(*([1] * (x.ndim - 1)), x.shape[-1])
        scale_low = _channel_values(x, self.low_scale, self.group_size)
        scale_high = _channel_values(x, self.high_scale, self.group_size)
        high_chunks = self.integer_chunks(high_q)
        outputs = []
        for timestep, chunk in enumerate(high_chunks):
            high_output = chunk * scale_high
            if timestep == 0:
                high_output = high_output - high_zero * scale_high
                low_output = (low_q - low_zero) * scale_low
            else:
                low_output = torch.zeros_like(x, dtype=torch.float32)
            outputs.append(torch.where(mask, low_output, high_output))
        return torch.stack(outputs, dim=0).to(x.dtype)


class MultiThresholdNeuron(nn.Module):
    def __init__(self, state: dict[str, Any]):
        super().__init__()
        _validate_state_header(state, "mtn")
        self.T = int(state["T"])
        self.K = int(state["K"])
        self.group_size = int(state["group_size"])
        self.threshold_factor = float(state.get("threshold_factor", 0.75))
        self.register_buffer("base_scale", state["base_scale"].float())


    def temporal(self, incoming: torch.Tensor) -> torch.Tensor:
        if incoming.shape[0] != self.T:
            raise ValueError(f"MTN expects T={self.T}, got {incoming.shape[0]}")
        scale = _channel_values(incoming[0], self.base_scale, self.group_size)
        levels = torch.arange(self.K, device=incoming.device, dtype=incoming.dtype)
        shape = (self.K,) + (1,) * incoming[0].ndim
        thresholds = scale.unsqueeze(0) / torch.pow(2.0, levels).view(shape)
        firing = thresholds * self.threshold_factor
        membrane = torch.zeros_like(incoming[0])
        outputs = []
        for timestep in range(self.T):
            membrane = membrane + incoming[timestep]
            sign = membrane.sign().detach()
            magnitude = membrane.abs()
            all_fired = (magnitude.unsqueeze(0) >= firing).to(incoming.dtype)
            selected = all_fired.clone()
            selected[1:] = selected[1:] - all_fired[:-1]
            spike = (thresholds * selected * sign.unsqueeze(0)).sum(dim=0)
            membrane = membrane - spike
            outputs.append(spike)
        return torch.stack(outputs, dim=0)


class Clipper(nn.Module):
    def __init__(self, state: dict[str, Any]):
        super().__init__()
        _validate_state_header(state, "clip")
        expected = {
            "gif_high_qmax": GIF_HIGH_QMAX,
            "gif_per_step_qmax": GIF_STEP_QMAX,
            "gif_integer_decomposition": GIF_INTEGER_DECOMPOSITION,
        }
        mismatched = {
            key: (value, state.get(key))
            for key, value in expected.items()
            if key not in state or state[key] != value
        }
        if mismatched:
            raise ValueError(
                "Incompatible legacy Clip/GIF range metadata; re-materialize "
                f"calibration states (mismatched={mismatched})"
            )
        self.group_size = int(state["group_size"])
        self.register_buffer("lower", state["lower"].float())
        self.register_buffer("upper", state["upper"].float())
        gif_ranges = {}
        for range_name in ("gif_low_range", "gif_high_range"):
            raw_range = state.get(range_name)
            if not isinstance(raw_range, (tuple, list)) or len(raw_range) != 2:
                raise ValueError(f"Clip state is missing valid {range_name} metadata")
            range_lower = raw_range[0].float()
            range_upper = raw_range[1].float()
            if (
                range_lower.shape != self.lower.shape
                or range_upper.shape != self.upper.shape
                or torch.any(range_lower >= range_upper)
            ):
                raise ValueError(f"Clip state has invalid {range_name} metadata")
            gif_ranges[range_name] = (range_lower, range_upper)
        gif_lower = torch.maximum(
            gif_ranges["gif_low_range"][0], gif_ranges["gif_high_range"][0]
        )
        gif_upper = torch.minimum(
            gif_ranges["gif_low_range"][1], gif_ranges["gif_high_range"][1]
        )
        tolerance = 1e-6
        if torch.any(self.lower < gif_lower - tolerance) or torch.any(
            self.upper > gif_upper + tolerance
        ):
            raise ValueError("Clip interval is inconsistent with saved GIF ranges")
        if torch.any(self.lower >= self.upper):
            raise ValueError("Every clipping interval must satisfy lower < upper")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lower = _channel_values(x, self.lower, self.group_size)
        upper = _channel_values(x, self.upper, self.group_size)
        return hard_clip(x, lower, upper)
