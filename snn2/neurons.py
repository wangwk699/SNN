from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .phase_statistics import (
    PHASE_TAU_ACCUMULATOR_DTYPE,
    PHASE_TAU_CALIBRATION,
    PHASE_TAU_CHANNEL_POLICY,
    PHASE_TAU_EMA_FACTOR,
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
    SOFTMAX_SITE5_GIF_POLICY,
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


def _state_layout(state: dict[str, Any]) -> dict[str, Any]:
    required = {
        "parameter_layout", "configured_group_size", "group_size",
        "num_heads", "channels_per_head", "groups_per_head",
    }
    missing = sorted(required - state.keys())
    if missing:
        raise ValueError(f"Grouped state is missing layout metadata: {missing}")
    configured = int(state["configured_group_size"])
    if configured != -1 and configured <= 0:
        raise ValueError("configured_group_size must be -1 or positive")
    layout = {key: state[key] for key in required}
    kind = layout["parameter_layout"]
    group = int(layout["group_size"])
    groups = int(layout["groups_per_head"])
    heads = layout["num_heads"]
    width = layout["channels_per_head"]
    if kind == "last_dim_grouped":
        if heads is not None or not isinstance(width, int) or group <= 0 or groups != width // group or width % group:
            raise ValueError("Invalid last_dim_grouped metadata")
    elif kind == "attention_head_grouped":
        if not isinstance(heads, int) or heads <= 0 or not isinstance(width, int) or width <= 0 or group <= 0 or width % group or groups != width // group:
            raise ValueError("Invalid attention_head_grouped metadata")
    elif kind == "attention_head_scalar":
        if not isinstance(heads, int) or heads <= 0 or width is not None or group != -1 or groups != 1:
            raise ValueError("Invalid attention_head_scalar metadata")
    else:
        raise ValueError(f"Unsupported parameter_layout={kind!r}")
    return layout


def _expected_parameter_shape(layout: dict[str, Any]) -> tuple[int, ...]:
    if layout["parameter_layout"] == "last_dim_grouped":
        return (int(layout["groups_per_head"]),)
    return (int(layout["num_heads"]), int(layout["groups_per_head"]))


def _require_parameter_shape(name: str, value: torch.Tensor, layout: dict[str, Any]) -> None:
    expected = _expected_parameter_shape(layout)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
        actual = None if not isinstance(value, torch.Tensor) else tuple(value.shape)
        raise ValueError(f"{name} shape must be {expected}, got {actual}")


def _parameter_values(x: torch.Tensor, values: torch.Tensor, layout: dict[str, Any]) -> torch.Tensor:
    kind = layout["parameter_layout"]
    group_size = int(layout["group_size"])
    groups = int(layout["groups_per_head"])
    if kind == "last_dim_grouped":
        width = int(layout["channels_per_head"])
        if x.shape[-1] != width or values.shape != (groups,):
            raise ValueError(
                f"last_dim_grouped shape mismatch: runtime={tuple(x.shape)}, "
                f"parameter={tuple(values.shape)}, expected width/groups={(width, groups)}"
            )
        expanded = values.repeat_interleave(group_size)
        return expanded.to(x).view(*([1] * (x.ndim - 1)), width)
    if kind == "attention_head_grouped":
        heads = int(layout["num_heads"])
        width = int(layout["channels_per_head"])
        if x.ndim != 4 or x.shape[1] != heads or x.shape[-1] != width:
            raise ValueError(
                f"attention_head_grouped runtime shape mismatch: got {tuple(x.shape)}, "
                f"expected [B,{heads},L,{width}]"
            )
        if values.shape != (heads, groups):
            raise ValueError(f"Expected grouped parameter shape {(heads, groups)}, got {tuple(values.shape)}")
        return values.repeat_interleave(group_size, dim=-1).to(x).view(1, heads, 1, width)
    if kind == "attention_head_scalar":
        heads = int(layout["num_heads"])
        if x.ndim != 4 or x.shape[1] != heads or values.shape != (heads, 1):
            raise ValueError(
                f"attention_head_scalar mismatch: runtime={tuple(x.shape)}, "
                f"parameter={tuple(values.shape)}, expected heads={heads}"
            )
        return values.to(x).view(1, heads, 1, 1)
    raise ValueError(f"Unsupported parameter_layout={kind!r}")


def _mask_values(x: torch.Tensor, mask: torch.Tensor, layout: dict[str, Any]) -> torch.Tensor:
    kind = layout["parameter_layout"]
    if kind == "last_dim_grouped":
        expected = (int(layout["channels_per_head"]),)
        if mask.shape != expected or x.shape[-1] != expected[0]:
            raise ValueError(f"GIF mask shape mismatch: expected {expected}, got {tuple(mask.shape)}")
        return mask.to(device=x.device).view(*([1] * (x.ndim - 1)), expected[0])
    if kind == "attention_head_grouped":
        expected = (int(layout["num_heads"]), int(layout["channels_per_head"]))
        if x.ndim != 4 or mask.shape != expected or (x.shape[1], x.shape[-1]) != expected:
            raise ValueError(f"GIF attention mask shape mismatch: expected {expected}, got {tuple(mask.shape)}")
        return mask.to(device=x.device).view(1, expected[0], 1, expected[1])
    raise ValueError(f"GIF mask does not support parameter_layout={kind!r}")


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
            state.get("tau_calibration") != PHASE_TAU_CALIBRATION
            or float(state.get("tau_ema_factor", -1.0)) != PHASE_TAU_EMA_FACTOR
            or state.get("tau_accumulator_dtype") != PHASE_TAU_ACCUMULATOR_DTYPE
            or state.get("tau_channel_policy") != PHASE_TAU_CHANNEL_POLICY
            or state.get("tau_reduction_policy") != PHASE_TAU_REDUCTION_POLICY
        ):
            raise ValueError(
                "Incompatible grouped Phase tau calibration; re-run calibration"
            )
        self.T = int(state["T"])
        self.base = float(state["base"])
        self.layout = _state_layout(state)
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
        _require_parameter_shape("Phase tau", self.tau, self.layout)
        if self.tau.shape != self.v0.shape:
            raise ValueError("Phase tau and v0 shapes must match")

    def encode(self, x: torch.Tensor, return_temporal: bool) -> torch.Tensor:
        sign = x.sign().detach()
        tau = _parameter_values(x, self.tau, self.layout)
        v0 = _parameter_values(x, self.v0, self.layout)
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
        if state.get("gif_policy") != "ordinary_grouped_qmax30":
            raise ValueError("StaticGIF only accepts ordinary grouped GIF states")
        self.layout = _state_layout(state)
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
        for name in ("low_scale", "low_zero", "high_scale", "high_zero"):
            _require_parameter_shape(f"GIF {name}", getattr(self, name), self.layout)
        expected_mask = (
            (int(self.layout["channels_per_head"]),)
            if self.layout["parameter_layout"] == "last_dim_grouped"
            else (int(self.layout["num_heads"]), int(self.layout["channels_per_head"]))
        )
        if tuple(self.mask_low.shape) != expected_mask:
            raise ValueError(f"GIF mask_low shape must be {expected_mask}, got {tuple(self.mask_low.shape)}")

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
        scale = _parameter_values(x, scale_values, self.layout).clamp_min(1e-8)
        zero = _parameter_values(x, zero_values, self.layout)
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
        mask = _mask_values(x, self.mask_low, self.layout)
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
        mask = _mask_values(x, self.mask_low, self.layout)
        scale_low = _parameter_values(x, self.low_scale, self.layout)
        scale_high = _parameter_values(x, self.high_scale, self.layout)
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
        self.layout = _state_layout(state)
        self.threshold_factor = float(state.get("threshold_factor", 0.75))
        self.register_buffer("base_scale", state["base_scale"].float())
        _require_parameter_shape("MTN base_scale", self.base_scale, self.layout)


    def temporal(self, incoming: torch.Tensor) -> torch.Tensor:
        if incoming.shape[0] != self.T:
            raise ValueError(f"MTN expects T={self.T}, got {incoming.shape[0]}")
        scale = _parameter_values(incoming[0], self.base_scale, self.layout)
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
            "ordinary_gif_high_qmax": GIF_HIGH_QMAX,
            "ordinary_gif_per_step_qmax": GIF_STEP_QMAX,
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
        self.layout = _state_layout(state)
        self.register_buffer("lower", state["lower"].float())
        self.register_buffer("upper", state["upper"].float())
        _require_parameter_shape("Clip lower", self.lower, self.layout)
        _require_parameter_shape("Clip upper", self.upper, self.layout)
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
        lower = _parameter_values(x, self.lower, self.layout)
        upper = _parameter_values(x, self.upper, self.layout)
        return hard_clip(x, lower, upper)


class SoftmaxIdentityGIF(nn.Module):
    def __init__(self, state: dict[str, Any]):
        super().__init__()
        _validate_state_header(state, "gif")
        expected = {
            "parameter_layout": "softmax_identity",
            "gif_policy": SOFTMAX_SITE5_GIF_POLICY,
            "group_size": -1,
            "group_size_source": "site5_identity_override",
            "reference_n_bits": 16,
            "reference_metric": "fix0to1",
            "quantization_applied": False,
            "temporal_steps": GIF_LOCAL_STEPS,
            "temporal_policy": "identity",
        }
        mismatched = {k: (v, state.get(k)) for k, v in expected.items() if state.get(k) != v}
        forbidden = sorted(
            key
            for key in (
                "range_min", "range_max", "quantization_bits", "qmin",
                "qmax", "scale", "zero_point",
            )
            if key in state
        )
        if mismatched or forbidden:
            raise ValueError(
                "Invalid SpikeLLM-aligned Site 5 GIF identity state: "
                f"mismatched={mismatched}, forbidden={forbidden}"
            )
        self.num_heads = int(state["num_heads"])
        self._temporal_steps = int(state["temporal_steps"])
        if self.num_heads <= 0 or int(state.get("configured_group_size", 0)) == 0:
            raise ValueError("Site 5 GIF requires valid head/group metadata")

    @property
    def temporal_steps(self) -> int:
        return self._temporal_steps

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4 or x.shape[1] != self.num_heads:
            raise ValueError(
                f"SoftmaxIdentityGIF expects [B,{self.num_heads},Q,K], got {tuple(x.shape)}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        return x

    def temporal(self, incoming: torch.Tensor) -> torch.Tensor:
        if incoming.ndim != 5 or incoming.shape[0] != self.temporal_steps:
            raise ValueError(
                f"SoftmaxIdentityGIF expects [T,B,H,Q,K] with T={self.temporal_steps}, "
                f"got {tuple(incoming.shape)}"
            )
        self._validate_input(incoming[0])
        return incoming


def gif_module_from_state(state: dict[str, Any]) -> nn.Module:
    if state.get("gif_policy") == SOFTMAX_SITE5_GIF_POLICY:
        return SoftmaxIdentityGIF(state)
    return StaticGIF(state)
