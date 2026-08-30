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
    GIF_SALIENT_POLICY,
    GIF_ALL_LOW_POLICY,
    GIF_IDENTITY_POLICY,
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
        native = x.ndim == 4 and x.shape[1] == heads and x.shape[-1] == width
        merged = x.ndim == 3 and x.shape[-1] == heads * width
        if not native and not merged:
            raise ValueError(
                f"attention_head_grouped runtime shape mismatch: got {tuple(x.shape)}, "
                f"expected [B,{heads},L,{width}] or [B,L,{heads * width}]"
            )
        if values.shape != (heads, groups):
            raise ValueError(f"Expected grouped parameter shape {(heads, groups)}, got {tuple(values.shape)}")
        expanded = values.repeat_interleave(group_size, dim=-1).to(x)
        return (
            expanded.view(1, heads, 1, width)
            if native else expanded.reshape(1, 1, heads * width)
        )
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
        native = x.ndim == 4 and (x.shape[1], x.shape[-1]) == expected
        merged = x.ndim == 3 and x.shape[-1] == expected[0] * expected[1]
        if mask.shape != expected or (not native and not merged):
            raise ValueError(f"GIF attention mask shape mismatch: expected {expected}, got {tuple(mask.shape)}")
        mask = mask.to(device=x.device)
        return (
            mask.view(1, expected[0], 1, expected[1])
            if native else mask.reshape(1, 1, expected[0] * expected[1])
        )
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
        T: int,
        surrogate_slope: float | None = None,
    ):
        super().__init__()
        _validate_state_header(state, "phase")
        forbidden = {"T", "base", "max_spikes", "v0", "surrogate_slope"} & state.keys()
        if forbidden:
            raise ValueError(
                f"Legacy pre-A/B Phase state contains runtime fields {sorted(forbidden)}; re-run Stage A"
            )
        if (
            state.get("tau_calibration") != PHASE_TAU_CALIBRATION
            or float(state.get("tau_ema_factor", -1.0)) != PHASE_TAU_EMA_FACTOR
            or state.get("tau_accumulator_dtype") != PHASE_TAU_ACCUMULATOR_DTYPE
            or state.get("tau_channel_policy") != PHASE_TAU_CHANNEL_POLICY
            or state.get("tau_reduction_policy") != PHASE_TAU_REDUCTION_POLICY
        ):
            raise ValueError("Incompatible grouped Phase tau calibration; re-run Stage A")
        self.T = int(T)
        if self.T <= 0:
            raise ValueError("Phase T must be positive")
        self.layout = _state_layout(state)
        self.slope = None if surrogate_slope is None else float(surrogate_slope)
        if self.slope is not None and (not math.isfinite(self.slope) or self.slope <= 0.0):
            raise ValueError("Phase surrogate_slope must be a positive finite number")
        self.register_buffer("tau", state["tau"].float())
        self.register_buffer("v0", (0.5 * self.tau * 2.0 ** (-self.T)).float())
        _require_parameter_shape("Phase tau", self.tau, self.layout)

    def encode(self, x: torch.Tensor, return_temporal: bool) -> torch.Tensor:
        sign = x.sign().detach()
        tau = _parameter_values(x, self.tau, self.layout)
        v0 = _parameter_values(x, self.v0, self.layout)
        membrane = x.abs() + v0
        outputs = []
        for timestep in range(self.T):
            amplitude = tau * 2.0 ** (-(timestep + 1))
            distance = membrane - amplitude
            spike = (
                (distance > 0).to(distance.dtype)
                if self.slope is None
                else HeavisideSigmoid.apply(distance, self.slope)
            )
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
        if state.get("gif_policy") != GIF_SALIENT_POLICY:
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
        self.mask_policy = state.get("mask_policy", "single")
        self.mask_roles = tuple(state.get("mask_roles", ()))
        if self.mask_policy == "multi_role":
            masks = state.get("mask_low_by_role")
            if not isinstance(masks, dict) or set(masks) != set(self.mask_roles):
                raise ValueError("Invalid multi-role GIF mask state")
            self._mask_buffer_names = {}
            for role, mask in masks.items():
                name = f"mask_low_role_{role}"
                self.register_buffer(name, mask.bool())
                self._mask_buffer_names[role] = name
        else:
            self.register_buffer("mask_low", state["mask_low"].bool())
        for name in ("low_scale", "low_zero", "high_scale", "high_zero"):
            _require_parameter_shape(f"GIF {name}", getattr(self, name), self.layout)
        expected_mask = (
            (int(self.layout["channels_per_head"]),)
            if self.layout["parameter_layout"] == "last_dim_grouped"
            else (int(self.layout["num_heads"]), int(self.layout["channels_per_head"]))
        )
        masks = (
            [getattr(self, name) for name in self._mask_buffer_names.values()]
            if self.mask_policy == "multi_role" else [self.mask_low]
        )
        if any(tuple(mask.shape) != expected_mask for mask in masks):
            raise ValueError(f"GIF masks must have shape {expected_mask}")

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

    def _mask(self, role: str | None) -> torch.Tensor:
        if self.mask_policy == "multi_role":
            if role not in self._mask_buffer_names:
                raise ValueError(f"GIF role must be one of {self.mask_roles}, got {role!r}")
            return getattr(self, self._mask_buffer_names[role])
        if role is not None:
            raise ValueError("Single-mask GIF does not accept a role")
        return self.mask_low

    def forward(self, x: torch.Tensor, *, role: str | None = None) -> torch.Tensor:
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
        mask = _mask_values(x, self._mask(role), self.layout)
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

    def temporal(self, incoming: torch.Tensor, *, role: str | None = None) -> torch.Tensor:
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
        mask = _mask_values(x, self._mask(role), self.layout)
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
    def __init__(
        self,
        state: dict[str, Any],
        *,
        T: int,
        K: int,
        threshold_factor: float,
    ):
        super().__init__()
        _validate_state_header(state, "mtn")
        forbidden = {"T", "K", "threshold_factor"} & state.keys()
        if forbidden:
            raise ValueError(
                f"Legacy pre-A/B MTN state contains runtime fields {sorted(forbidden)}; re-run Stage A"
            )
        self.T, self.K = int(T), int(K)
        self.threshold_factor = float(threshold_factor)
        if self.T <= 0 or self.K <= 0:
            raise ValueError("MTN T and K must be positive")
        if not math.isfinite(self.threshold_factor) or self.threshold_factor <= 0.0:
            raise ValueError("MTN threshold_factor must be positive and finite")
        self.layout = _state_layout(state)
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
        mismatched = {key: (value, state.get(key)) for key, value in expected.items() if state.get(key) != value}
        if mismatched:
            raise ValueError(f"Incompatible Stage B Clip metadata: {mismatched}; re-run Stage B")
        self.layout = _state_layout(state)
        self.role_policy = state.get("clip_role_policy")
        self.roles = tuple(state.get("clip_roles", ()))
        if self.role_policy == "role_specific":
            lowers, uppers = state.get("lower_by_role"), state.get("upper_by_role")
            if not isinstance(lowers, dict) or not isinstance(uppers, dict) or set(lowers) != set(self.roles) or set(uppers) != set(self.roles):
                raise ValueError("Role-specific Clip state has incomplete role intervals")
            self._role_buffers = {}
            for role in self.roles:
                lower_name, upper_name = f"lower_role_{role}", f"upper_role_{role}"
                self.register_buffer(lower_name, lowers[role].float())
                self.register_buffer(upper_name, uppers[role].float())
                _require_parameter_shape(f"Clip {role} lower", getattr(self, lower_name), self.layout)
                _require_parameter_shape(f"Clip {role} upper", getattr(self, upper_name), self.layout)
                if torch.any(getattr(self, lower_name) >= getattr(self, upper_name)):
                    raise ValueError(f"Every {role} clipping interval must satisfy lower < upper")
                self._role_buffers[role] = (lower_name, upper_name)
        elif self.role_policy == "single":
            self.register_buffer("lower", state["lower"].float())
            self.register_buffer("upper", state["upper"].float())
            _require_parameter_shape("Clip lower", self.lower, self.layout)
            _require_parameter_shape("Clip upper", self.upper, self.layout)
            if torch.any(self.lower >= self.upper):
                raise ValueError("Every clipping interval must satisfy lower < upper")
        else:
            raise ValueError("Clip state must declare clip_role_policy single or role_specific")

    def forward(self, x: torch.Tensor, *, role: str | None = None) -> torch.Tensor:
        if self.role_policy == "role_specific":
            if role not in self._role_buffers:
                raise ValueError(f"Clip role must be one of {self.roles}, got {role!r}")
            lower_name, upper_name = self._role_buffers[role]
            lower_values, upper_values = getattr(self, lower_name), getattr(self, upper_name)
        else:
            if role is not None:
                raise ValueError("Single-role Clip does not accept a role")
            lower_values, upper_values = self.lower, self.upper
        lower = _parameter_values(x, lower_values, self.layout)
        upper = _parameter_values(x, upper_values, self.layout)
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

    def forward(self, x: torch.Tensor, *, role: str | None = None) -> torch.Tensor:
        if role is not None:
            raise ValueError("Site 5 GIF does not accept a role")
        self._validate_input(x)
        return x

    def temporal(self, incoming: torch.Tensor, *, role: str | None = None) -> torch.Tensor:
        if role is not None:
            raise ValueError("Site 5 GIF does not accept a role")
        if incoming.ndim != 5 or incoming.shape[0] != self.temporal_steps:
            raise ValueError(
                f"SoftmaxIdentityGIF expects [T,B,H,Q,K] with T={self.temporal_steps}, "
                f"got {tuple(incoming.shape)}"
            )
        self._validate_input(incoming[0])
        return incoming


class AllLowStaticGIF(StaticGIF):
    def __init__(self, state: dict[str, Any]):
        nn.Module.__init__(self)
        _validate_state_header(state, "gif")
        expected = {
            "gif_policy": GIF_ALL_LOW_POLICY,
            "base_bits": GIF_BASE_BITS,
            "add_bits": GIF_ADD_BITS,
            "low_qmin": 0,
            "low_qmax": GIF_LOW_QMAX,
            "temporal_steps": GIF_LOCAL_STEPS,
            "per_step_qmin": 0,
            "per_step_qmax": GIF_STEP_QMAX,
            "quantization_path": "low_only",
            "quantization_applied": True,
            "saliency_enabled": False,
            "temporal_policy": "low_at_t0_zero_at_t1",
        }
        mismatched = {
            key: (value, state.get(key))
            for key, value in expected.items()
            if key not in state
            or type(state[key]) is not type(value)
            or state[key] != value
        }
        forbidden_names = {
            "mask_low", "mask_low_by_role", "mask_roles",
            "saliency_score", "saliency_score_by_role",
            "high_scale", "high_zero", "high_qmin", "high_qmax",
            "integer_decomposition",
        }
        forbidden = sorted(forbidden_names & state.keys())
        if mismatched or forbidden:
            raise ValueError(
                "Invalid all-low GIF state: "
                f"mismatched={mismatched}, forbidden={forbidden}"
            )
        self.layout = _state_layout(state)
        self.register_buffer("low_scale", state["low_scale"].float())
        self.register_buffer("low_zero", state["low_zero"].float())
        _require_parameter_shape("GIF low_scale", self.low_scale, self.layout)
        _require_parameter_shape("GIF low_zero", self.low_zero, self.layout)

    @property
    def temporal_steps(self) -> int:
        return GIF_LOCAL_STEPS

    def forward(self, x: torch.Tensor, *, role: str | None = None) -> torch.Tensor:
        if role is not None:
            raise ValueError("All-low GIF does not accept a role")
        return self._quantize(x, self.low_scale, self.low_zero, qmin=0, qmax=GIF_LOW_QMAX)[0]

    def temporal(self, incoming: torch.Tensor, *, role: str | None = None) -> torch.Tensor:
        if incoming.shape[0] != self.temporal_steps:
            raise ValueError(f"GIF expects T={self.temporal_steps}")
        quantized = self.forward(incoming.sum(dim=0), role=role)
        return torch.stack((quantized, torch.zeros_like(quantized)), dim=0)


class IdentityGIF(nn.Module):
    def __init__(self, state: dict[str, Any]):
        super().__init__()
        _validate_state_header(state, "gif")
        if state.get("gif_policy") != GIF_IDENTITY_POLICY:
            raise ValueError("Invalid identity GIF policy")
        if state.get("quantization_applied") is not False:
            raise ValueError("Identity GIF must disable quantization")
        self._temporal_steps = int(state["temporal_steps"])

    @property
    def temporal_steps(self) -> int:
        return self._temporal_steps

    def forward(self, x: torch.Tensor, *, role: str | None = None) -> torch.Tensor:
        if role is not None:
            raise ValueError("Identity GIF does not accept a role")
        return x

    def temporal(self, incoming: torch.Tensor, *, role: str | None = None) -> torch.Tensor:
        if incoming.shape[0] != self.temporal_steps:
            raise ValueError(f"Identity GIF expects T={self.temporal_steps}")
        return incoming


def gif_module_from_state(state: dict[str, Any]) -> nn.Module:
    policy = state.get("gif_policy")
    if policy == SOFTMAX_SITE5_GIF_POLICY:
        return SoftmaxIdentityGIF(state)
    if policy == GIF_ALL_LOW_POLICY:
        return AllLowStaticGIF(state)
    if policy == GIF_IDENTITY_POLICY:
        return IdentityGIF(state)
    return StaticGIF(state)
