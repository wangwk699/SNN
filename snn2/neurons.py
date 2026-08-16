from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


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
    def __init__(self, state: dict[str, Any]):
        super().__init__()
        self.T = int(state["T"])
        self.base = float(state["base"])
        self.group_size = int(state["group_size"])
        self.slope = float(state["surrogate_slope"])
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
            spike = HeavisideSigmoid.apply(membrane - amplitude, self.slope)
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
        return self.encode(incoming.sum(dim=0), return_temporal=True)


class StaticGIF(nn.Module):
    def __init__(self, state: dict[str, Any]):
        super().__init__()
        self.base_bits = int(state["base_bits"])
        self.add_bits = int(state["add_bits"])
        self.group_size = int(state["group_size"])
        self.register_buffer("low_scale", state["low_scale"].float())
        self.register_buffer("low_zero", state["low_zero"].float())
        self.register_buffer("high_scale", state["high_scale"].float())
        self.register_buffer("high_zero", state["high_zero"].float())
        self.register_buffer("mask_low", state["mask_low"].bool())

    @staticmethod
    def round_ste(x: torch.Tensor) -> torch.Tensor:
        return (x.round() - x).detach() + x

    def _quantize(
        self, x: torch.Tensor, bits: int, scale_values: torch.Tensor, zero_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = _channel_values(x, scale_values, self.group_size).clamp_min(1e-8)
        zero = _channel_values(x, zero_values, self.group_size)
        qmin, qmax = 0, 2**bits - 1
        q = (self.round_ste(x.float() / scale.float()) + zero.float()).clamp(qmin, qmax)
        dequantized = (q - zero.float()) * scale.float()
        return dequantized.to(x.dtype), q, zero.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low, _, _ = self._quantize(x, self.base_bits, self.low_scale, self.low_zero)
        high, _, _ = self._quantize(
            x, self.base_bits + self.add_bits, self.high_scale, self.high_zero
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
        return 2**self.add_bits

    def temporal(self, incoming: torch.Tensor) -> torch.Tensor:
        x = incoming.sum(dim=0)
        _, low_q, low_zero = self._quantize(x, self.base_bits, self.low_scale, self.low_zero)
        _, high_q, high_zero = self._quantize(
            x, self.base_bits + self.add_bits, self.high_scale, self.high_zero
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
        residual = high_q
        high_chunks = []
        step_max = 2**self.base_bits - 1
        for _ in range(self.temporal_steps):
            chunk = residual.clamp(0, step_max)
            high_chunks.append(chunk)
            residual = residual - chunk
        if torch.any(residual.abs() > 1e-5):
            raise RuntimeError("GIF integer decomposition left a non-zero residual")
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
        self.group_size = int(state["group_size"])
        self.register_buffer("lower", state["lower"].float())
        self.register_buffer("upper", state["upper"].float())
        if torch.any(self.lower >= self.upper):
            raise ValueError("Every clipping interval must satisfy lower < upper")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lower = _channel_values(x, self.lower, self.group_size)
        upper = _channel_values(x, self.upper, self.group_size)
        return hard_clip(x, lower, upper)
