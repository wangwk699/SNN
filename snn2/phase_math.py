"""Shared runtime mathematics for the configurable Phase neuron base."""

from __future__ import annotations

import math
from typing import Any

import torch


def validate_phase_base(base: Any) -> float:
    """Return a finite Phase geometric base strictly greater than one."""
    if isinstance(base, bool):
        raise ValueError("phase.base must be a finite number greater than 1.0")
    try:
        value = float(base)
    except (TypeError, ValueError) as exc:
        raise ValueError("phase.base must be a finite number greater than 1.0") from exc
    if not math.isfinite(value) or value <= 1.0:
        raise ValueError("phase.base must be a finite number greater than 1.0")
    return value


def format_phase_base(base: Any) -> str:
    """Canonical, stable spelling of a validated base for artifact paths."""
    return format(validate_phase_base(base), ".12g")


def phase_amplitude_sum_factor(base: Any, T: int) -> float:
    """Return ``sum(base ** -k for k in range(1, T + 1))``."""
    value = validate_phase_base(base)
    if not isinstance(T, int) or isinstance(T, bool) or T <= 0:
        raise ValueError("Phase T must be a positive integer")
    return (1.0 - value ** (-T)) / (value - 1.0)


def phase_representable_bound(
    tau: torch.Tensor, *, T: int, base: Any
) -> torch.Tensor:
    """Maximum absolute magnitude representable by ``T`` Phase timesteps."""
    return tau * phase_amplitude_sum_factor(base, T)
