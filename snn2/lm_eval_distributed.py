"""Tulu lm-eval document data-parallel helpers for preinitialized models."""
from __future__ import annotations

from typing import Iterable, Mapping

import torch
from accelerate import Accelerator
from lm_eval.models.huggingface import HFLM


class DistributedPreinitializedHFLM(HFLM):
    """Make lm-eval see Accelerate ranks for an already-loaded model replica."""

    def __init__(self, *, accelerator: Accelerator, **kwargs):
        super().__init__(**kwargs)
        self.accelerator = accelerator
        self._rank = int(accelerator.process_index)
        self._world_size = int(accelerator.num_processes)
        self._device = accelerator.device


def indices_for_rank(indices: Iterable[int], *, rank: int, world_size: int) -> list[int]:
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    return list(indices)[rank::world_size]


def sum_execution_counters(counters: Iterable[Mapping[str, int]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for counter in counters:
        for key, value in counter.items():
            result[key] = result.get(key, 0) + int(value)
    return result


def gather_sum_execution_counter(local_counter: Mapping[str, int], *, world_size: int) -> dict[str, int]:
    if world_size == 1:
        return {key: int(value) for key, value in local_counter.items()}
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized for multi-process lm-eval")
    gathered = [None] * world_size
    torch.distributed.all_gather_object(gathered, dict(local_counter))
    return sum_execution_counters(gathered)


def distributed_max_seconds(local_seconds: float, *, device: torch.device, world_size: int) -> float:
    if world_size == 1:
        return float(local_seconds)
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized for multi-process lm-eval")
    value = torch.tensor([float(local_seconds)], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
    return float(value.item())
