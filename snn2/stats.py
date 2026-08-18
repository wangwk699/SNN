from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .artifacts import write_json

from .sites import (
    SITE_COORDINATES, SITE_COUNT, SITE_IDS, SITE_NAMES, SITE_TOPOLOGY_VERSION,
    site_key, topology_metadata,
)

@dataclass
class SiteStatistics:
    channels: int
    variable_channels: bool
    value_min: torch.Tensor
    value_max: torch.Tensor
    abs_max: torch.Tensor
    sum_abs: torch.Tensor
    sum_sq: torch.Tensor
    saliency_sum: torch.Tensor
    saliency_row_count: torch.Tensor
    row_count: torch.Tensor
    tensor_count: torch.Tensor

    @classmethod
    def create(cls, channels: int, variable_channels: bool = False) -> "SiteStatistics":
        return cls(
            channels=channels,
            variable_channels=variable_channels,
            value_min=torch.full((channels,), torch.inf, dtype=torch.float64),
            value_max=torch.full((channels,), -torch.inf, dtype=torch.float64),
            abs_max=torch.zeros(channels, dtype=torch.float64),
            sum_abs=torch.zeros(channels, dtype=torch.float64),
            sum_sq=torch.zeros(channels, dtype=torch.float64),
            saliency_sum=torch.zeros(channels, dtype=torch.float64),
            saliency_row_count=torch.zeros(channels, dtype=torch.int64),
            row_count=torch.zeros((), dtype=torch.int64),
            tensor_count=torch.zeros((), dtype=torch.int64),
        )

    @torch.no_grad()
    def update(self, activation: torch.Tensor) -> None:
        values = activation.detach().reshape(-1, activation.shape[-1])
        active = int(values.shape[-1])
        if (not self.variable_channels and active != self.channels) or active > self.channels:
            raise ValueError(f"Channel dimension changed from {self.channels} to {values.shape[-1]}")
        work = values.float()
        # self.value_min[:active].minimum_(work.amin(dim=0).double().cpu())
        # self.value_max[:active].maximum_(work.amax(dim=0).double().cpu())
        # self.abs_max[:active].maximum_(work.abs().amax(dim=0).double().cpu())

        current_min = work.amin(dim=0).double().cpu()
        current_max = work.amax(dim=0).double().cpu()
        current_abs_max = work.abs().amax(dim=0).double().cpu()
        self.value_min[:active].copy_(torch.minimum(self.value_min[:active], current_min))
        self.value_max[:active].copy_(torch.maximum(self.value_max[:active], current_max))
        self.abs_max[:active].copy_(torch.maximum(self.abs_max[:active], current_abs_max))

        self.sum_abs[:active].add_(work.abs().sum(dim=0).double().cpu())
        self.sum_sq[:active].add_(work.square().sum(dim=0).double().cpu())
        self.row_count.add_(values.shape[0])
        self.tensor_count.add_(1)

    @torch.no_grad()
    def update_saliency(self, score: torch.Tensor) -> None:
        values = score.detach().reshape(-1, score.shape[-1])
        active = int(values.shape[-1])
        if (not self.variable_channels and active != self.channels) or active > self.channels:
            raise ValueError(
                f"Saliency channel dimension changed from {self.channels} to {values.shape[-1]}"
            )
        self.saliency_sum[:active].add_(values.float().sum(dim=0).double().cpu())
        self.saliency_row_count[:active].add_(values.shape[0])

    @torch.no_grad()
    def update_saliency_reduced(self, score_sum: torch.Tensor, row_count: int) -> None:
        active = int(score_sum.numel())
        if (not self.variable_channels and active != self.channels) or active > self.channels:
            raise ValueError(
                f"Reduced saliency channel dimension changed from {self.channels} to {active}"
            )
        self.saliency_sum[:active].add_(score_sum.detach().double().cpu())
        self.saliency_row_count[:active].add_(int(row_count))

    def distributed_reduce(self) -> None:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        backend = torch.distributed.get_backend()
        device = torch.device("cuda") if backend == "nccl" else torch.device("cpu")
        reductions = (
            (self.value_min, torch.distributed.ReduceOp.MIN),
            (self.value_max, torch.distributed.ReduceOp.MAX),
            (self.abs_max, torch.distributed.ReduceOp.MAX),
            (self.sum_abs, torch.distributed.ReduceOp.SUM),
            (self.sum_sq, torch.distributed.ReduceOp.SUM),
            (self.saliency_sum, torch.distributed.ReduceOp.SUM),
            (self.saliency_row_count, torch.distributed.ReduceOp.SUM),
            (self.row_count, torch.distributed.ReduceOp.SUM),
            (self.tensor_count, torch.distributed.ReduceOp.SUM),
        )
        for tensor, operation in reductions:
            work = tensor.to(device)
            torch.distributed.all_reduce(work, op=operation)
            tensor.copy_(work.cpu())

    def state_dict(self) -> dict[str, Any]:
        return {
            "channels": self.channels,
            "variable_channels": self.variable_channels,
            "value_min": self.value_min,
            "value_max": self.value_max,
            "abs_max": self.abs_max,
            "sum_abs": self.sum_abs,
            "sum_sq": self.sum_sq,
            "saliency_sum": self.saliency_sum,
            "saliency_row_count": self.saliency_row_count,
            "row_count": self.row_count,
            "tensor_count": self.tensor_count,
        }

    def summary(self) -> dict[str, Any]:
        count = max(int(self.row_count.item()), 1)
        return {
            "channels": self.channels,
            "row_count": int(self.row_count.item()),
            "tensor_count": int(self.tensor_count.item()),
            "global_min": float(self.value_min.min().item()),
            "global_max": float(self.value_max.max().item()),
            "global_abs_max": float(self.abs_max.max().item()),
            "mean_abs": float((self.sum_abs / count).mean().item()),
            "mean_square": float((self.sum_sq / count).mean().item()),
            "saliency_row_count_min_seen": int(
                self.saliency_row_count[self.saliency_row_count > 0].min().item()
            ) if torch.any(self.saliency_row_count > 0) else 0,
            "saliency_observed_channels": int((self.saliency_row_count > 0).sum().item()),
            "mean_operator_saliency": float(
                (
                    self.saliency_sum[self.saliency_row_count > 0]
                    / self.saliency_row_count[self.saliency_row_count > 0].clamp_min(1)
                ).mean().item()
                if torch.any(self.saliency_row_count > 0)
                else 0.0
            ),
        }


class StatisticsStore:
    def __init__(self, max_channels_by_site: dict[int, int] | None = None):
        self.items: dict[str, SiteStatistics] = {}
        self.max_channels_by_site = dict(max_channels_by_site or {})

    def update(self, layer_index: int, site_index: int, activation: torch.Tensor) -> None:
        key = site_key(layer_index, site_index)
        if key not in self.items:
            variable = site_index in self.max_channels_by_site
            channels = self.max_channels_by_site.get(site_index, int(activation.shape[-1]))
            self.items[key] = SiteStatistics.create(channels, variable_channels=variable)
        self.items[key].update(activation)

    def update_saliency(
        self, layer_index: int, site_index: int, score: torch.Tensor
    ) -> None:
        key = site_key(layer_index, site_index)
        if key not in self.items:
            raise RuntimeError(f"Activation statistics must be recorded before saliency: {key}")
        self.items[key].update_saliency(score)

    def update_saliency_reduced(
        self,
        layer_index: int,
        site_index: int,
        score_sum: torch.Tensor,
        row_count: int,
    ) -> None:
        key = site_key(layer_index, site_index)
        if key not in self.items:
            raise RuntimeError(f"Activation statistics must be recorded before saliency: {key}")
        self.items[key].update_saliency_reduced(score_sum, row_count)

    def reduce_and_save(self, root: str | Path) -> dict[str, Any]:
        root = Path(root)
        manifest: dict[str, Any] = {"format_version": 1, **topology_metadata(), "sites": {}}
        for key, stats in sorted(self.items.items()):
            stats.distributed_reduce()
            directory = root / key
            directory.mkdir(parents=True, exist_ok=True)
            torch.save(stats.state_dict(), directory / "statistics.pt")
            summary = stats.summary()
            write_json(directory / "statistics_summary.json", summary)
            manifest["sites"][key] = summary
        write_json(root / "statistics_manifest.json", manifest)
        return manifest
