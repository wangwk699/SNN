from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .artifacts import write_json
from .phase_statistics import (
    PHASE_TAU_ACCUMULATOR_DTYPE,
    PHASE_TAU_CALIBRATION,
    PHASE_TAU_CHANNEL_POLICY,
    PHASE_TAU_EMA_FACTOR,
    PHASE_TAU_REDUCTION_POLICY,
)
from .sites import (
    is_attention_head_grouped_site,
    is_softmax_site,
    site_key,
    topology_metadata,
)
from .temporal_ops import STATISTICS_FORMAT_VERSION


def statistics_layout(site_index: int | None) -> str:
    if site_index is None:
        return "last_dim"
    if is_softmax_site(site_index):
        return "attention_softmax"
    if is_attention_head_grouped_site(site_index):
        return "attention_head"
    return "last_dim"


def _shape_metadata(layout_kind: str, activation: torch.Tensor) -> tuple[int | None, int | None, int]:
    if layout_kind == "last_dim":
        if activation.ndim < 1:
            raise ValueError("last_dim statistics require a non-scalar tensor")
        channels = int(activation.shape[-1])
        return None, None, channels
    if activation.ndim != 4:
        raise ValueError(
            f"{layout_kind} statistics require [B,H,L,D/K], got {tuple(activation.shape)}"
        )
    heads = int(activation.shape[1])
    if layout_kind == "attention_head":
        width = int(activation.shape[-1])
        return heads, width, heads * width
    return heads, None, heads


@dataclass
class SiteStatistics:
    site_index: int | None
    layout_kind: str
    num_heads: int | None
    channels_per_head: int | None
    channels: int
    value_min: torch.Tensor
    value_max: torch.Tensor
    abs_max: torch.Tensor
    sum_abs: torch.Tensor
    sum_sq: torch.Tensor
    saliency_sum: torch.Tensor
    saliency_row_count: torch.Tensor
    row_count: torch.Tensor
    tensor_count: torch.Tensor
    phase_ema_abs_max: torch.Tensor
    phase_ema_updates: torch.Tensor

    @classmethod
    def create(cls, site_index: int | None, activation: torch.Tensor) -> "SiteStatistics":
        layout = statistics_layout(site_index)
        heads, width, channels = _shape_metadata(layout, activation)
        shape = (
            (channels,)
            if layout == "last_dim"
            else ((heads, width) if layout == "attention_head" else (heads,))
        )
        saliency_shape = shape if layout != "attention_softmax" else (0,)
        return cls(
            site_index=site_index,
            layout_kind=layout,
            num_heads=heads,
            channels_per_head=width,
            channels=channels,
            value_min=torch.full(shape, torch.inf, dtype=torch.float64),
            value_max=torch.full(shape, -torch.inf, dtype=torch.float64),
            abs_max=torch.zeros(shape, dtype=torch.float64),
            sum_abs=torch.zeros(shape, dtype=torch.float64),
            sum_sq=torch.zeros(shape, dtype=torch.float64),
            saliency_sum=torch.zeros(saliency_shape, dtype=torch.float64),
            saliency_row_count=torch.zeros(saliency_shape, dtype=torch.int64),
            row_count=torch.zeros((), dtype=torch.int64),
            tensor_count=torch.zeros((), dtype=torch.int64),
            phase_ema_abs_max=torch.zeros(shape, dtype=torch.float32),
            phase_ema_updates=torch.zeros(shape, dtype=torch.int64),
        )

    def _reduced(self, activation: torch.Tensor) -> tuple[torch.Tensor, int]:
        heads, width, channels = _shape_metadata(self.layout_kind, activation)
        if (heads, width, channels) != (self.num_heads, self.channels_per_head, self.channels):
            raise ValueError(
                "Statistics layout changed: "
                f"expected heads/width/channels={(self.num_heads, self.channels_per_head, self.channels)}, "
                f"got={(heads, width, channels)}"
            )
        work = activation.detach().float()
        if self.layout_kind == "last_dim":
            values = work.reshape(-1, work.shape[-1])
            return values, int(values.shape[0])
        if self.layout_kind == "attention_head":
            # [B,H,L,D] -> [B*L,H,D], preserving H and D.
            values = work.permute(0, 2, 1, 3).reshape(-1, work.shape[1], work.shape[3])
            return values, int(values.shape[0])
        # Site 5: all query/key positions reduce independently inside each head.
        values = work.permute(0, 2, 3, 1).reshape(-1, work.shape[1])
        return values, int(values.shape[0])

    @torch.no_grad()
    def update(self, activation: torch.Tensor) -> None:
        values, rows = self._reduced(activation)
        current_min = values.amin(dim=0).double().cpu()
        current_max = values.amax(dim=0).double().cpu()
        current_abs = values.abs().amax(dim=0).float().cpu()
        self.value_min.copy_(torch.minimum(self.value_min, current_min))
        self.value_max.copy_(torch.maximum(self.value_max, current_max))
        self.abs_max.copy_(torch.maximum(self.abs_max, current_abs.double()))
        first = self.phase_ema_updates == 0
        self.phase_ema_abs_max.copy_(
            torch.where(
                first,
                current_abs,
                PHASE_TAU_EMA_FACTOR * self.phase_ema_abs_max
                + (1.0 - PHASE_TAU_EMA_FACTOR) * current_abs,
            )
        )
        self.phase_ema_updates.add_(1)
        self.sum_abs.add_(values.abs().sum(dim=0).double().cpu())
        self.sum_sq.add_(values.square().sum(dim=0).double().cpu())
        self.row_count.add_(rows)
        self.tensor_count.add_(1)

    @torch.no_grad()
    def update_saliency(self, score: torch.Tensor) -> None:
        if self.layout_kind == "attention_softmax":
            raise ValueError("Softmax Site 5 does not collect GIF saliency")
        values, rows = self._reduced(score)
        self.saliency_sum.add_(values.sum(dim=0).double().cpu())
        self.saliency_row_count.add_(rows)

    def distributed_reduce(self) -> None:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        if torch.distributed.get_world_size() > 1:
            raise RuntimeError(
                "Phase EMA calibration is order-dependent and only supports single-process calibration"
            )

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": STATISTICS_FORMAT_VERSION,
            "site_index": self.site_index,
            "layout_kind": self.layout_kind,
            "num_heads": self.num_heads,
            "channels_per_head": self.channels_per_head,
            "channels": self.channels,
            "value_min": self.value_min,
            "value_max": self.value_max,
            "abs_max": self.abs_max,
            "sum_abs": self.sum_abs,
            "sum_sq": self.sum_sq,
            "saliency_sum": self.saliency_sum,
            "saliency_row_count": self.saliency_row_count,
            "row_count": self.row_count,
            "tensor_count": self.tensor_count,
            "phase_ema_abs_max": self.phase_ema_abs_max,
            "phase_ema_updates": self.phase_ema_updates,
            "phase_tau_calibration": PHASE_TAU_CALIBRATION,
            "phase_tau_ema_factor": PHASE_TAU_EMA_FACTOR,
            "phase_tau_accumulator_dtype": PHASE_TAU_ACCUMULATOR_DTYPE,
            "phase_tau_channel_policy": PHASE_TAU_CHANNEL_POLICY,
            "phase_tau_reduction_policy": PHASE_TAU_REDUCTION_POLICY,
        }

    def summary(self) -> dict[str, Any]:
        count = max(int(self.row_count.item()), 1)
        observed = self.saliency_row_count > 0
        return {
            "format_version": STATISTICS_FORMAT_VERSION,
            "site_index": self.site_index,
            "layout_kind": self.layout_kind,
            "num_heads": self.num_heads,
            "channels_per_head": self.channels_per_head,
            "channels": self.channels,
            "statistics_shape": list(self.value_min.shape),
            "row_count": int(self.row_count.item()),
            "tensor_count": int(self.tensor_count.item()),
            "global_min": float(self.value_min.min()),
            "global_max": float(self.value_max.max()),
            "global_abs_max": float(self.abs_max.max()),
            "mean_abs": float((self.sum_abs / count).mean()),
            "mean_square": float((self.sum_sq / count).mean()),
            "saliency_observed_channels": int(observed.sum()),
            "phase_tau_calibration": PHASE_TAU_CALIBRATION,
            "phase_tau_ema_factor": PHASE_TAU_EMA_FACTOR,
            "phase_tau_accumulator_dtype": PHASE_TAU_ACCUMULATOR_DTYPE,
            "phase_tau_channel_policy": PHASE_TAU_CHANNEL_POLICY,
            "phase_tau_reduction_policy": PHASE_TAU_REDUCTION_POLICY,
            "phase_ema_updates_min_seen": int(self.phase_ema_updates.min()),
            "phase_ema_updates_max": int(self.phase_ema_updates.max()),
        }


class StatisticsStore:
    def __init__(self) -> None:
        self.items: dict[str, SiteStatistics] = {}
        self.global_items: dict[str, SiteStatistics] = {}

    def update(self, layer_index: int, site_index: int, activation: torch.Tensor) -> None:
        key = site_key(layer_index, site_index)
        if key not in self.items:
            self.items[key] = SiteStatistics.create(site_index, activation)
        self.items[key].update(activation)

    def update_saliency(self, layer_index: int, site_index: int, score: torch.Tensor) -> None:
        key = site_key(layer_index, site_index)
        if key not in self.items:
            raise RuntimeError(f"Activation statistics must be recorded before saliency: {key}")
        self.items[key].update_saliency(score)

    def update_global(self, name: str, activation: torch.Tensor) -> None:
        if name not in self.global_items:
            self.global_items[name] = SiteStatistics.create(None, activation)
        self.global_items[name].update(activation)

    def reduce_and_save(self, root: str | Path) -> dict[str, Any]:
        root = Path(root)
        manifest: dict[str, Any] = {
            "format_version": STATISTICS_FORMAT_VERSION,
            "statistics_layout_policy": "native_site_layout_v2",
            **topology_metadata(),
            "sites": {},
        }
        for key, stats in sorted(self.items.items()):
            stats.distributed_reduce()
            directory = root / key
            directory.mkdir(parents=True, exist_ok=True)
            torch.save(stats.state_dict(), directory / "statistics.pt")
            summary = stats.summary()
            write_json(directory / "statistics_summary.json", summary)
            manifest["sites"][key] = summary
        manifest["global_states"] = {}
        for name, stats in sorted(self.global_items.items()):
            stats.distributed_reduce()
            directory = root / "_global" / name
            directory.mkdir(parents=True, exist_ok=True)
            torch.save(stats.state_dict(), directory / "statistics.pt")
            summary = stats.summary()
            write_json(directory / "statistics_summary.json", summary)
            manifest["global_states"][name] = summary
        write_json(root / "statistics_manifest.json", manifest)
        return manifest
