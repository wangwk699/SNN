from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .neurons import Clipper, MultiThresholdNeuron, PhaseSurrogate, StaticGIF
from .sites import site_key
from .stats import StatisticsStore


class SiteController:
    def __init__(self, mode: str = "identity", site_root: str | Path | None = None):
        self.mode = mode
        self.site_root = Path(site_root) if site_root is not None else None
        self.statistics = StatisticsStore()
        self._modules: dict[str, dict[str, torch.nn.Module]] = {}
        self.temporal_steps: int | None = None

    def _load(self, layer_index: int, site_index: int) -> dict[str, torch.nn.Module]:
        key = site_key(layer_index, site_index)
        if self.site_root is None:
            raise RuntimeError("A calibration site_root is required for replacement/deployment")

        if self.mode == "phase":
            required = ("phase", "clip")
        elif self.mode == "gif":
            required = ("gif", "clip")
        elif self.mode.startswith("deploy_"):
            neuron = self.mode.removeprefix("deploy_")
            if neuron not in {"phase", "gif", "mtn"}:
                raise ValueError(f"Unknown deployment neuron: {neuron}")
            required = (neuron,)
        else:
            raise ValueError(f"Mode {self.mode!r} does not load calibration states")

        directory = self.site_root / key
        modules = self._modules.setdefault(key, {})
        factories = {
            "phase": PhaseSurrogate,
            "gif": StaticGIF,
            "mtn": MultiThresholdNeuron,
            "clip": Clipper,
        }
        for name in required:
            if name not in modules:
                state = torch.load(
                    directory / f"{name}_state.pt", map_location="cpu", weights_only=False
                )
                modules[name] = factories[name](state)
        return modules

    def set_deployment(self, neuron: str) -> int:
        if neuron not in {"phase", "gif", "mtn"}:
            raise ValueError(neuron)
        self.mode = f"deploy_{neuron}"
        if self.site_root is None:
            raise RuntimeError("Deployment requires site_root")
        state_name = f"{neuron}_state.pt"
        first = next(self.site_root.glob(f"layer_*/site_*/{state_name}"), None)
        if first is None:
            raise FileNotFoundError(f"No {state_name} files under {self.site_root}")
        state = torch.load(first, map_location="cpu", weights_only=False)
        if neuron in {"phase", "mtn"}:
            self.temporal_steps = int(state["T"])
        else:
            self.temporal_steps = 2 ** int(state["add_bits"])
        return self.temporal_steps

    def record_saliency(self, layer_index: int, site_index: int, score: torch.Tensor) -> None:
        if self.mode == "collect":
            self.statistics.update_saliency(layer_index, site_index, score)

    def record_saliency_reduced(
        self,
        layer_index: int,
        site_index: int,
        score_sum: torch.Tensor,
        row_count: int,
    ) -> None:
        if self.mode == "collect":
            self.statistics.update_saliency_reduced(
                layer_index, site_index, score_sum, row_count
            )

    def apply(self, layer_index: int, site_index: int, x: torch.Tensor) -> torch.Tensor:
        if self.mode in {"identity", "none"}:
            return x
        if self.mode == "collect":
            self.statistics.update(layer_index, site_index, x)
            return x
        modules = self._load(layer_index, site_index)
        for module in modules.values():
            first_buffer = next(module.buffers(), None)
            if first_buffer is not None and first_buffer.device != x.device:
                module.to(x.device)
        if self.mode == "phase":
            return modules["clip"](modules["phase"](x))
        if self.mode == "gif":
            return modules["clip"](modules["gif"](x))
        if self.mode.startswith("deploy_"):
            if self.temporal_steps is None:
                raise RuntimeError("Call set_deployment before a temporal forward")
            if x.shape[0] % self.temporal_steps != 0:
                raise ValueError(
                    f"Batch dimension {x.shape[0]} is not divisible by T={self.temporal_steps}"
                )
            batch = x.shape[0] // self.temporal_steps
            temporal = x.reshape(self.temporal_steps, batch, *x.shape[1:])
            neuron = self.mode.removeprefix("deploy_")
            output = modules[neuron].temporal(temporal)
            return output.reshape_as(x)
        raise ValueError(f"Unknown controller mode: {self.mode}")
