from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .neurons import Clipper, MultiThresholdNeuron, PhaseSurrogate, StaticGIF
from .sites import site_key
from .stats import StatisticsStore
from .state_validation import validate_site_state_bundle
from .temporal_ops import from_temporal, to_temporal


class SiteController:
    def __init__(
        self,
        mode: str = "identity",
        site_root: str | Path | None = None,
        *,
        common_clip_enabled: bool = False,
    ):
        self.mode = mode
        self.common_clip_enabled = bool(common_clip_enabled)
        if self.mode not in {"phase", "gif"} and self.common_clip_enabled:
            raise ValueError(
                "common_clip_enabled only applies to phase/gif ANN replacement modes"
            )
        self.site_root = Path(site_root) if site_root is not None else None
        self.statistics = StatisticsStore()
        self._modules: dict[str, dict[str, torch.nn.Module]] = {}
        self.temporal_steps: int | None = None
        self._final_norm_phase: PhaseSurrogate | None = None

    def _load(self, layer_index: int, site_index: int) -> dict[str, torch.nn.Module]:
        key = site_key(layer_index, site_index)
        if self.site_root is None:
            raise RuntimeError("A calibration site_root is required for replacement/deployment")

        if self.mode == "phase":
            required = ("phase", "clip") if self.common_clip_enabled else ("phase",)
        elif self.mode == "gif":
            required = ("gif", "clip") if self.common_clip_enabled else ("gif",)
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
        if self.common_clip_enabled:
            raise ValueError("SNN deployment cannot enable common Clip")
        if self.site_root is None:
            raise RuntimeError("Deployment requires site_root")
        validation = validate_site_state_bundle(
            self.site_root, require_clip=False
        )
        self.mode = f"deploy_{neuron}"
        self.temporal_steps = int(validation["temporal_steps"][neuron])
        self._bundle_validation = validation
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

    def record_activation(
        self,
        layer_index: int,
        site_index: int,
        x: torch.Tensor,
        *,
        phase_activation: torch.Tensor | None = None,
    ) -> None:
        """Record calibration statistics without changing the runtime tensor."""
        if self.mode == "collect":
            self.statistics.update(
                layer_index,
                site_index,
                x,
                phase_activation=phase_activation,
            )

    def apply(
        self,
        layer_index: int,
        site_index: int,
        x: torch.Tensor,
        *,
        phase_activation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.mode in {"identity", "none"}:
            return x
        if self.mode == "collect":
            self.statistics.update(
                layer_index,
                site_index,
                x,
                phase_activation=phase_activation,
            )
            return x
        modules = self._load(layer_index, site_index)
        for module in modules.values():
            first_buffer = next(module.buffers(), None)
            if first_buffer is not None and first_buffer.device != x.device:
                module.to(x.device)
        if self.mode == "phase":
            output = modules["phase"](x)
            return modules["clip"](output) if self.common_clip_enabled else output
        if self.mode == "gif":
            output = modules["gif"](x)
            return modules["clip"](output) if self.common_clip_enabled else output
        if self.mode.startswith("deploy_"):
            if self.temporal_steps is None:
                raise RuntimeError("Call set_deployment before a temporal forward")
            temporal = to_temporal(x, self.temporal_steps)
            neuron = self.mode.removeprefix("deploy_")
            output = modules[neuron].temporal(temporal)
            if output.shape != temporal.shape:
                raise ValueError(
                    f"{neuron} temporal output shape {output.shape} != input {temporal.shape}"
                )
            if output.dtype != x.dtype or output.device != x.device:
                raise ValueError("Deployment site changed dtype or device")
            return from_temporal(output)
        raise ValueError(f"Unknown controller mode: {self.mode}")

    def apply_final_norm_phase(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode != "deploy_phase":
            return x
        if self.site_root is None or self.temporal_steps is None:
            raise RuntimeError("Final RMSNorm Phase deployment requires initialized site states")
        if self._final_norm_phase is None:
            path = self.site_root / "_global" / "final_rmsnorm" / "phase_state.pt"
            state = torch.load(path, map_location="cpu", weights_only=False)
            self._final_norm_phase = PhaseSurrogate(state)
        first_buffer = next(self._final_norm_phase.buffers(), None)
        if first_buffer is not None and first_buffer.device != x.device:
            self._final_norm_phase.to(x.device)
        temporal = to_temporal(x, self.temporal_steps)
        output = self._final_norm_phase.temporal(temporal)
        if output.shape != temporal.shape or output.dtype != x.dtype or output.device != x.device:
            raise ValueError("Final RMSNorm Phase neuron changed shape, dtype, or device")
        return from_temporal(output)
