from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .neurons import Clipper, MultiThresholdNeuron, PhaseSurrogate, gif_module_from_state
from .sites import site_key, site_supports_clip, site_supports_clip_for_mode
from .stats import StatisticsStore
from .state_validation import ClipBundlePolicy, validate_site_state_bundle
from .temporal_ops import from_temporal, to_temporal


class SiteController:
    def __init__(
        self,
        mode: str = "identity",
        site_root: str | Path | None = None,
        *,
        clip_root: str | Path | None = None,
        common_clip_enabled: bool = False,
        phase_T: int | None = None,
        mtn_T: int | None = None,
        mtn_K: int | None = None,
        mtn_threshold_factor: float | None = None,
        phase_surrogate_slope: float | None = None,
    ):
        self.mode = mode
        self.common_clip_enabled = bool(common_clip_enabled)
        self.phase_surrogate_slope = None if phase_surrogate_slope is None else float(phase_surrogate_slope)
        self.phase_T = None if phase_T is None else int(phase_T)
        self.mtn_T = None if mtn_T is None else int(mtn_T)
        self.mtn_K = None if mtn_K is None else int(mtn_K)
        self.mtn_threshold_factor = None if mtn_threshold_factor is None else float(mtn_threshold_factor)
        if self.mode == "phase" and (self.phase_surrogate_slope is None or self.phase_T is None):
            raise ValueError("Phase ANN replacement requires explicit phase_T and phase_surrogate_slope")
        if self.mode not in {"phase", "gif"} and self.common_clip_enabled:
            raise ValueError("common_clip_enabled only applies to phase/gif ANN replacement modes")
        self.site_root = Path(site_root) if site_root is not None else None
        self.clip_root = Path(clip_root) if clip_root is not None else None
        if self.common_clip_enabled and self.clip_root is None:
            raise ValueError("common Clip requires an explicit Stage B clip_root")
        self.statistics = StatisticsStore()
        self._modules: dict[str, dict[str, torch.nn.Module]] = {}
        self.temporal_steps: int | None = None
        self._final_norm_phase: PhaseSurrogate | None = None
        self.regression_recorder = None
        self.regression_bypass_final_norm_phase = False

    def set_regression_recorder(self, recorder) -> None:
        self.regression_recorder = recorder

    def record_regression(self, name: str, value: torch.Tensor) -> None:
        recorder = self.regression_recorder
        if recorder is not None:
            recorder.record(name, value, temporal=self.mode.startswith("deploy_"))

    def _load(self, layer_index: int, site_index: int) -> dict[str, torch.nn.Module]:
        key = site_key(layer_index, site_index)
        if self.site_root is None:
            raise RuntimeError("A calibration site_root is required for replacement/deployment")

        clip_enabled = (
            self.common_clip_enabled
            and site_supports_clip_for_mode(site_index, self.mode)
            and not (self.mode == "phase" and site_index in {1, 7})
        )
        if self.mode == "phase":
            required = ("phase", "clip") if clip_enabled else ("phase",)
        elif self.mode == "gif":
            required = ("gif", "clip") if clip_enabled else ("gif",)
        elif self.mode.startswith("deploy_"):
            neuron = self.mode.removeprefix("deploy_")
            if neuron not in {"phase", "gif", "mtn"}:
                raise ValueError(f"Unknown deployment neuron: {neuron}")
            required = (neuron,)
        else:
            raise ValueError(f"Mode {self.mode!r} does not load calibration states")
        directory = self.site_root / key
        modules = self._modules.setdefault(key, {})
        for name in required:
            if name in modules:
                continue
            state_directory = self.clip_root / key if name == "clip" else directory
            state = torch.load(state_directory / f"{name}_state.pt", map_location="cpu", weights_only=False)
            if name == "phase":
                modules[name] = PhaseSurrogate(
                    state, T=int(self.phase_T),
                    surrogate_slope=self.phase_surrogate_slope if self.mode == "phase" else None,
                )
            elif name == "mtn":
                modules[name] = MultiThresholdNeuron(
                    state, T=int(self.mtn_T), K=int(self.mtn_K),
                    threshold_factor=float(self.mtn_threshold_factor),
                )
            elif name == "gif":
                modules[name] = gif_module_from_state(state)
            else:
                modules[name] = Clipper(state)
        return modules

    def set_deployment(
        self, neuron: str, *, clip_bundle_policy: ClipBundlePolicy
    ) -> int:
        if neuron not in {"phase", "gif", "mtn"}:
            raise ValueError(neuron)
        if self.common_clip_enabled:
            raise ValueError("SNN deployment cannot enable common Clip")
        if self.site_root is None:
            raise RuntimeError("Deployment requires site_root")
        validation = validate_site_state_bundle(
            self.site_root, clip_policy=clip_bundle_policy
        )
        self.mode = f"deploy_{neuron}"
        if neuron == "phase":
            if self.phase_T is None:
                raise ValueError("Phase deployment requires phase_T")
            self.temporal_steps = self.phase_T
        elif neuron == "mtn":
            if None in (self.mtn_T, self.mtn_K, self.mtn_threshold_factor):
                raise ValueError("MTN deployment requires mtn_T, mtn_K and threshold_factor")
            self.temporal_steps = self.mtn_T
        else:
            self.temporal_steps = int(validation["temporal_steps"]["gif"])
        self._bundle_validation = validation
        return self.temporal_steps

    def apply_role_clip(
        self, layer_index: int, site_index: int, x: torch.Tensor, *, role: str
    ) -> torch.Tensor:
        if self.mode not in {"phase", "gif"} or not self.common_clip_enabled:
            return x
        if site_index not in {1, 7}:
            raise ValueError("Role Clip is only valid for multi-role Site 1/7")
        key = site_key(layer_index, site_index)
        modules = self._modules.setdefault(key, {})
        if "clip" not in modules:
            if self.clip_root is None:
                raise RuntimeError("Role Clip requires Stage B clip_root")
            state = torch.load(self.clip_root / key / "clip_state.pt", map_location="cpu", weights_only=False)
            modules["clip"] = Clipper(state)
        clip = modules["clip"]
        first_buffer = next(clip.buffers(), None)
        if first_buffer is not None and first_buffer.device != x.device:
            clip.to(x.device)
        return clip(x, role=role)

    def record_saliency(
        self, layer_index: int, site_index: int, score: torch.Tensor,
        *, role: str = "default", source: str = "unspecified"
    ) -> None:
        if self.mode == "collect":
            self.statistics.update_saliency(
                layer_index, site_index, score, role=role, source=source
            )

    def record_activation(
        self,
        layer_index: int,
        site_index: int,
        x: torch.Tensor,
    ) -> None:
        """Record calibration statistics without changing the runtime tensor."""
        if self.mode == "collect":
            self.statistics.update(layer_index, site_index, x)

    def apply(
        self,
        layer_index: int,
        site_index: int,
        x: torch.Tensor,
        *,
        gif_role: str | None = None,
    ) -> torch.Tensor:
        recorder = self.regression_recorder
        checkpoint = f"layer_{layer_index:03d}/site_{site_index:02d}"
        if gif_role is not None:
            checkpoint += f"/gif_{gif_role}"
        if recorder is not None:
            self.record_regression(f"{checkpoint}/pre", x)
        if self.mode in {"identity", "none"}:
            if recorder is not None:
                self.record_regression(f"{checkpoint}/post", x)
            return x
        if self.mode == "collect":
            self.statistics.update(layer_index, site_index, x)
            if recorder is not None:
                self.record_regression(f"{checkpoint}/post", x)
            return x
        modules = self._load(layer_index, site_index)
        for module in modules.values():
            first_buffer = next(module.buffers(), None)
            if first_buffer is not None and first_buffer.device != x.device:
                module.to(x.device)
        if self.mode == "phase":
            output = modules["phase"](x)
            # Site 1/7 use role-specific Clip only in branch pre-hooks;
            # never apply a cached role Clipper to shared RMSNorm output.
            if site_index not in {1, 7} and "clip" in modules:
                output = modules["clip"](output)
            if recorder is not None:
                self.record_regression(f"{checkpoint}/post", output)
            return output
        if self.mode == "gif":
            output = modules["gif"](x, role=gif_role)
            output = self.apply_role_clip(layer_index, site_index, output, role=gif_role) if site_index in {1, 7} and self.common_clip_enabled else (modules["clip"](output) if "clip" in modules else output)
            if recorder is not None:
                self.record_regression(f"{checkpoint}/post", output)
            return output
        if self.mode.startswith("deploy_"):
            if self.temporal_steps is None:
                raise RuntimeError("Call set_deployment before a temporal forward")
            temporal = to_temporal(x, self.temporal_steps)
            neuron = self.mode.removeprefix("deploy_")
            output = (
                modules[neuron].temporal(temporal, role=gif_role)
                if neuron == "gif" else modules[neuron].temporal(temporal)
            )
            if output.shape != temporal.shape:
                raise ValueError(
                    f"{neuron} temporal output shape {output.shape} != input {temporal.shape}"
                )
            if output.dtype != x.dtype or output.device != x.device:
                raise ValueError("Deployment site changed dtype or device")
            output = from_temporal(output)
            if recorder is not None:
                self.record_regression(f"{checkpoint}/post", output)
            return output
        raise ValueError(f"Unknown controller mode: {self.mode}")

    def apply_final_norm_phase(self, x: torch.Tensor) -> torch.Tensor:
        recorder = self.regression_recorder
        if recorder is not None:
            self.record_regression("final_norm/before_global_phase", x)
        if self.mode != "deploy_phase" or self.regression_bypass_final_norm_phase:
            if recorder is not None:
                self.record_regression("final_norm/after_global_phase", x)
            return x
        if self.site_root is None or self.temporal_steps is None:
            raise RuntimeError("Final RMSNorm Phase deployment requires initialized site states")
        if self._final_norm_phase is None:
            path = self.site_root / "_global" / "final_rmsnorm" / "phase_state.pt"
            state = torch.load(path, map_location="cpu", weights_only=False)
            self._final_norm_phase = PhaseSurrogate(state, T=int(self.phase_T))
        first_buffer = next(self._final_norm_phase.buffers(), None)
        if first_buffer is not None and first_buffer.device != x.device:
            self._final_norm_phase.to(x.device)
        temporal = to_temporal(x, self.temporal_steps)
        output = self._final_norm_phase.temporal(temporal)
        if output.shape != temporal.shape or output.dtype != x.dtype or output.device != x.device:
            raise ValueError("Final RMSNorm Phase neuron changed shape, dtype, or device")
        output = from_temporal(output)
        if recorder is not None:
            self.record_regression("final_norm/after_global_phase", output)
        return output
