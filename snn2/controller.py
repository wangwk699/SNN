from __future__ import annotations

from pathlib import Path
import math
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
        checkpoint_attention_core: bool = False,
        checkpoint_mlp: bool = False,
        gif_round_gradient_estimator: str = "STE",
        gif_htge_t: float = 16.0,
    ):
        self.mode = mode
        self.common_clip_enabled = bool(common_clip_enabled)
        self.phase_surrogate_slope = None if phase_surrogate_slope is None else float(phase_surrogate_slope)
        self.phase_T = None if phase_T is None else int(phase_T)
        self.mtn_T = None if mtn_T is None else int(mtn_T)
        self.mtn_K = None if mtn_K is None else int(mtn_K)
        self.mtn_threshold_factor = None if mtn_threshold_factor is None else float(mtn_threshold_factor)
        self.checkpoint_attention_core = bool(checkpoint_attention_core)
        self.checkpoint_mlp = bool(checkpoint_mlp)
        estimator = str(gif_round_gradient_estimator)
        value = float(gif_htge_t)
        if estimator not in {"STE", "HTGE"}:
            raise ValueError("gif_round_gradient_estimator must be STE or HTGE")
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("gif_htge_t must be a positive finite number")
        self.gif_round_gradient_estimator = estimator
        self.gif_htge_t = value
        if self.mode == "phase" and (self.phase_surrogate_slope is None or self.phase_T is None):
            raise ValueError("Phase ANN replacement requires explicit phase_T and phase_surrogate_slope")
        if self.mode not in {"phase", "gif"} and self.common_clip_enabled:
            raise ValueError("common_clip_enabled only applies to phase/gif ANN replacement modes")
        self.site_root = Path(site_root) if site_root is not None else None
        self.clip_root = Path(clip_root) if clip_root is not None else None
        if self.common_clip_enabled and self.clip_root is None:
            raise ValueError("common Clip requires an explicit Stage B clip_root")
        self.statistics = StatisticsStore()
        self.gif_mse_collector = None
        self._modules: dict[str, dict[str, torch.nn.Module]] = {}
        self.temporal_steps: int | None = None
        self._final_norm_phase: PhaseSurrogate | None = None
        self._final_norm_mtn: MultiThresholdNeuron | None = None
        self.regression_recorder = None
        self.regression_bypass_final_norm_neuron = False
        self.calibration_neuron: str | None = None
        self.calibration_block_index: int | None = None
        self.calibration_collect_current_block = False

    @property
    def temporal_execution_enabled(self) -> bool:
        return self.mode.startswith("deploy_") or self.mode in {"calibration_collect", "calibration_deploy"} or (self.mode == "gif_mse_collect" and self.temporal_steps is not None)

    @property
    def sequential_calibration_active(self) -> bool:
        return self.mode in {"calibration_collect", "calibration_deploy"} or (self.mode == "gif_mse_collect" and self.temporal_steps is not None)

    @property
    def sequential_calibration_neuron(self) -> str | None:
        return self.calibration_neuron if self.sequential_calibration_active else None

    @property
    def collecting_statistics(self) -> bool:
        return self.mode in {"collect", "calibration_collect", "gif_mse_collect"}


    @property
    def collecting_saliency(self) -> bool:
        return self.mode in {"collect", "calibration_collect"}

    def logical_activation_for_calibration(self, x: torch.Tensor) -> torch.Tensor:
        if not self.sequential_calibration_active:
            return x
        if self.temporal_steps is None:
            raise RuntimeError("Sequential calibration timestep is unset")
        return to_temporal(x, self.temporal_steps).sum(dim=0)

    def begin_sequential_calibration(self, neuron: str, block_index: int) -> int:
        if neuron not in {"phase", "gif", "mtn"}:
            raise ValueError(f"Unknown sequential calibration neuron: {neuron}")
        if block_index < 0:
            raise ValueError("block_index must be non-negative")
        self.calibration_neuron, self.calibration_block_index = neuron, int(block_index)
        self.calibration_collect_current_block = True
        self.mode = "calibration_collect"
        self.temporal_steps = (int(self.phase_T) if neuron == "phase" else int(self.mtn_T) if neuron == "mtn" else 2)
        if self.temporal_steps <= 0:
            raise ValueError("Sequential calibration requires temporal runtime parameters")
        return self.temporal_steps

    def begin_gif_mse_collection(self, *, block_index: int | None = None) -> None:
        if block_index is None:
            self.calibration_neuron = None
            self.calibration_block_index = None
            self.temporal_steps = None
        else:
            self.calibration_neuron = "gif"
            self.calibration_block_index = int(block_index)
            self.temporal_steps = 2
        self.calibration_collect_current_block = True
        self.mode = "gif_mse_collect"

    def begin_sequential_deployment(self, block_index: int) -> None:
        if self.calibration_neuron is None or self.calibration_block_index != int(block_index):
            raise RuntimeError("Sequential deployment must follow collection for the same block")
        self.calibration_collect_current_block = False
        self.mode = "calibration_deploy"

    def end_sequential_calibration(self) -> None:
        self.mode = "collect"
        self.calibration_neuron = None
        self.calibration_block_index = None
        self.calibration_collect_current_block = False
        self.temporal_steps = None

    def clear_layer_module_cache(self, layer_index: int) -> None:
        prefix = f"layer_{int(layer_index):03d}/"
        for key in list(self._modules):
            if key.startswith(prefix):
                del self._modules[key]

    def clear_runtime_module_cache(self) -> None:
        self._modules.clear()
        self._final_norm_phase = None
        self._final_norm_mtn = None

    def set_regression_recorder(self, recorder) -> None:
        self.regression_recorder = recorder

    def record_regression(self, name: str, value: torch.Tensor) -> None:
        recorder = self.regression_recorder
        if recorder is not None:
            recorder.record(name, value, temporal=self.temporal_execution_enabled)

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
        elif self.temporal_execution_enabled:
            neuron = self.calibration_neuron if self.mode == "calibration_deploy" else self.mode.removeprefix("deploy_")
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
                modules[name] = (
                    gif_module_from_state(
                        state,
                        round_gradient_estimator=self.gif_round_gradient_estimator,
                        htge_t=self.gif_htge_t,
                    )
                    if self.mode == "gif" else gif_module_from_state(state)
                )
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
        if self.mode == "gif_mse_collect":
            return
        if self.collecting_statistics:
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
        if self.mode == "gif_mse_collect":
            if self.gif_mse_collector is None:
                raise RuntimeError("GIF MSE collector is not installed")
            self.gif_mse_collector.update(
                layer_index, site_index, self.logical_activation_for_calibration(x)
            )
            return
        if self.collecting_statistics:
            self.statistics.update(layer_index, site_index, self.logical_activation_for_calibration(x))

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
        if self.collecting_statistics:
            if self.mode == "gif_mse_collect":
                if self.gif_mse_collector is None:
                    raise RuntimeError("GIF MSE collector is not installed")
                self.gif_mse_collector.update(
                    layer_index, site_index, self.logical_activation_for_calibration(x), role=gif_role
                )
            else:
                self.statistics.update(layer_index, site_index, self.logical_activation_for_calibration(x))
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
        if self.temporal_execution_enabled:
            if self.temporal_steps is None:
                raise RuntimeError("Call set_deployment before a temporal forward")
            temporal = to_temporal(x, self.temporal_steps)
            neuron = self.calibration_neuron if self.mode == "calibration_deploy" else self.mode.removeprefix("deploy_")
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

    def apply_final_norm_neuron(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the clip-free global final-RMSNorm neuron for the active topology."""
        self.record_regression("final_norm/before_global_neuron", x)
        if self.mode == "gif_mse_collect":
            return x
        if self.mode == "calibration_collect":
            self.statistics.update_global("final_rmsnorm", self.logical_activation_for_calibration(x))
            self.record_regression("final_norm/after_global_identity", x)
            return x
        if self.regression_bypass_final_norm_neuron or self.mode in {"identity", "none", "collect", "gif", "deploy_gif"}:
            self.record_regression("final_norm/after_global_identity", x)
            return x
        if self.site_root is None:
            raise RuntimeError("Final RMSNorm neuron requires initialized site states")
        root = self.site_root / "_global" / "final_rmsnorm"
        if self.mode == "phase":
            if self._final_norm_phase is None:
                state = torch.load(root / "phase_state.pt", map_location="cpu", weights_only=False)
                self._final_norm_phase = PhaseSurrogate(state, T=int(self.phase_T), surrogate_slope=self.phase_surrogate_slope)
            module, temporal = self._final_norm_phase, False
        elif self.mode == "deploy_phase":
            if self._final_norm_phase is None:
                state = torch.load(root / "phase_state.pt", map_location="cpu", weights_only=False)
                self._final_norm_phase = PhaseSurrogate(state, T=int(self.phase_T))
            module, temporal = self._final_norm_phase, True
        elif self.mode == "deploy_mtn":
            if self._final_norm_mtn is None:
                state = torch.load(root / "mtn_state.pt", map_location="cpu", weights_only=False)
                self._final_norm_mtn = MultiThresholdNeuron(state, T=int(self.mtn_T), K=int(self.mtn_K), threshold_factor=float(self.mtn_threshold_factor))
            module, temporal = self._final_norm_mtn, True
        else:
            self.record_regression("final_norm/after_global_identity", x)
            return x
        first_buffer = next(module.buffers(), None)
        if first_buffer is not None and first_buffer.device != x.device:
            module.to(x.device)
        if temporal:
            if self.temporal_steps is None:
                raise RuntimeError("Temporal Final RMSNorm neuron requires deployment initialization")
            incoming = to_temporal(x, self.temporal_steps)
            output = module.temporal(incoming)
            if output.shape != incoming.shape:
                raise ValueError("Final RMSNorm neuron changed temporal shape")
            output = from_temporal(output)
        else:
            output = module(x)
        if output.shape != x.shape or output.dtype != x.dtype or output.device != x.device:
            raise ValueError("Final RMSNorm neuron changed shape, dtype, or device")
        self.record_regression("final_norm/after_global_neuron", output)
        return output
