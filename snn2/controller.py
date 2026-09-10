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
        gif_quantizer_clip_backward: str = "hard_clip",
        outer_clip_backward: str = "hard_clip",
        diagnostics_max_calls_per_site: int = 0,
        phase_T: int | None = None,
        mtn_T: int | None = None,
        mtn_K: int | None = None,
        mtn_threshold_factor: float | None = None,
        phase_surrogate_slope: float | None = None,
        checkpoint_attention_core: bool = False,
        checkpoint_mlp: bool = False,
    ):
        self.mode = mode
        self.common_clip_enabled = bool(common_clip_enabled)
        self.gif_quantizer_clip_backward = gif_quantizer_clip_backward
        self.outer_clip_backward = outer_clip_backward
        self.diagnostics_max_calls_per_site = int(diagnostics_max_calls_per_site)
        if self.gif_quantizer_clip_backward not in {"hard_clip", "ste"}:
            raise ValueError("Invalid GIF quantizer clip backward policy")
        if self.outer_clip_backward not in {"hard_clip", "ste"}:
            raise ValueError("Invalid outer Clip backward policy")
        if self.diagnostics_max_calls_per_site < 0:
            raise ValueError("diagnostics_max_calls_per_site must be non-negative")
        self._replacement_diagnostics: dict[str, dict[str, float | int]] = {}
        self.phase_surrogate_slope = None if phase_surrogate_slope is None else float(phase_surrogate_slope)
        self.phase_T = None if phase_T is None else int(phase_T)
        self.mtn_T = None if mtn_T is None else int(mtn_T)
        self.mtn_K = None if mtn_K is None else int(mtn_K)
        self.mtn_threshold_factor = None if mtn_threshold_factor is None else float(mtn_threshold_factor)
        self.checkpoint_attention_core = bool(checkpoint_attention_core)
        self.checkpoint_mlp = bool(checkpoint_mlp)
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
        self._final_norm_mtn: MultiThresholdNeuron | None = None
        self.regression_recorder = None
        self.regression_bypass_final_norm_neuron = False

    def set_regression_recorder(self, recorder) -> None:
        self.regression_recorder = recorder

    def record_regression(self, name: str, value: torch.Tensor) -> None:
        recorder = self.regression_recorder
        if recorder is not None:
            recorder.record(name, value, temporal=self.mode.startswith("deploy_"))

    def _record_replacement_diagnostics(
        self,
        key: str,
        x: torch.Tensor,
        quantized: torch.Tensor,
        output: torch.Tensor,
        *,
        role: str | None,
    ) -> None:
        if self.diagnostics_max_calls_per_site == 0:
            return
        diagnostic_key = key if role is None else f"{key}/role_{role}"
        stats = self._replacement_diagnostics.setdefault(
            diagnostic_key,
            {
                "calls": 0,
                "elements": 0,
                "quantization_changed": 0,
                "outer_clip_changed": 0,
                "quantization_error_sq": 0.0,
                "output_error_sq": 0.0,
                "input_sq": 0.0,
                "output_sq": 0.0,
                "input_output_dot": 0.0,
                "input_abs_max": 0.0,
                "output_abs_max": 0.0,
            },
        )
        if int(stats["calls"]) >= self.diagnostics_max_calls_per_site:
            return
        with torch.no_grad():
            total_elements = x.numel()
            if total_elements == 0:
                return
            max_elements = 65_536
            stride = max(1, (total_elements + max_elements - 1) // max_elements)
            indices = torch.arange(
                0, total_elements, stride, device=x.device
            )[:max_elements]
            source = torch.take(x.detach(), indices).float()
            quantized_value = torch.take(quantized.detach(), indices).float()
            final = torch.take(output.detach(), indices).float()
            quantization_error = quantized_value - source
            output_error = final - source
            stats["calls"] = int(stats["calls"]) + 1
            stats["elements"] = int(stats["elements"]) + source.numel()
            stats["quantization_changed"] = int(stats["quantization_changed"]) + int(
                torch.count_nonzero(quantized_value != source).item()
            )
            stats["outer_clip_changed"] = int(stats["outer_clip_changed"]) + int(
                torch.count_nonzero(final != quantized_value).item()
            )
            stats["quantization_error_sq"] = float(stats["quantization_error_sq"]) + float(
                torch.sum(quantization_error.square()).item()
            )
            stats["output_error_sq"] = float(stats["output_error_sq"]) + float(
                torch.sum(output_error.square()).item()
            )
            stats["input_sq"] = float(stats["input_sq"]) + float(
                torch.sum(source.square()).item()
            )
            stats["output_sq"] = float(stats["output_sq"]) + float(
                torch.sum(final.square()).item()
            )
            stats["input_output_dot"] = float(stats["input_output_dot"]) + float(
                torch.sum(source * final).item()
            )
            stats["input_abs_max"] = max(
                float(stats["input_abs_max"]), float(source.abs().max().item())
            )
            stats["output_abs_max"] = max(
                float(stats["output_abs_max"]), float(final.abs().max().item())
            )

    @staticmethod
    def _summarize_replacement_diagnostics(
        stats: dict[str, float | int],
    ) -> dict[str, float | int]:
        elements = int(stats["elements"])
        input_sq = float(stats["input_sq"])
        output_sq = float(stats["output_sq"])
        denominator = (input_sq * output_sq) ** 0.5
        return {
            "calls": int(stats["calls"]),
            "elements": elements,
            "quantization_change_ratio": (
                int(stats["quantization_changed"]) / elements if elements else 0.0
            ),
            "outer_clip_saturation_ratio": (
                int(stats["outer_clip_changed"]) / elements if elements else 0.0
            ),
            "quantization_mse": (
                float(stats["quantization_error_sq"]) / elements if elements else 0.0
            ),
            "output_mse": (
                float(stats["output_error_sq"]) / elements if elements else 0.0
            ),
            "output_relative_l2": (
                (float(stats["output_error_sq"]) / input_sq) ** 0.5
                if input_sq > 0.0
                else 0.0
            ),
            "input_output_cosine": (
                float(stats["input_output_dot"]) / denominator
                if denominator > 0.0
                else 0.0
            ),
            "input_abs_max": float(stats["input_abs_max"]),
            "output_abs_max": float(stats["output_abs_max"]),
        }

    def replacement_diagnostics_snapshot(self) -> dict[str, object]:
        raw_global: dict[str, float | int] = {
            "calls": 0,
            "elements": 0,
            "quantization_changed": 0,
            "outer_clip_changed": 0,
            "quantization_error_sq": 0.0,
            "output_error_sq": 0.0,
            "input_sq": 0.0,
            "output_sq": 0.0,
            "input_output_dot": 0.0,
            "input_abs_max": 0.0,
            "output_abs_max": 0.0,
        }
        sum_fields = {
            "calls",
            "elements",
            "quantization_changed",
            "outer_clip_changed",
            "quantization_error_sq",
            "output_error_sq",
            "input_sq",
            "output_sq",
            "input_output_dot",
        }
        for stats in self._replacement_diagnostics.values():
            for field in sum_fields:
                raw_global[field] = raw_global[field] + stats[field]
            raw_global["input_abs_max"] = max(
                float(raw_global["input_abs_max"]), float(stats["input_abs_max"])
            )
            raw_global["output_abs_max"] = max(
                float(raw_global["output_abs_max"]), float(stats["output_abs_max"])
            )
        return {
            "scope": "rank_local_first_calls_per_site_deterministic_strided_sample",
            "max_elements_per_call": 65_536,
            "max_calls_per_site": self.diagnostics_max_calls_per_site,
            "gif_quantizer_clip_backward": self.gif_quantizer_clip_backward,
            "outer_clip_backward": self.outer_clip_backward,
            "global": self._summarize_replacement_diagnostics(raw_global),
            "per_site": {
                key: self._summarize_replacement_diagnostics(value)
                for key, value in sorted(self._replacement_diagnostics.items())
            },
        }

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
                modules[name] = gif_module_from_state(
                    state,
                    quantizer_clip_backward=self.gif_quantizer_clip_backward,
                )
            else:
                modules[name] = Clipper(
                    state, backward_policy=self.outer_clip_backward
                )
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
            modules["clip"] = Clipper(
                state, backward_policy=self.outer_clip_backward
            )
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
            quantized = modules["gif"](x, role=gif_role)
            output = (
                self.apply_role_clip(
                    layer_index, site_index, quantized, role=gif_role
                )
                if site_index in {1, 7} and self.common_clip_enabled
                else (modules["clip"](quantized) if "clip" in modules else quantized)
            )
            self._record_replacement_diagnostics(
                site_key(layer_index, site_index),
                x,
                quantized,
                output,
                role=gif_role,
            )
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

    def apply_final_norm_neuron(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the clip-free global final-RMSNorm neuron for the active topology."""
        self.record_regression("final_norm/before_global_neuron", x)
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
