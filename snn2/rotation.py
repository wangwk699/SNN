from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import CausalLMCollator, tokenize_dataset
from .hadamard import (
    HadamardSpec,
    make_spec,
    random_hadamard,
    transform_weight_left_transpose,
    transform_weight_right,
)


@dataclass
class ModelParts:
    backbone: nn.Module
    layers: Iterable[nn.Module]
    embedding: nn.Embedding
    final_norm: nn.Module
    lm_head: nn.Linear


class RotationRegressionError(RuntimeError):
    """Hard failure carrying the serializable regression result for provenance."""

    def __init__(self, result: dict[str, Any]):
        self.result = result
        super().__init__(
            "Rotation logits regression failed: "
            f"relative_l2_error={result['relative_l2_error']:.6g} exceeds "
            f"threshold={result['threshold']['relative_l2_error']:.6g}"
        )


class _StreamingAbsErrorHistogram:
    """Deterministic, bounded-memory approximation of global error percentiles."""

    def __init__(self, num_bins: int = 8192, initial_max: float = 1.0) -> None:
        if num_bins <= 0 or num_bins % 2 != 0:
            raise ValueError("num_bins must be a positive even integer")
        if initial_max <= 0.0:
            raise ValueError("initial_max must be positive")
        self.num_bins = int(num_bins)
        self.range_max = float(initial_max)
        self.counts = torch.zeros(self.num_bins, dtype=torch.int64, device="cpu")
        self.total_count = 0

    def _expand_to(self, local_max: float) -> None:
        while local_max > self.range_max:
            previous_total = int(self.counts.sum().item())
            merged = self.counts.reshape(self.num_bins // 2, 2).sum(dim=1)
            new_counts = torch.zeros_like(self.counts)
            new_counts[: self.num_bins // 2] = merged
            self.counts = new_counts
            self.range_max *= 2.0
            if int(self.counts.sum().item()) != previous_total:
                raise RuntimeError("Absolute-error histogram rebin lost counts")

    def update(self, absolute_error: torch.Tensor) -> None:
        if absolute_error.numel() == 0:
            return
        local_max = float(absolute_error.max().item())
        if not math.isfinite(local_max):
            raise RuntimeError("Rotation regression produced non-finite absolute error")
        self._expand_to(local_max)
        histogram = torch.histc(
            absolute_error.float(),
            bins=self.num_bins,
            min=0.0,
            max=self.range_max,
        ).to(dtype=torch.int64, device="cpu")
        count = int(absolute_error.numel())
        if int(histogram.sum().item()) != count:
            raise RuntimeError("Absolute-error histogram did not count every logit error")
        self.counts += histogram
        self.total_count += count

    def percentile(self, q: float) -> float:
        if not 0.0 < q <= 1.0:
            raise ValueError("q must be in (0, 1]")
        if self.total_count <= 0:
            raise RuntimeError("Cannot compute a percentile from an empty histogram")
        target = math.ceil(q * self.total_count)
        cumulative = torch.cumsum(self.counts, dim=0)
        index = int(
            torch.searchsorted(
                cumulative,
                torch.tensor(target, dtype=cumulative.dtype),
            ).item()
        )
        index = min(index, self.num_bins - 1)
        bin_width = self.range_max / self.num_bins
        return float((index + 1) * bin_width)


def _exact_scalar_distribution(
    values: torch.Tensor,
    *,
    include_max: bool = True,
) -> dict[str, float]:
    values = values.detach().double().cpu().flatten()
    if values.numel() == 0:
        raise ValueError("Expected non-empty scalar diagnostic")
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError("Non-finite scalar diagnostic")
    if bool((values < 0).any()):
        raise RuntimeError("Negative scalar diagnostic")
    result = {
        "mean": float(values.mean().item()),
        "p50": float(torch.quantile(values, 0.50, interpolation="linear").item()),
        "p90": float(torch.quantile(values, 0.90, interpolation="linear").item()),
        "p99": float(torch.quantile(values, 0.99, interpolation="linear").item()),
    }
    if include_max:
        result["max"] = float(values.max().item())
    return result


def _optional_exact_scalar_distribution(
    chunks: list[torch.Tensor],
    expected_count: int,
    *,
    include_max: bool = True,
) -> dict[str, float | int | None]:
    if expected_count == 0:
        result: dict[str, float | int | None] = {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p99": None,
        }
        if include_max:
            result["max"] = None
        return result
    values = torch.cat(chunks)
    if values.numel() != expected_count:
        raise RuntimeError("Scalar diagnostic count does not match its token partition")
    return {"count": expected_count, **_exact_scalar_distribution(values, include_max=include_max)}


class _LogitsErrorAccumulator:
    def __init__(self) -> None:
        self.num_samples = 0
        self.num_tokens = 0
        self.num_elements = 0
        self.max_abs_error = 0.0
        self.sum_abs_error = 0.0
        self.sum_squared_error = 0.0
        self.sum_squared_base = 0.0
        self.sum_squared_rotated = 0.0
        self.top1_agreement_count = 0
        self.top1_disagreement_count = 0
        self.abs_error_histogram = _StreamingAbsErrorHistogram(
            num_bins=8192,
            initial_max=1.0,
        )
        self.margin_safe_token_count = 0
        self.margin_unsafe_token_count = 0
        self.margin_safe_agreement_count = 0
        self.margin_safe_disagreement_count = 0
        self.margin_unsafe_agreement_count = 0
        self.margin_unsafe_disagreement_count = 0
        self.base_top1_margin_chunks: list[torch.Tensor] = []
        self.per_token_max_abs_error_chunks: list[torch.Tensor] = []
        self.disagreement_base_top1_margin_chunks: list[torch.Tensor] = []
        self.disagreement_per_token_max_abs_error_chunks: list[torch.Tensor] = []
        self.disagreement_stability_ratio_chunks: list[torch.Tensor] = []

    def update(
        self,
        base_logits: torch.Tensor,
        rotated_logits: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        chunk_tokens: int = 16,
    ) -> None:
        if base_logits.shape != rotated_logits.shape:
            raise ValueError(
                "Base and rotated logits shapes differ: "
                f"{tuple(base_logits.shape)} != {tuple(rotated_logits.shape)}"
            )
        if attention_mask.shape != base_logits.shape[:2]:
            raise ValueError("attention_mask must match the logits batch and sequence dimensions")
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")

        device = rotated_logits.device
        mask = attention_mask.to(device=device, dtype=torch.bool)
        vocab_size = int(base_logits.shape[-1])
        if vocab_size < 2:
            raise RuntimeError("Margin-aware regression requires vocabulary size >= 2")
        valid_tokens = int(mask.sum().item())
        self.num_samples += int(base_logits.shape[0])
        self.num_tokens += valid_tokens
        self.num_elements += valid_tokens * vocab_size

        mask_cpu = attention_mask.to(device="cpu", dtype=torch.bool)
        base_top1 = base_logits.argmax(dim=-1)
        rotated_top1 = rotated_logits.argmax(dim=-1).detach().to("cpu")
        top1_matches = base_top1[mask_cpu] == rotated_top1[mask_cpu]
        agreement = int(top1_matches.sum().item())
        top1_total = int(top1_matches.numel())
        if top1_total != valid_tokens:
            raise RuntimeError("Top-1 comparison did not cover every valid token")
        self.top1_agreement_count += agreement
        self.top1_disagreement_count += top1_total - agreement

        for start in range(0, int(base_logits.shape[1]), chunk_tokens):
            stop = min(start + chunk_tokens, int(base_logits.shape[1]))
            chunk_mask = mask[:, start:stop]
            if not bool(chunk_mask.any()):
                continue
            base = base_logits[:, start:stop].to(device=device, dtype=torch.float32)[chunk_mask]
            rotated = rotated_logits[:, start:stop].to(dtype=torch.float32)[chunk_mask]
            difference = base - rotated
            absolute_error = difference.abs()
            self.max_abs_error = max(
                self.max_abs_error,
                float(absolute_error.max().item()),
            )
            self.sum_abs_error += float(absolute_error.sum(dtype=torch.float64).item())
            self.sum_squared_error += float(
                difference.square().sum(dtype=torch.float64).item()
            )
            self.sum_squared_base += float(base.square().sum(dtype=torch.float64).item())
            self.sum_squared_rotated += float(
                rotated.square().sum(dtype=torch.float64).item()
            )
            self.abs_error_histogram.update(absolute_error)

            top2 = torch.topk(base, k=2, dim=-1, largest=True, sorted=True)
            base_top1_margin = top2.values[:, 0] - top2.values[:, 1]
            per_token_max_abs_error = absolute_error.amax(dim=-1)
            if not bool(torch.isfinite(base_top1_margin).all()) or bool(
                (base_top1_margin < 0).any()
            ):
                raise RuntimeError("Base Top-1 margin is invalid")
            if not bool(torch.isfinite(per_token_max_abs_error).all()) or bool(
                (per_token_max_abs_error < 0).any()
            ):
                raise RuntimeError("Per-token maximum absolute error is invalid")

            chunk_mask_cpu = mask_cpu[:, start:stop]
            base_top1_valid = base_top1[:, start:stop][chunk_mask_cpu]
            rotated_top1_valid = rotated_top1[:, start:stop][chunk_mask_cpu]
            disagreement_mask = (base_top1_valid != rotated_top1_valid).to(device=device)
            if (
                disagreement_mask.numel() != base_top1_margin.numel()
                or disagreement_mask.numel() != per_token_max_abs_error.numel()
            ):
                raise RuntimeError("Margin diagnostics are misaligned with Top-1 decisions")

            margin_safe = base_top1_margin > 2.0 * per_token_max_abs_error
            margin_unsafe = ~margin_safe
            agreement_mask = ~disagreement_mask
            self.margin_safe_token_count += int(margin_safe.sum().item())
            self.margin_unsafe_token_count += int(margin_unsafe.sum().item())
            self.margin_safe_agreement_count += int(
                (margin_safe & agreement_mask).sum().item()
            )
            self.margin_safe_disagreement_count += int(
                (margin_safe & disagreement_mask).sum().item()
            )
            self.margin_unsafe_agreement_count += int(
                (margin_unsafe & agreement_mask).sum().item()
            )
            self.margin_unsafe_disagreement_count += int(
                (margin_unsafe & disagreement_mask).sum().item()
            )
            self.base_top1_margin_chunks.append(base_top1_margin.detach().float().cpu())
            self.per_token_max_abs_error_chunks.append(
                per_token_max_abs_error.detach().float().cpu()
            )
            if bool(disagreement_mask.any()):
                disagreement_margin = base_top1_margin[disagreement_mask]
                disagreement_delta = per_token_max_abs_error[disagreement_mask]
                disagreement_ratio = (
                    2.0 * disagreement_delta.double()
                    / (disagreement_margin.double() + 1e-12)
                )
                if not bool(torch.isfinite(disagreement_ratio).all()):
                    raise RuntimeError("Non-finite margin stability ratio")
                self.disagreement_base_top1_margin_chunks.append(
                    disagreement_margin.detach().float().cpu()
                )
                self.disagreement_per_token_max_abs_error_chunks.append(
                    disagreement_delta.detach().float().cpu()
                )
                self.disagreement_stability_ratio_chunks.append(
                    disagreement_ratio.detach().cpu()
                )

    def metrics(self) -> dict[str, Any]:
        if self.num_elements == 0:
            raise RuntimeError("Rotation regression compared no valid logits")
        if self.abs_error_histogram.total_count != self.num_elements:
            raise RuntimeError("Absolute-error histogram count does not match compared logits")
        top1_total = self.top1_agreement_count + self.top1_disagreement_count
        if top1_total != self.num_tokens:
            raise RuntimeError("Top-1 counts do not match compared tokens")
        if self.margin_safe_token_count + self.margin_unsafe_token_count != self.num_tokens:
            raise RuntimeError("Margin-safe and margin-unsafe counts do not match tokens")
        if (
            self.margin_safe_agreement_count + self.margin_safe_disagreement_count
            != self.margin_safe_token_count
        ):
            raise RuntimeError("Margin-safe agreement counts are inconsistent")
        if (
            self.margin_unsafe_agreement_count + self.margin_unsafe_disagreement_count
            != self.margin_unsafe_token_count
        ):
            raise RuntimeError("Margin-unsafe agreement counts are inconsistent")
        if (
            self.margin_safe_agreement_count + self.margin_unsafe_agreement_count
            != self.top1_agreement_count
        ):
            raise RuntimeError("Margin agreement counts do not match Top-1 agreement")
        if (
            self.margin_safe_disagreement_count + self.margin_unsafe_disagreement_count
            != self.top1_disagreement_count
        ):
            raise RuntimeError("Margin disagreement counts do not match Top-1 disagreement")
        if self.margin_safe_disagreement_count != 0:
            raise RuntimeError(
                "Margin-safe token changed Top-1 despite m_t > 2*delta_t; "
                "metric alignment is inconsistent"
            )

        all_margin = torch.cat(self.base_top1_margin_chunks)
        all_delta = torch.cat(self.per_token_max_abs_error_chunks)
        if all_margin.numel() != self.num_tokens or all_delta.numel() != self.num_tokens:
            raise RuntimeError("All-token margin diagnostics do not match compared tokens")
        disagreement_count = self.top1_disagreement_count
        margin_aware_diagnostic = {
            "definition": "base_top1_margin_gt_2x_per_token_max_abs_error",
            "margin_safe_token_count": self.margin_safe_token_count,
            "margin_unsafe_token_count": self.margin_unsafe_token_count,
            "margin_safe_fraction": self.margin_safe_token_count / self.num_tokens,
            "margin_safe_agreement_count": self.margin_safe_agreement_count,
            "margin_safe_disagreement_count": self.margin_safe_disagreement_count,
            "margin_unsafe_agreement_count": self.margin_unsafe_agreement_count,
            "margin_unsafe_disagreement_count": self.margin_unsafe_disagreement_count,
            "disagreement_margin_unsafe_fraction": (
                1.0
                if disagreement_count == 0
                else self.margin_unsafe_disagreement_count / disagreement_count
            ),
            "base_top1_margin_all_tokens": _exact_scalar_distribution(all_margin),
            "per_token_max_abs_error_all_tokens": _exact_scalar_distribution(all_delta),
            "base_top1_margin_disagreement_tokens": _optional_exact_scalar_distribution(
                self.disagreement_base_top1_margin_chunks,
                disagreement_count,
            ),
            "per_token_max_abs_error_disagreement_tokens": _optional_exact_scalar_distribution(
                self.disagreement_per_token_max_abs_error_chunks,
                disagreement_count,
            ),
            "stability_ratio_disagreement_tokens": {
                "definition": "2_delta_over_base_top1_margin_plus_1e-12",
                **_optional_exact_scalar_distribution(
                    self.disagreement_stability_ratio_chunks,
                    disagreement_count,
                    include_max=False,
                ),
            },
        }
        return {
            "num_samples": self.num_samples,
            "num_tokens_compared": self.num_tokens,
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.sum_abs_error / self.num_elements,
            "p99_abs_error": self.abs_error_histogram.percentile(0.99),
            "p999_abs_error": self.abs_error_histogram.percentile(0.999),
            "top1_agreement": self.top1_agreement_count / top1_total,
            "top1_agreement_count": self.top1_agreement_count,
            "top1_disagreement_count": self.top1_disagreement_count,
            "relative_l2_error": (
                self.sum_squared_error**0.5 / (self.sum_squared_base**0.5 + 1e-12)
            ),
            "base_logits_l2": self.sum_squared_base**0.5,
            "rotated_logits_l2": self.sum_squared_rotated**0.5,
            "absolute_error_percentile_estimator": {
                "method": "streaming_linear_histogram",
                "num_bins": self.abs_error_histogram.num_bins,
                "final_range_max": self.abs_error_histogram.range_max,
                "reported_value": "bin_upper_edge",
                "exact": False,
            },
            "margin_aware_diagnostic": margin_aware_diagnostic,
        }


def compute_logits_error_metrics(
    base_logits: torch.Tensor,
    rotated_logits: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict[str, Any]:
    """Compute regression metrics for one synthetic or real logits batch."""
    accumulator = _LogitsErrorAccumulator()
    accumulator.update(base_logits, rotated_logits, attention_mask)
    return accumulator.metrics()


def enforce_rotation_regression(
    result: dict[str, Any], relative_l2_threshold: float
) -> dict[str, Any]:
    """Attach the acceptance decision and hard-fail when rotation is not equivalent."""
    threshold = float(relative_l2_threshold)
    if threshold <= 0.0:
        raise ValueError("relative_l2_threshold must be positive")
    checked = {
        **result,
        "threshold": {"relative_l2_error": threshold},
        "passed": float(result["relative_l2_error"]) <= threshold,
    }
    if not checked["passed"]:
        raise RotationRegressionError(checked)
    return checked


def _model_input_device(model: nn.Module) -> torch.device:
    return model.get_input_embeddings().weight.device


def validate_rotation_logits(
    original_model: nn.Module,
    rotated_model: nn.Module,
    tokenizer: Any,
    calibration_dataset: Any,
    cfg: dict[str, Any],
    controller: Any,
    *,
    calibration_manifest_path: str | Path,
    calibration_manifest_sha256: str,
) -> dict[str, Any]:
    """Compare Original Base and fully integrated Rotated Base on calibration data."""
    expected_samples = int(cfg["calibration"]["num_samples"])
    if expected_samples != 128 or len(calibration_dataset) != expected_samples:
        raise RuntimeError(
            "Rotation regression requires the existing 128-sample calibration selection; "
            f"config={expected_samples}, dataset={len(calibration_dataset)}"
        )
    if getattr(controller, "mode", None) != "identity":
        raise RuntimeError("Rotation regression requires SiteController(mode='identity')")
    if not bool(getattr(rotated_model.config, "snn2_site_integration", False)):
        raise RuntimeError("Rotated model is missing SNN2 model integration")
    if set(getattr(rotated_model.config, "snn2_online_rotations", [])) != {"R3", "R4"}:
        raise RuntimeError("Rotated model regression must include online R3 and R4")

    tokenized = tokenize_dataset(calibration_dataset, tokenizer, cfg, prefix_ids=None)
    loader = DataLoader(
        tokenized,
        batch_size=int(cfg["calibration"].get("batch_size", 1)),
        shuffle=False,
        collate_fn=CausalLMCollator(tokenizer),
    )
    original_model.eval()
    rotated_model.eval()
    base_device = _model_input_device(original_model)
    rotated_device = _model_input_device(rotated_model)
    accumulator = _LogitsErrorAccumulator()

    with torch.inference_mode():
        for batch in loader:
            attention_mask = batch["attention_mask"]
            base_output = original_model(
                input_ids=batch["input_ids"].to(base_device),
                attention_mask=attention_mask.to(base_device),
                use_cache=False,
            )
            # Moving each reference batch to CPU bounds GPU memory while both
            # 8B models are resident for the strict model-to-model comparison.
            base_logits = base_output.logits.detach().to("cpu")
            del base_output
            rotated_output = rotated_model(
                input_ids=batch["input_ids"].to(rotated_device),
                attention_mask=attention_mask.to(rotated_device),
                use_cache=False,
            )
            accumulator.update(base_logits, rotated_output.logits.detach(), attention_mask)
            del base_logits, rotated_output

    result = {
        "format_version": 3,
        "purpose": "base_vs_rotated_logits_regression",
        "model_name": cfg["experiment"]["model_name"],
        "rotation_seed": int(cfg["rotation"]["seed"]),
        "dtype": cfg["training"].get("dtype", "bfloat16"),
        "calibration_manifest_path": str(Path(calibration_manifest_path).resolve()),
        "calibration_manifest_sha256": calibration_manifest_sha256,
        **accumulator.metrics(),
    }
    return enforce_rotation_regression(
        result,
        float(cfg["rotation"].get("regression_relative_l2_threshold", 0.01)),
    )

def get_model_parts(model: nn.Module) -> ModelParts:
    backbone = getattr(model, "model", None)
    if backbone is None or not hasattr(backbone, "layers"):
        raise TypeError("Only Hugging Face Llama/Qwen-style decoder-only models are supported")
    return ModelParts(
        backbone=backbone,
        layers=backbone.layers,
        embedding=backbone.embed_tokens,
        final_norm=backbone.norm,
        lm_head=model.lm_head,
    )


def fuse_rmsnorm_scale(norm: nn.Module, linears: Iterable[nn.Linear]) -> None:
    if not hasattr(norm, "weight"):
        raise TypeError(f"Expected RMSNorm-like module, got {type(norm).__name__}")
    scale = norm.weight.detach().to(dtype=torch.float64)
    for linear in linears:
        weight = linear.weight.data
        linear_scale = scale.to(device=weight.device)
        linear.weight.data = (weight.to(torch.float64) * linear_scale).to(weight.dtype)
    norm.weight.data.fill_(1.0)


def untie_input_output_embeddings(parts: ModelParts, config: Any) -> bool:
    if parts.embedding.weight.data_ptr() != parts.lm_head.weight.data_ptr():
        return False
    parts.lm_head.weight = nn.Parameter(parts.lm_head.weight.detach().clone())
    config.tie_word_embeddings = False
    return True


def _rotate_output_bias(linear: nn.Linear, spec: HadamardSpec, device: str) -> None:
    if linear.bias is not None:
        rotated = transform_weight_right(linear.bias.data.unsqueeze(0), spec, device).squeeze(0)
        linear.bias.data.copy_(rotated)


def _rotate_value_projection(attn: nn.Module, spec: HadamardSpec, device: str) -> None:
    v_proj = attn.v_proj
    kv_heads = int(attn.config.num_key_value_heads)
    head_dim = int(getattr(attn, "head_dim", attn.config.hidden_size // attn.config.num_attention_heads))
    weight = v_proj.weight.data.reshape(kv_heads, head_dim, -1)
    chunks = [transform_weight_left_transpose(chunk, spec, device) for chunk in weight]
    v_proj.weight.data.copy_(torch.stack(chunks).reshape_as(v_proj.weight.data))
    if v_proj.bias is not None:
        bias = v_proj.bias.data.reshape(kv_heads, head_dim)
        bias = random_hadamard(bias.to(device=device, dtype=torch.float32), spec)
        v_proj.bias.data.copy_(bias.to(v_proj.bias.data))


def _rotate_o_projection_input(attn: nn.Module, spec: HadamardSpec, device: str) -> None:
    o_proj = attn.o_proj
    heads = int(attn.config.num_attention_heads)
    head_dim = int(getattr(attn, "head_dim", attn.config.hidden_size // heads))
    weight = o_proj.weight.data.reshape(o_proj.out_features, heads, head_dim)
    work = weight.to(device=device, dtype=torch.float32)
    work = random_hadamard(work, spec)
    o_proj.weight.data.copy_(work.to(o_proj.weight.data).reshape_as(o_proj.weight.data))


@torch.no_grad()
def fuse_rotations(model: nn.Module, seed: int = 42, device: str = "cuda") -> dict[str, Any]:
    if getattr(model.config, "snn2_rotation_fused", False):
        raise RuntimeError("Refusing to fuse rotations twice")
    parts = get_model_parts(model)
    config = model.config
    hidden = int(config.hidden_size)
    heads = int(config.num_attention_heads)
    head_dim = int(getattr(config, "head_dim", hidden // heads))
    first_layer = next(iter(parts.layers))
    intermediate = int(first_layer.mlp.down_proj.in_features)
    specs = {
        "R1": make_spec("R1_residual", hidden, seed),
        "R2": make_spec("R2_value", head_dim, seed + 1),
        "R3": make_spec("R3_qk", head_dim, seed + 2),
        "R4": make_spec("R4_mlp", intermediate, seed + 3),
    }

    embeddings_were_untied = untie_input_output_embeddings(parts, config)

    # RMSNorm scales must be absorbed before residual-space rotation.
    for layer in parts.layers:
        fuse_rmsnorm_scale(
            layer.input_layernorm,
            [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
        )
        fuse_rmsnorm_scale(
            layer.post_attention_layernorm,
            [layer.mlp.gate_proj, layer.mlp.up_proj],
        )
    fuse_rmsnorm_scale(parts.final_norm, [parts.lm_head])

    r1, r2, r4 = specs["R1"], specs["R2"], specs["R4"]
    parts.embedding.weight.data.copy_(transform_weight_right(parts.embedding.weight.data, r1, device))
    parts.lm_head.weight.data.copy_(transform_weight_right(parts.lm_head.weight.data, r1, device))

    for layer in parts.layers:
        attn, mlp = layer.self_attn, layer.mlp
        for linear in (attn.q_proj, attn.k_proj, attn.v_proj, mlp.gate_proj, mlp.up_proj):
            linear.weight.data.copy_(transform_weight_right(linear.weight.data, r1, device))
        for linear in (attn.o_proj, mlp.down_proj):
            linear.weight.data.copy_(transform_weight_left_transpose(linear.weight.data, r1, device))
            _rotate_output_bias(linear, r1, device)

        _rotate_value_projection(attn, r2, device)
        _rotate_o_projection_input(attn, r2, device)
        mlp.down_proj.weight.data.copy_(
            transform_weight_right(mlp.down_proj.weight.data, r4, device)
        )

    model.config.snn2_rotation_fused = True
    model.config.snn2_rotation_seed = int(seed)
    model.config.snn2_online_rotations = ["R3", "R4"]
    return {
        "format_version": 1,
        "seed": int(seed),
        "sharing": {
            "R1": "global residual hidden dimension, shared across layers",
            "R2": "global head dimension, shared across heads and layers",
            "R3": "global head dimension, online after RoPE, shared across layers",
            "R4": "global MLP intermediate dimension, online before down_proj, shared across layers",
        },
        "fused_into_weights": ["R1", "R2", "R4_inverse"],
        "online": ["R3", "R4"],
        "input_output_embeddings_untied_before_fusion": embeddings_were_untied,
        "specs": {name: spec.state_dict() for name, spec in specs.items()},
    }


def save_rotation_state(state: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_rotation_state(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_specs(state: dict[str, Any]) -> dict[str, HadamardSpec]:
    return {name: HadamardSpec.from_state_dict(spec) for name, spec in state["specs"].items()}
