from __future__ import annotations

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
        valid_tokens = int(mask.sum().item())
        self.num_samples += int(base_logits.shape[0])
        self.num_tokens += valid_tokens
        self.num_elements += valid_tokens * vocab_size

        for start in range(0, int(base_logits.shape[1]), chunk_tokens):
            stop = min(start + chunk_tokens, int(base_logits.shape[1]))
            chunk_mask = mask[:, start:stop]
            if not bool(chunk_mask.any()):
                continue
            base = base_logits[:, start:stop].to(device=device, dtype=torch.float32)[chunk_mask]
            rotated = rotated_logits[:, start:stop].to(dtype=torch.float32)[chunk_mask]
            difference = base - rotated
            self.max_abs_error = max(
                self.max_abs_error,
                float(difference.abs().max().item()),
            )
            self.sum_abs_error += float(difference.abs().sum(dtype=torch.float64).item())
            self.sum_squared_error += float(
                difference.square().sum(dtype=torch.float64).item()
            )
            self.sum_squared_base += float(base.square().sum(dtype=torch.float64).item())
            self.sum_squared_rotated += float(
                rotated.square().sum(dtype=torch.float64).item()
            )

    def metrics(self) -> dict[str, Any]:
        if self.num_elements == 0:
            raise RuntimeError("Rotation regression compared no valid logits")
        return {
            "num_samples": self.num_samples,
            "num_tokens_compared": self.num_tokens,
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.sum_abs_error / self.num_elements,
            "relative_l2_error": (
                self.sum_squared_error**0.5 / (self.sum_squared_base**0.5 + 1e-12)
            ),
            "base_logits_l2": self.sum_squared_base**0.5,
            "rotated_logits_l2": self.sum_squared_rotated**0.5,
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
        "format_version": 1,
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
        dtype = linear.weight.dtype
        linear.weight.data = (linear.weight.data.to(torch.float64) * scale).to(dtype)
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
