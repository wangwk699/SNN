from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

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
