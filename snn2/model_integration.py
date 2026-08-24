from __future__ import annotations

import math
import types
from typing import Any

import torch
import torch.nn.functional as F

from .controller import SiteController
from .hadamard import HadamardSpec, random_hadamard
from .phase_statistics import phase_statistical_view
from .rotation import get_model_parts, load_specs
from .temporal_model import deployment_attention_forward
from .temporal_ops import (
    from_temporal,
    temporal_bias_once,
    temporal_rmsnorm,
    temporal_silu,
    temporal_symmetric_hadamard,
    to_temporal,
)


def repeat_kv(hidden_states: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return hidden_states
    batch, kv_heads, length, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, kv_heads, groups, length, head_dim
    )
    return hidden_states.reshape(batch, kv_heads * groups, length, head_dim)


def snn2_eager_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
):
    controller: SiteController = module._snn2_controller
    layer_index = int(module._snn2_layer_index)
    current_length = int(query.shape[-2])
    past_length = max(int(key.shape[-2]) - current_length, 0)

    r3: HadamardSpec | None = getattr(module, "_snn2_r3", None)
    if r3 is not None:
        # RoSTE-aligned R3 preserves the activation dtype. Unsupported CUDA
        # dtypes fail in the pinned FHT backend instead of silently falling back.
        query = random_hadamard(query.contiguous(), r3)
        key = random_hadamard(key.contiguous(), r3)

    if controller.mode.startswith("deploy_"):
        return deployment_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            dropout=dropout,
            controller=controller,
            layer_index=layer_index,
            repeat_kv=repeat_kv,
            softcap=kwargs.get("softcap"),
        )

    query = controller.apply(
        layer_index,
        2,
        query,
        phase_activation=(
            phase_statistical_view(2, query) if controller.mode == "collect" else None
        ),
    )

    groups = int(getattr(module, "num_key_value_groups", 1))
    if controller.mode == "collect":
        statistics_key = key[..., past_length:, :] if past_length else key
        statistics_value = value[..., past_length:, :] if past_length else value
        phase_key = repeat_kv(statistics_key, groups)
        phase_value = repeat_kv(statistics_value, groups)
        controller.record_activation(
            layer_index,
            3,
            statistics_key,
            phase_activation=phase_statistical_view(3, phase_key),
        )
        controller.record_activation(
            layer_index,
            4,
            statistics_value,
            phase_activation=phase_statistical_view(4, phase_value),
        )
    else:
        key = controller.apply(layer_index, 3, key)
        value = controller.apply(layer_index, 4, value)

    key = repeat_kv(key, groups)
    value = repeat_kv(value, groups)
    qk = torch.matmul(query, key.transpose(2, 3))
    if controller.mode == "collect":
        controller.record_saliency(layer_index, 2, query * torch.matmul(qk, key))
        key_score = key * torch.matmul(qk.transpose(2, 3), query)
        if past_length:
            key_score = key_score[..., past_length:, :]
        controller.record_saliency(layer_index, 3, key_score)
    scale = float(scaling if scaling is not None else getattr(module, "scaling", 1.0 / math.sqrt(query.shape[-1])))
    weights = qk * scale
    if attention_mask is not None:
        weights = weights + attention_mask[..., : key.shape[-2]]
    if kwargs.get("softcap") is not None:
        cap = float(kwargs["softcap"])
        weights = torch.tanh(weights / cap) * cap
    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = controller.apply(
        layer_index,
        5,
        weights,
        phase_activation=(
            phase_statistical_view(5, weights)
            if controller.mode == "collect"
            else None
        ),
    )
    weights = F.dropout(weights, p=dropout, training=module.training)
    output = torch.matmul(weights, value)
    if controller.mode == "collect":
        value_score = value * torch.matmul(weights.transpose(2, 3), output)
        if past_length:
            value_score = value_score[..., past_length:, :]
        controller.record_saliency(layer_index, 4, value_score)
        position_score = torch.zeros(
            weights.shape[-1], device=weights.device, dtype=torch.float32
        )
        chunk_size = 64
        for start in range(0, weights.shape[-2], chunk_size):
            stop = min(start + chunk_size, weights.shape[-2])
            back = torch.matmul(output[:, :, start:stop], value.transpose(2, 3))
            position_score.add_(
                (weights[:, :, start:stop].float() * back.float()).sum(dim=(0, 1, 2))
            )
        controller.record_saliency_reduced(
            layer_index,
            5,
            position_score,
            weights.shape[0] * weights.shape[1] * weights.shape[2],
        )
    output = controller.apply(
        layer_index,
        6,
        output,
        phase_activation=(
            phase_statistical_view(6, output)
            if controller.mode == "collect"
            else None
        ),
    )
    return output.transpose(1, 2).contiguous(), weights


def register_attention_backend() -> None:
    # Transformers 4.53+ dispatches causal-mask construction separately from
    # the attention callable. Without this mapping, custom backends receive no
    # causal mask and identity integration becomes non-causal.
    try:
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask

        ALL_MASK_ATTENTION_FUNCTIONS.register("snn2_eager", eager_mask)
    except ImportError:
        # Older supported Transformers versions construct eager masks directly.
        pass
    try:
        from transformers import AttentionInterface

        try:
            AttentionInterface.register("snn2_eager", snn2_eager_attention_forward)
        except ValueError as exc:
            if "already" not in str(exc).lower():
                raise
        return
    except (ImportError, AttributeError):
        pass
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS["snn2_eager"] = snn2_eager_attention_forward
    except (ImportError, TypeError) as exc:
        raise RuntimeError(
            "This Transformers version does not expose the AttentionInterface registry"
        ) from exc


def _make_mlp_forward(controller: SiteController, layer_index: int, r4: HadamardSpec | None):
    def forward(mlp, x: torch.Tensor):
        if controller.mode.startswith("deploy_"):
            steps = int(controller.temporal_steps or 0)
            gate = from_temporal(temporal_silu(to_temporal(mlp.gate_proj(x), steps)))
            gate = controller.apply(layer_index, 8, gate)
            up = controller.apply(layer_index, 9, mlp.up_proj(x))
            product = from_temporal(
                temporal_symmetric_hadamard(
                    to_temporal(gate, steps), to_temporal(up, steps)
                )
            )
        else:
            gate = mlp.act_fn(mlp.gate_proj(x))
            gate = controller.apply(layer_index, 8, gate)
            up = mlp.up_proj(x)
            up = controller.apply(layer_index, 9, up)
            if controller.mode == "collect":
                product_saliency = gate.square() * up.square()
                controller.record_saliency(layer_index, 8, product_saliency)
                controller.record_saliency(layer_index, 9, product_saliency)
            product = gate * up
        if r4 is not None:
            product_dtype = product.dtype
            product = random_hadamard(product.to(torch.float32), r4).to(product_dtype)
        product = controller.apply(layer_index, 10, product)
        return mlp.down_proj(product)

    return forward


def _linear_score(
    inputs: torch.Tensor, output: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return inputs * torch.matmul(output, weight)


def record_down_proj_saliency(
    controller: SiteController,
    layer_index: int,
    inputs: tuple[torch.Tensor, ...],
    output: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    """Record the R4 product consumer sensitivity at Site 10."""
    controller.record_saliency(layer_index, 10, _linear_score(inputs[0], output, weight))


def _install_temporal_rmsnorm(
    norm: torch.nn.Module, controller: SiteController
) -> None:
    if hasattr(norm, "_snn2_original_forward"):
        raise RuntimeError("Temporal RMSNorm integration is already installed")
    original_forward = norm.forward

    def forward(module, x: torch.Tensor, *args: Any, **kwargs: Any):
        if not controller.mode.startswith("deploy_"):
            return original_forward(x, *args, **kwargs)
        steps = int(controller.temporal_steps or 0)
        return from_temporal(temporal_rmsnorm(to_temporal(x, steps), module))

    norm._snn2_original_forward = original_forward
    norm.forward = types.MethodType(forward, norm)



def install_model_integration(
    model: torch.nn.Module,
    controller: SiteController,
    rotation_state: dict[str, Any] | None,
) -> None:
    register_attention_backend()
    specs = load_specs(rotation_state) if rotation_state is not None else {}
    r3, r4 = specs.get("R3"), specs.get("R4")
    parts = get_model_parts(model)
    handles = getattr(model, "_snn2_handles", [])
    if handles:
        raise RuntimeError("SNN2 model integration is already installed")

    wrapped_norms = []
    seen_norms: set[int] = set()
    norm_candidates = [parts.final_norm]
    for layer in parts.layers:
        norm_candidates.extend(
            [
                layer.input_layernorm,
                layer.post_attention_layernorm,
                getattr(layer.self_attn, "q_norm", None),
                getattr(layer.self_attn, "k_norm", None),
            ]
        )
    for norm in norm_candidates:
        if norm is None or id(norm) in seen_norms:
            continue
        seen_norms.add(id(norm))
        _install_temporal_rmsnorm(norm, controller)
        wrapped_norms.append(norm)

    def temporal_linear_bias_hook(module, _inputs, output):
        if not controller.mode.startswith("deploy_") or module.bias is None:
            return output
        return temporal_bias_once(
            output, module.bias, int(controller.temporal_steps or 0)
        )

    seen_linears: set[int] = set()
    for module in model.modules():
        if (
            isinstance(module, torch.nn.Linear)
            and module.bias is not None
            and id(module) not in seen_linears
        ):
            seen_linears.add(id(module))
            handles.append(module.register_forward_hook(temporal_linear_bias_hook))

    def temporal_embedding_hook(_module, _inputs, output):
        if not controller.mode.startswith("deploy_"):
            return output
        steps = int(controller.temporal_steps or 0)
        if steps <= 0 or output.shape[0] % steps != 0:
            raise ValueError("Temporal embedding batch is incompatible with deployment steps")
        batch = output.shape[0] // steps
        temporal = output.reshape(steps, batch, *output.shape[1:]) / steps
        return temporal.reshape_as(output)

    handles.append(parts.embedding.register_forward_hook(temporal_embedding_hook))
    handles.append(
        parts.final_norm.register_forward_hook(
            lambda _module, _inputs, output: controller.apply_final_norm_phase(output)
        )
    )

    for layer_index, layer in enumerate(parts.layers):
        attention = layer.self_attn
        attention._snn2_controller = controller
        attention._snn2_layer_index = layer_index
        attention._snn2_r3 = r3

        def norm1_hook(_module, _inputs, output, index=layer_index):
            return controller.apply(index, 1, output)

        def norm2_hook(_module, _inputs, output, index=layer_index):
            return controller.apply(index, 7, output)

        handles.append(layer.input_layernorm.register_forward_hook(norm1_hook))
        handles.append(layer.post_attention_layernorm.register_forward_hook(norm2_hook))

        def branch_linear_hook(_module, inputs, output, index=layer_index):
            controller.record_saliency(index, 1, _linear_score(inputs[0], output, _module.weight))

        for projection in (attention.q_proj, attention.k_proj, attention.v_proj):
            handles.append(projection.register_forward_hook(branch_linear_hook))

        def output_linear_hook(_module, inputs, output, index=layer_index, attn=attention):
            score = _linear_score(inputs[0], output, _module.weight)
            heads = getattr(attn, "num_heads", None)
            if heads is None:
                heads = getattr(attn, "num_attention_heads", None)
                heads = attn.config.num_attention_heads
            heads = int(heads)
            score = score.reshape(score.shape[0], score.shape[1], heads, -1).transpose(1, 2)
            controller.record_saliency(index, 6, score)

        handles.append(attention.o_proj.register_forward_hook(output_linear_hook))

        def mlp_input_hook(_module, inputs, output, index=layer_index):
            controller.record_saliency(index, 7, _linear_score(inputs[0], output, _module.weight))

        handles.append(layer.mlp.gate_proj.register_forward_hook(mlp_input_hook))
        handles.append(layer.mlp.up_proj.register_forward_hook(mlp_input_hook))

        def down_input_hook(_module, inputs, output, index=layer_index):
            record_down_proj_saliency(controller, index, inputs, output, _module.weight)

        handles.append(layer.mlp.down_proj.register_forward_hook(down_input_hook))
        layer.mlp.forward = types.MethodType(_make_mlp_forward(controller, layer_index, r4), layer.mlp)

    model._snn2_handles = handles
    model._snn2_wrapped_norms = wrapped_norms
    model.config._attn_implementation = "snn2_eager"
    model.config._attn_implementation_internal = "snn2_eager"
    model.config.snn2_site_integration = True


def _repeat_temporal_batch_tensor(
    value: torch.Tensor,
    *,
    steps: int,
    batch: int,
    name: str,
) -> torch.Tensor:
    """Expand a batch tensor in time-major order, failing on ambiguous shapes."""
    if value.ndim == 0:
        raise ValueError(f"{name} cannot be scalar")
    if value.shape[0] == steps * batch:
        temporal = value.reshape(steps, batch, *value.shape[1:])
        if any(
            not torch.equal(temporal[0], temporal[timestep])
            for timestep in range(1, steps)
        ):
            raise ValueError(f"{name} must be identical in every temporal frame")
        return value
    if value.shape[0] == batch:
        return value.repeat(steps, *([1] * (value.ndim - 1)))
    if value.shape[0] == 1:
        expanded = value.expand(batch, *value.shape[1:])
        return expanded.repeat(steps, *([1] * (value.ndim - 1)))
    raise ValueError(
        f"{name} leading dimension is incompatible with T={steps}, B={batch}"
    )


def temporal_forward(
    model: torch.nn.Module,
    controller: SiteController,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    **kwargs: Any,
):
    if model.training:
        raise RuntimeError("Temporal deployment requires model.eval()")
    if controller.temporal_steps is None:
        raise RuntimeError("Controller deployment timestep is unset")
    steps = int(controller.temporal_steps)
    batch = int(input_ids.shape[0])
    repeated_ids = input_ids.repeat(steps, 1)
    repeated_mask = attention_mask.repeat(steps, 1)
    model_kwargs = dict(kwargs)
    if model_kwargs.get("position_ids") is not None:
        position_ids = model_kwargs["position_ids"]
        if position_ids.ndim != 2 or position_ids.shape[-1] != input_ids.shape[-1]:
            raise ValueError("position_ids must have shape [B, L] or [T*B, L]")
        model_kwargs["position_ids"] = _repeat_temporal_batch_tensor(
            position_ids,
            steps=steps,
            batch=batch,
            name="position_ids",
        )
    cache_position = model_kwargs.get("cache_position")
    if cache_position is not None and (
        cache_position.ndim != 1 or cache_position.shape[0] != input_ids.shape[-1]
    ):
        raise ValueError("cache_position must be one-dimensional with length L")
    outputs = model(
        input_ids=repeated_ids,
        attention_mask=repeated_mask,
        use_cache=False,
        **model_kwargs,
    )
    logits = outputs.logits.reshape(steps, batch, *outputs.logits.shape[1:]).sum(dim=0)
    return logits
