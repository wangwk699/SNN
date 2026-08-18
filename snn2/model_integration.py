from __future__ import annotations

import math
import types
from typing import Any

import torch
import torch.nn.functional as F

from .controller import SiteController
from .hadamard import HadamardSpec, random_hadamard
from .rotation import get_model_parts, load_specs


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
        query = random_hadamard(query, r3)
        key = random_hadamard(key, r3)
    query = controller.apply(layer_index, 2, query)

    if past_length:
        prefix_key, current_key = key[..., :past_length, :], key[..., past_length:, :]
        prefix_value, current_value = value[..., :past_length, :], value[..., past_length:, :]
        current_key = controller.apply(layer_index, 3, current_key)
        current_value = controller.apply(layer_index, 4, current_value)
        key = torch.cat((prefix_key, current_key), dim=-2)
        value = torch.cat((prefix_value, current_value), dim=-2)
    else:
        key = controller.apply(layer_index, 3, key)
        value = controller.apply(layer_index, 4, value)

    groups = int(getattr(module, "num_key_value_groups", 1))
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
    if past_length:
        prefix_weights = weights[..., :past_length]
        current_weights = controller.apply(layer_index, 5, weights[..., past_length:])
        weights = torch.cat((prefix_weights, current_weights), dim=-1)
    else:
        weights = controller.apply(layer_index, 5, weights)
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
        if past_length:
            position_score = position_score[past_length:]
        controller.record_saliency_reduced(
            layer_index,
            5,
            position_score,
            weights.shape[0] * weights.shape[1] * weights.shape[2],
        )
    output = controller.apply(layer_index, 6, output)
    return output.transpose(1, 2).contiguous(), weights


def register_attention_backend() -> None:
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
            product = random_hadamard(product, r4)
        product = controller.apply(layer_index, 10, product)
        return mlp.down_proj(product)

    return forward


def _linear_score(inputs: torch.Tensor, output: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return inputs * torch.matmul(output, weight)


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

    def temporal_embedding_hook(_module, _inputs, output):
        if not controller.mode.startswith("deploy_"):
            return output
        steps = int(controller.temporal_steps or 0)
        if steps <= 0 or output.shape[0] % steps != 0:
            raise ValueError("Temporal embedding batch is incompatible with deployment steps")
        batch = output.shape[0] // steps
        temporal = output.reshape(steps, batch, *output.shape[1:]).clone()
        temporal[1:] = 0
        return temporal.reshape_as(output)

    handles.append(parts.embedding.register_forward_hook(temporal_embedding_hook))

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
            controller.record_saliency(index, 10, _linear_score(inputs[0], output, _module.weight))

        handles.append(layer.mlp.down_proj.register_forward_hook(down_input_hook))
        layer.mlp.forward = types.MethodType(_make_mlp_forward(controller, layer_index, r4), layer.mlp)

    model._snn2_handles = handles
    model.config._attn_implementation = "snn2_eager"
    model.config._attn_implementation_internal = "snn2_eager"
    model.config.snn2_site_integration = True


def temporal_forward(
    model: torch.nn.Module,
    controller: SiteController,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    **kwargs: Any,
):
    if controller.temporal_steps is None:
        raise RuntimeError("Controller deployment timestep is unset")
    steps = controller.temporal_steps
    repeated_ids = input_ids.repeat(steps, 1)
    repeated_mask = attention_mask.repeat(steps, 1)
    outputs = model(
        input_ids=repeated_ids,
        attention_mask=repeated_mask,
        use_cache=False,
        **kwargs,
    )
    logits = outputs.logits.reshape(steps, input_ids.shape[0], *outputs.logits.shape[1:]).sum(dim=0)
    return logits
