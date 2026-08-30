from __future__ import annotations

import math
import types
from typing import Any

import torch
import torch.nn.functional as F

from .controller import SiteController
from .hadamard import HadamardSpec, random_hadamard
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


def _record_saliency(
    controller: SiteController, layer_index: int, site_index: int,
    score: torch.Tensor, *, role: str = "default", source: str
) -> None:
    if getattr(controller, "mode", None) != "collect":
        return
    try:
        controller.record_saliency(
            layer_index, site_index, score, role=role, source=source
        )
    except TypeError:
        # Lightweight third-party/test controllers may expose the legacy sink.
        controller.record_saliency(layer_index, site_index, score)


def _record_regression(
    controller: SiteController, name: str, value: torch.Tensor
) -> None:
    recorder = getattr(controller, "regression_recorder", None)
    if recorder is not None:
        recorder.record(name, value, temporal=controller.mode.startswith("deploy_"))


def repeat_kv(hidden_states: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return hidden_states
    batch, kv_heads, length, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, kv_heads, groups, length, head_dim
    )
    return hidden_states.reshape(batch, kv_heads * groups, length, head_dim)


def merge_attention_heads(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected [B,H,L,D], got {tuple(x.shape)}")
    batch, heads, length, head_dim = x.shape
    return x.transpose(1, 2).contiguous().reshape(batch, length, heads * head_dim)


def restore_attention_heads(x: torch.Tensor, *, num_heads: int, head_dim: int) -> torch.Tensor:
    if x.ndim != 3 or x.shape[-1] != num_heads * head_dim:
        raise ValueError(
            f"Expected [B,L,{num_heads * head_dim}], got {tuple(x.shape)}"
        )
    return x.reshape(x.shape[0], x.shape[1], num_heads, head_dim).transpose(1, 2).contiguous()


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
    _record_regression(
        controller, f"layer_{layer_index:03d}/attn/q_post_rope_before_r3", query
    )
    _record_regression(
        controller, f"layer_{layer_index:03d}/attn/k_post_rope_before_r3", key
    )
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

    query = controller.apply(layer_index, 2, query)

    groups = int(getattr(module, "num_key_value_groups", 1))
    key = repeat_kv(key, groups)
    value = repeat_kv(value, groups)
    num_heads, head_dim = int(key.shape[1]), int(key.shape[-1])

    if controller.mode == "collect":
        statistics_key = key[..., past_length:, :] if past_length else key
        statistics_value = value[..., past_length:, :] if past_length else value
        controller.record_activation(layer_index, 3, statistics_key)
        controller.record_activation(layer_index, 4, statistics_value)
    else:
        key = restore_attention_heads(
            controller.apply(layer_index, 3, merge_attention_heads(key)),
            num_heads=num_heads, head_dim=head_dim,
        )
        value = restore_attention_heads(
            controller.apply(layer_index, 4, merge_attention_heads(value)),
            num_heads=num_heads, head_dim=head_dim,
        )

    if controller.mode == "collect":
        q64 = query.detach().to(torch.float64)
        k64 = key.detach().to(torch.float64)
        qk64 = torch.matmul(q64, k64.transpose(-2, -1))
        key_score = k64 * torch.matmul(qk64.transpose(-2, -1), q64)
        if past_length:
            key_score = key_score[..., past_length:, :]
        _record_saliency(
            controller, layer_index, 3, key_score, source="spikellm_qk_k_fp64"
        )

    qk = torch.matmul(query, key.transpose(2, 3))
    scale = float(scaling if scaling is not None else getattr(module, "scaling", 1.0 / math.sqrt(query.shape[-1])))
    weights = qk * scale
    _record_regression(controller, f"layer_{layer_index:03d}/attn/qk_scaled", weights)
    if attention_mask is not None:
        weights = weights + attention_mask[..., : key.shape[-2]]
    if kwargs.get("softcap") is not None:
        cap = float(kwargs["softcap"])
        weights = torch.tanh(weights / cap) * cap
    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    _record_regression(controller, f"layer_{layer_index:03d}/attn/softmax_before_site5", weights)
    if controller.mode == "collect":
        statistics_weights = weights[..., past_length:] if past_length else weights
        controller.record_activation(layer_index, 5, statistics_weights)
    else:
        weights = controller.apply(layer_index, 5, weights)
    weights = F.dropout(weights, p=dropout, training=module.training)

    if controller.mode == "collect":
        p64 = weights.detach().to(torch.float64)
        v64 = value.detach().to(torch.float64)
        pv64 = torch.matmul(p64, v64)
        value_score = v64 * torch.matmul(p64.transpose(-2, -1), pv64)
        if past_length:
            value_score = value_score[..., past_length:, :]
        _record_saliency(
            controller, layer_index, 4, value_score, source="spikellm_pv_v_fp64"
        )

    output_heads = torch.matmul(weights, value)
    _record_regression(controller, f"layer_{layer_index:03d}/attn/pv_head_output_before_merge", output_heads)
    output = merge_attention_heads(output_heads)
    _record_regression(controller, f"layer_{layer_index:03d}/attn/pv_merged_before_site6", output)
    output = controller.apply(layer_index, 6, output)
    return output, weights


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
        gate_projection = mlp.gate_proj(x)
        _record_regression(
            controller, f"layer_{layer_index:03d}/mlp/gate_proj", gate_projection
        )
        up_projection = mlp.up_proj(x)
        _record_regression(
            controller, f"layer_{layer_index:03d}/mlp/up_proj", up_projection
        )
        if controller.mode.startswith("deploy_"):
            steps = int(controller.temporal_steps or 0)
            gate = from_temporal(temporal_silu(to_temporal(gate_projection, steps)))
            gate = controller.apply(layer_index, 8, gate)
            up = controller.apply(layer_index, 9, up_projection)
            product = from_temporal(
                temporal_symmetric_hadamard(
                    to_temporal(gate, steps), to_temporal(up, steps)
                )
            )
        else:
            gate = mlp.act_fn(gate_projection)
            gate = controller.apply(layer_index, 8, gate)
            up = up_projection
            up = controller.apply(layer_index, 9, up)
            product = gate * up
        if r4 is not None:
            product_dtype = product.dtype
            product = random_hadamard(product.to(torch.float32), r4).to(product_dtype)
        _record_regression(
            controller,
            f"layer_{layer_index:03d}/mlp/product_before_site10",
            product,
        )
        product = controller.apply(layer_index, 10, product)
        output = mlp.down_proj(product)
        _record_regression(
            controller, f"layer_{layer_index:03d}/mlp/down_proj_output", output
        )
        return output

    return forward


def _linear_score(
    inputs: torch.Tensor, output_or_weight: torch.Tensor, weight: torch.Tensor | None = None
) -> torch.Tensor:
    """SpikeLLM linear-consumer saliency, recomputed entirely in FP32."""
    weight = output_or_weight if weight is None else weight
    x32 = inputs.detach().to(torch.float32)
    w32 = weight.detach().to(torch.float32)
    projected = torch.matmul(x32, w32.transpose(-1, -2))
    return torch.matmul(projected, w32) * x32


def record_down_proj_saliency(
    controller: SiteController,
    layer_index: int,
    inputs: tuple[torch.Tensor, ...],
    output: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    """Record the R4 product consumer sensitivity at Site 10."""
    _record_saliency(
        controller, layer_index, 10, _linear_score(inputs[0], weight),
        source="spikellm_linear_fp32",
    )


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
            _record_regression(controller, "embedding/output", output)
            return output
        steps = int(controller.temporal_steps or 0)
        if steps <= 0 or output.shape[0] % steps != 0:
            raise ValueError("Temporal embedding batch is incompatible with deployment steps")
        batch = output.shape[0] // steps
        temporal = output.reshape(steps, batch, *output.shape[1:]) / steps
        output = temporal.reshape_as(output)
        _record_regression(controller, "embedding/output", output)
        return output

    handles.append(parts.embedding.register_forward_hook(temporal_embedding_hook))
    handles.append(
        parts.final_norm.register_forward_hook(
            lambda _module, _inputs, output: controller.apply_final_norm_phase(output)
        )
    )

    def lm_head_regression_hook(_module, _inputs, output):
        _record_regression(controller, "lm_head/output", output)

    def model_regression_hook(_module, _inputs, output):
        logits = getattr(output, "logits", None)
        if logits is not None:
            _record_regression(controller, "model/logits", logits)

    handles.append(parts.lm_head.register_forward_hook(lm_head_regression_hook))
    handles.append(model.register_forward_hook(model_regression_hook))

    for layer_index, layer in enumerate(parts.layers):
        attention = layer.self_attn
        attention._snn2_controller = controller
        attention._snn2_layer_index = layer_index
        attention._snn2_r3 = r3

        def q_norm_regression_hook(_module, _inputs, output, index=layer_index):
            _record_regression(
                controller, f"layer_{index:03d}/attn/q_after_norm", output
            )

        def k_norm_regression_hook(_module, _inputs, output, index=layer_index):
            _record_regression(
                controller, f"layer_{index:03d}/attn/k_after_norm", output
            )

        q_norm = getattr(attention, "q_norm", None)
        k_norm = getattr(attention, "k_norm", None)
        if q_norm is not None:
            handles.append(q_norm.register_forward_hook(q_norm_regression_hook))
        if k_norm is not None:
            handles.append(k_norm.register_forward_hook(k_norm_regression_hook))

        def layer_input_hook(_module, inputs, index=layer_index, attn=attention):
            if getattr(controller, "regression_recorder", None) is not None:
                attn._snn2_regression_residual = inputs[0]
                _record_regression(controller, f"layer_{index:03d}/input", inputs[0])

        def layer_output_hook(_module, _inputs, output, index=layer_index):
            hidden = output[0] if isinstance(output, tuple) else output
            _record_regression(controller, f"layer_{index:03d}/output", hidden)

        handles.append(layer.register_forward_pre_hook(layer_input_hook))
        handles.append(layer.register_forward_hook(layer_output_hook))

        def norm1_hook(_module, _inputs, output, index=layer_index):
            if controller.mode in {"gif", "deploy_gif"}:
                controller.record_activation(index, 1, output)
                return output
            return controller.apply(index, 1, output)

        def norm2_hook(_module, _inputs, output, index=layer_index):
            if controller.mode in {"gif", "deploy_gif"}:
                controller.record_activation(index, 7, output)
                return output
            return controller.apply(index, 7, output)

        handles.append(layer.input_layernorm.register_forward_hook(norm1_hook))
        handles.append(layer.post_attention_layernorm.register_forward_hook(norm2_hook))

        def make_gif_branch_pre_hook(site_index, role, index=layer_index):
            def hook(_module, inputs):
                if controller.mode in {"gif", "deploy_gif"}:
                    replaced = controller.apply(index, site_index, inputs[0], gif_role=role)
                    return (replaced, *inputs[1:])
                if (
                    controller.mode == "phase"
                    and controller.common_clip_enabled
                    and site_index in {1, 7}
                ):
                    replaced = controller.apply_role_clip(
                        index, site_index, inputs[0], role=role
                    )
                    return (replaced, *inputs[1:])
                return None
            return hook

        def make_branch_linear_hook(label, index=layer_index):
            def branch_linear_hook(_module, inputs, output):
                _record_regression(
                    controller, f"layer_{index:03d}/attn/{label}_proj_output", output
                )
                if controller.mode == "collect":
                    _record_saliency(
                        controller, index, 1, _linear_score(inputs[0], _module.weight),
                        role=label, source="spikellm_linear_fp32",
                    )
            return branch_linear_hook

        for label, projection in (("q", attention.q_proj), ("k", attention.k_proj), ("v", attention.v_proj)):
            handles.append(projection.register_forward_pre_hook(
                make_gif_branch_pre_hook(1, label)
            ))
            handles.append(projection.register_forward_hook(make_branch_linear_hook(label)))

        def output_linear_hook(_module, inputs, output, index=layer_index, attn=attention):
            _record_regression(
                controller, f"layer_{index:03d}/attn/o_proj_output", output
            )
            residual = getattr(attn, "_snn2_regression_residual", None)
            if residual is not None:
                _record_regression(
                    controller,
                    f"layer_{index:03d}/post_attention_residual",
                    residual + output,
                )
            if controller.mode == "collect":
                score = _linear_score(inputs[0], _module.weight)
                _record_saliency(
                    controller, index, 6, score, source="spikellm_linear_fp32"
                )

        handles.append(attention.o_proj.register_forward_hook(output_linear_hook))

        def make_mlp_input_hook(role, index=layer_index):
            def hook(_module, inputs, output):
                if controller.mode == "collect":
                    _record_saliency(
                        controller, index, 7, _linear_score(inputs[0], _module.weight),
                        role=role, source="spikellm_linear_fp32",
                    )
            return hook

        for role, projection in (("gate", layer.mlp.gate_proj), ("up", layer.mlp.up_proj)):
            handles.append(projection.register_forward_pre_hook(
                make_gif_branch_pre_hook(7, role)
            ))
            handles.append(projection.register_forward_hook(make_mlp_input_hook(role)))

        def down_input_hook(_module, inputs, output, index=layer_index):
            if controller.mode == "collect":
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
