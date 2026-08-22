from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import torch


def _as_legacy_cache(past_key_values: Any):
    if past_key_values is None:
        return ()
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    if not isinstance(past_key_values, (tuple, list)):
        raise TypeError(
            "Unsupported past_key_values type for Prefix cache: "
            f"{type(past_key_values)!r}"
        )
    legacy = []
    for layer in past_key_values:
        if not isinstance(layer, (tuple, list)) or len(layer) < 2:
            raise TypeError("Each Prefix cache layer must contain key and value tensors")
        key, value = layer[0], layer[1]
        legacy.append((key.detach(), value.detach()))
    return tuple(legacy)


@torch.no_grad()
def build_prefix_key_values(model: torch.nn.Module, prefix_ids: list[int]):
    ids = [int(value) for value in prefix_ids]
    if not ids:
        return None
    device = next(model.parameters()).device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    was_training = model.training
    model.eval()
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    finally:
        model.train(was_training)
    cache = _as_legacy_cache(outputs.past_key_values)
    if not cache:
        raise RuntimeError("Model did not return past_key_values for Prefix tokens")
    return tuple((key.cpu(), value.cpu()) for key, value in cache)


def save_prefix_key_values(path: str | Path, prefix_key_values) -> None:
    output = Path(path)
    if prefix_key_values is None:
        if output.exists():
            output.unlink()
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prefix_key_values, output)


def load_prefix_key_values(path: str | Path):
    value = torch.load(Path(path), map_location="cpu", weights_only=False)
    cache = _as_legacy_cache(value)
    if not cache:
        raise RuntimeError(f"Empty Prefix KV cache: {path}")
    return tuple((key.cpu(), val.cpu()) for key, val in cache)


def prefix_length(prefix_key_values) -> int:
    if not prefix_key_values:
        return 0
    return int(prefix_key_values[0][0].shape[-2])


def _layer_devices(model, layer_count, fallback):
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None or len(layers) != layer_count:
        return [fallback] * layer_count
    result = []
    for layer in layers:
        try:
            result.append(next(layer.parameters()).device)
        except StopIteration:
            result.append(fallback)
    return result


def _align_prefix_key_values(model, prefix_key_values, fallback):
    devices = _layer_devices(model, len(prefix_key_values), fallback)
    return tuple(
        (key.to(device), value.to(device))
        for (key, value), device in zip(prefix_key_values, devices)
    )


def _prefix_for_logical_batch(
    tensor: torch.Tensor, logical_batch_size: int
) -> torch.Tensor:
    saved_batch = int(tensor.shape[0])
    if saved_batch == 1:
        return tensor.expand(logical_batch_size, *tensor.shape[1:])
    if saved_batch != logical_batch_size:
        raise ValueError(
            f"Prefix cache batch={saved_batch} is incompatible with logical "
            f"batch={logical_batch_size}"
        )
    return tensor


def _fresh_dynamic_cache(
    prefix_key_values,
    *,
    logical_batch_size: int,
    temporal_steps: int | None = None,
):
    from transformers.cache_utils import DynamicCache

    logical_batch_size = int(logical_batch_size)
    if logical_batch_size <= 0:
        raise ValueError("logical_batch_size must be positive")
    steps = None if temporal_steps is None else int(temporal_steps)
    if steps is not None and steps <= 0:
        raise ValueError("temporal_steps must be positive")
    expanded = []
    for key, value in prefix_key_values:
        base_key = _prefix_for_logical_batch(key, logical_batch_size)
        base_value = _prefix_for_logical_batch(value, logical_batch_size)
        if steps is None:
            expanded.append((base_key, base_value))
            continue
        temporal_key = (
            base_key.unsqueeze(0)
            .expand(steps, logical_batch_size, *base_key.shape[1:])
            .div(steps)
            .reshape(steps * logical_batch_size, *base_key.shape[1:])
        )
        temporal_value = (
            base_value.unsqueeze(0)
            .expand(steps, logical_batch_size, *base_value.shape[1:])
            .div(steps)
            .reshape(steps * logical_batch_size, *base_value.shape[1:])
        )
        expanded.append((temporal_key, temporal_value))
    return DynamicCache.from_legacy_cache(tuple(expanded))


def _extend_attention_mask(
    attention_mask,
    *,
    batch_size: int,
    current_length: int,
    cached_prefix_length: int,
    device,
):
    if attention_mask is None or cached_prefix_length <= 0:
        return attention_mask
    if attention_mask.shape[-1] == current_length + cached_prefix_length:
        return attention_mask
    if attention_mask.shape[-1] != current_length:
        raise ValueError(
            "Attention-mask length is incompatible with Prefix KV injection: "
            f"mask={attention_mask.shape[-1]}, current={current_length}, "
            f"prefix={cached_prefix_length}"
        )
    prefix_mask = torch.ones(
        (batch_size, cached_prefix_length),
        dtype=attention_mask.dtype,
        device=device,
    )
    return torch.cat((prefix_mask, attention_mask), dim=-1)


def install_prefix_kv_forward(
    model: torch.nn.Module, prefix_key_values, *, controller: Any | None = None
) -> None:
    """Inject fixed Prefix K/V while leaving input_ids unchanged."""
    if not prefix_key_values:
        return
    if hasattr(model, "_snn2_prefix_original_forward"):
        raise RuntimeError("Prefix KV forward injection is already installed")

    frozen = tuple(
        (key.detach().cpu(), value.detach().cpu())
        for key, value in prefix_key_values
    )
    cached_prefix_length = prefix_length(frozen)
    original_forward = model.forward
    aligned = None

    @functools.wraps(original_forward)
    def wrapped_forward(*args: Any, **kwargs: Any):
        nonlocal aligned
        if kwargs.get("past_key_values") is not None:
            return original_forward(*args, **kwargs)

        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        inputs_embeds = kwargs.get("inputs_embeds")

        if input_ids is not None:
            batch_size = int(input_ids.shape[0])
            current_length = int(input_ids.shape[-1])
            device = input_ids.device
        elif inputs_embeds is not None:
            batch_size = int(inputs_embeds.shape[0])
            current_length = int(inputs_embeds.shape[-2])
            device = inputs_embeds.device
        else:
            return original_forward(*args, **kwargs)

        if aligned is None:
            aligned = _align_prefix_key_values(model, frozen, device)

        temporal_steps = None
        logical_batch_size = batch_size
        if controller is not None and controller.mode.startswith("deploy_"):
            temporal_steps = int(controller.temporal_steps or 0)
            if temporal_steps <= 0 or batch_size % temporal_steps != 0:
                raise ValueError(
                    "Temporal Prefix batch is incompatible with deployment steps"
                )
            logical_batch_size = batch_size // temporal_steps
        kwargs["past_key_values"] = _fresh_dynamic_cache(
            aligned,
            logical_batch_size=logical_batch_size,
            temporal_steps=temporal_steps,
        )
        kwargs["attention_mask"] = _extend_attention_mask(
            kwargs.get("attention_mask"),
            batch_size=batch_size,
            current_length=current_length,
            cached_prefix_length=cached_prefix_length,
            device=device,
        )
        if kwargs.get("position_ids") is not None:
            kwargs["position_ids"] = kwargs["position_ids"] + cached_prefix_length
        if kwargs.get("cache_position") is not None:
            kwargs["cache_position"] = kwargs["cache_position"] + cached_prefix_length
        return original_forward(*args, **kwargs)

    model._snn2_prefix_original_forward = original_forward
    model._snn2_prefix_key_values = frozen
    model._snn2_prefix_length = cached_prefix_length
    model.forward = wrapped_forward
