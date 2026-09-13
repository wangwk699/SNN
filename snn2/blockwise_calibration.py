"""SparseLLM-style block-wise temporal calibration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader

from .calibration import materialize_target_state
from .data import CausalLMCollator, tokenize_dataset
from .prefix_cache import fresh_prefix_dynamic_cache, install_prefix_kv_forward
from .rotation import get_model_parts
from .stats import StatisticsStore

Neuron = Literal["phase", "gif", "mtn"]


class _CatchFirstLayer(RuntimeError):
    pass


@dataclass
class _LayerInput:
    hidden_states: torch.Tensor
    kwargs: dict[str, Any]


def _device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _cpu(value: Any) -> Any:
    return value.detach().cpu() if isinstance(value, torch.Tensor) else value


def _run(layer: torch.nn.Module, item: _LayerInput, *, model: torch.nn.Module, prefix_key_values: Any, temporal_steps: int) -> torch.Tensor:
    device = _device(layer)
    kwargs = {name: value.to(device) if isinstance(value, torch.Tensor) else value
              for name, value in item.kwargs.items()}
    kwargs["use_cache"] = False
    cache = fresh_prefix_dynamic_cache(model, prefix_key_values, logical_batch_size=item.hidden_states.shape[0] // temporal_steps, temporal_steps=temporal_steps, device=device)
    if cache is not None:
        kwargs["past_key_values"] = cache
    output = layer(item.hidden_states.to(device), **kwargs)
    output = output[0] if isinstance(output, tuple) else output
    if not isinstance(output, torch.Tensor):
        raise TypeError("Decoder layer did not return hidden states")
    return output


@torch.no_grad()
def _bootstrap(model: torch.nn.Module, loader: DataLoader) -> list[_LayerInput]:
    parts, captured = get_model_parts(model), []

    def catcher(_module, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("Unable to capture first decoder hidden states")
        captured.append(_LayerInput(
            hidden.detach().cpu(),
            {name: _cpu(value) for name, value in kwargs.items()
             if name not in {"hidden_states", "past_key_value", "past_key_values"}},
        ))
        raise _CatchFirstLayer()

    handle = parts.layers[0].register_forward_pre_hook(catcher, with_kwargs=True)
    device = _device(model)
    try:
        for batch in loader:
            try:
                model(input_ids=batch["input_ids"].to(device),
                      attention_mask=batch["attention_mask"].to(device), use_cache=False)
            except _CatchFirstLayer:
                pass
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("Blockwise calibration failed to capture decoder inputs")
    return captured


def _save_target_statistics(store: StatisticsStore, root: Path, neuron: Neuron) -> None:
    for key, stats in store.items.items():
        path = root / key
        path.mkdir(parents=True, exist_ok=True)
        torch.save(stats.state_dict(), path / f"{neuron}_statistics.pt")
    for name, stats in store.global_items.items():
        path = root / "_global" / name
        path.mkdir(parents=True, exist_ok=True)
        torch.save(stats.state_dict(), path / f"{neuron}_statistics.pt")


@torch.no_grad()
def collect_blockwise_snn_conditioned_statistics(
    model: torch.nn.Module, controller: Any, tokenizer: Any, calibration_raw: Any,
    cfg: dict[str, Any], prefix_key_values: Any, site_root: str | Path, *, neuron: Neuron,
) -> dict[str, Any]:
    """Collect one independent target trajectory with O(L) block propagation."""
    if neuron not in {"phase", "gif", "mtn"}:
        raise ValueError(f"Unsupported blockwise neuron: {neuron}")
    if int(cfg["calibration"].get("batch_size", 1)) != 1:
        raise ValueError("Blockwise EMA calibration requires batch_size=1")
    root = Path(site_root)
    loader = DataLoader(tokenize_dataset(calibration_raw, tokenizer, cfg, prefix_ids=None),
                        batch_size=1, shuffle=False, collate_fn=CausalLMCollator(tokenizer))
    install_prefix_kv_forward(model, prefix_key_values, controller=controller)
    model.eval()
    controller._modules.clear()
    controller.begin_sequential_calibration(neuron, 0)
    cached, parts = _bootstrap(model, loader), get_model_parts(model)
    for index, layer in enumerate(parts.layers):
        controller.begin_sequential_calibration(neuron, index)
        controller.statistics = StatisticsStore()
        for item in cached:
            _run(layer, item, model=model, prefix_key_values=prefix_key_values, temporal_steps=int(controller.temporal_steps))
        _save_target_statistics(controller.statistics, root, neuron)
        materialize_target_state(root, cfg, neuron, layer_index=index)
        controller.begin_sequential_deployment(index)
        cached = [_LayerInput(_run(layer, item, model=model, prefix_key_values=prefix_key_values, temporal_steps=int(controller.temporal_steps)).detach().cpu(), item.kwargs) for item in cached]
        controller._modules.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if neuron in {"phase", "mtn"}:
        controller.begin_sequential_calibration(neuron, len(parts.layers))
        controller.statistics = StatisticsStore()
        for item in cached:
            parts.final_norm(item.hidden_states.to(_device(parts.final_norm)))
        _save_target_statistics(controller.statistics, root, neuron)
        materialize_target_state(root, cfg, neuron, global_only=True)
    controller.end_sequential_calibration()
    return {"neuron": neuron, "blocks": len(parts.layers), "samples": len(cached)}
