"""SparseLLM-style block-wise temporal calibration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import torch

from .artifacts import read_json
from torch.utils.data import DataLoader

from .calibration import materialize_target_state
from .config import gif_mse_refinement_enabled
from .gif_mse_integration import create_histogram_store, histogram_provenance, save_histogram_store
from .data import CausalLMCollator, tokenize_dataset
from .model_integration import prepare_temporal_model_inputs
from .prefix_cache import fresh_prefix_dynamic_cache, install_prefix_kv_forward
from .rotation import get_model_parts
from .sites import SITE_IDS, site_key
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


def _tree_map_tensors(value: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    """Apply ``fn`` recursively to tensors in decoder-forward kwargs."""
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(_tree_map_tensors(item, fn) for item in value)
    if isinstance(value, list):
        return [_tree_map_tensors(item, fn) for item in value]
    if isinstance(value, dict):
        return {key: _tree_map_tensors(item, fn) for key, item in value.items()}
    return value


def _cpu(value: Any) -> Any:
    return _tree_map_tensors(value, lambda tensor: tensor.detach().cpu())


def _to_device(value: Any, device: torch.device) -> Any:
    return _tree_map_tensors(value, lambda tensor: tensor.to(device))


def clear_target_trajectory_artifacts(site_root: str | Path, neuron: Neuron) -> None:
    """Remove only one neuron's stale sequential-calibration artifacts."""
    if neuron not in {"phase", "gif", "mtn"}:
        raise ValueError(f"Unsupported blockwise neuron: {neuron}")
    root = Path(site_root)
    for directory in root.glob("layer_*/site_*"):
        if directory.is_dir():
            for suffix in ("statistics", "state"):
                (directory / f"{neuron}_{suffix}.pt").unlink(missing_ok=True)
            if neuron == "gif":
                (directory / "gif_mse_histogram.pt").unlink(missing_ok=True)
    if neuron in {"phase", "mtn"}:
        global_directory = root / "_global" / "final_rmsnorm"
        for suffix in ("statistics", "state"):
            (global_directory / f"{neuron}_{suffix}.pt").unlink(missing_ok=True)


def _run(
    layer: torch.nn.Module, item: _LayerInput, *, model: torch.nn.Module,
    prefix_key_values: Any, temporal_steps: int,
) -> torch.Tensor:
    device = _device(layer)
    kwargs = _to_device(item.kwargs, device)
    kwargs["use_cache"] = False
    cache = fresh_prefix_dynamic_cache(
        model, prefix_key_values,
        logical_batch_size=item.hidden_states.shape[0] // temporal_steps,
        temporal_steps=temporal_steps, device=device,
    )
    if cache is not None:
        kwargs["past_key_value"] = cache
    output = layer(item.hidden_states.to(device), **kwargs)
    output = output[0] if isinstance(output, tuple) else output
    if not isinstance(output, torch.Tensor):
        raise TypeError("Decoder layer did not return hidden states")
    return output


@torch.no_grad()
def _bootstrap(
    model: torch.nn.Module, loader: DataLoader, controller: Any,
) -> list[_LayerInput]:
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
                repeated_ids, repeated_mask, model_kwargs = prepare_temporal_model_inputs(
                    batch["input_ids"], batch["attention_mask"],
                    steps=int(controller.temporal_steps),
                )
                model(
                    input_ids=repeated_ids.to(device), attention_mask=repeated_mask.to(device),
                    use_cache=False, **model_kwargs,
                )
            except _CatchFirstLayer:
                pass
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("Blockwise calibration failed to capture decoder inputs")
    return captured


def _save_target_statistics(
    store: StatisticsStore, root: Path, neuron: Neuron, *,
    layer_index: int | None = None, global_only: bool = False,
) -> None:
    """Validate exact collect coverage, then write this trajectory's statistics."""
    if global_only:
        if layer_index is not None or store.items or set(store.global_items) != {"final_rmsnorm"}:
            raise RuntimeError(
                "Blockwise final RMSNorm coverage mismatch: "
                f"items={sorted(store.items)}, globals={sorted(store.global_items)}"
            )
        path = root / "_global" / "final_rmsnorm"
        path.mkdir(parents=True, exist_ok=True)
        torch.save(store.global_items["final_rmsnorm"].state_dict(), path / f"{neuron}_statistics.pt")
        if not (path / f"{neuron}_statistics.pt").exists():
            raise FileNotFoundError(path / f"{neuron}_statistics.pt")
        return

    if layer_index is None:
        raise ValueError("layer_index is required for non-global blockwise statistics")
    expected = {site_key(layer_index, site_index) for site_index in SITE_IDS}
    actual = set(store.items)
    if actual != expected or store.global_items:
        raise RuntimeError(
            "Blockwise calibration site coverage mismatch: "
            f"neuron={neuron} layer={layer_index} "
            f"missing={sorted(expected - actual)} unexpected={sorted(actual - expected)} "
            f"global={sorted(store.global_items)}"
        )
    for key in sorted(expected):
        path = root / key
        path.mkdir(parents=True, exist_ok=True)
        target = path / f"{neuron}_statistics.pt"
        torch.save(store.items[key].state_dict(), target)
        if not target.exists():
            raise FileNotFoundError(target)


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
    clear_target_trajectory_artifacts(root, neuron)
    loader = DataLoader(tokenize_dataset(calibration_raw, tokenizer, cfg, prefix_ids=None),
                        batch_size=1, shuffle=False, collate_fn=CausalLMCollator(tokenizer))
    install_prefix_kv_forward(model, prefix_key_values, controller=controller)
    model.eval()
    controller.clear_runtime_module_cache()
    controller.begin_sequential_calibration(neuron, 0)
    cached, parts = _bootstrap(model, loader, controller), get_model_parts(model)
    for index, layer in enumerate(parts.layers):
        controller.begin_sequential_calibration(neuron, index)
        controller.statistics = StatisticsStore()
        for item in cached:
            _run(layer, item, model=model, prefix_key_values=prefix_key_values,
                 temporal_steps=int(controller.temporal_steps))
        _save_target_statistics(controller.statistics, root, neuron, layer_index=index)
        if neuron == "gif" and gif_mse_refinement_enabled(cfg):
            histogram_store = create_histogram_store(
                root, cfg, statistics_name="gif_statistics.pt", layer_index=index
            )
            controller.gif_mse_collector = histogram_store
            controller.begin_gif_mse_collection(block_index=index)
            try:
                for item in cached:
                    _run(
                        layer, item, model=model, prefix_key_values=prefix_key_values,
                        temporal_steps=int(controller.temporal_steps),
                    )
            finally:
                controller.gif_mse_collector = None
            manifest = read_json(root / "statistics_manifest.json")
            save_histogram_store(
                histogram_store, root,
                histogram_provenance(
                    cfg, manifest, trajectory_source="sequential_temporal_gif"
                ),
                statistics_name="gif_statistics.pt",
            )
        materialize_target_state(root, cfg, neuron, layer_index=index)
        controller.clear_layer_module_cache(index)
        controller.begin_sequential_deployment(index)
        cached = [
            _LayerInput(
                _run(layer, item, model=model, prefix_key_values=prefix_key_values,
                     temporal_steps=int(controller.temporal_steps)).detach().cpu(),
                item.kwargs,
            ) for item in cached
        ]
        controller.clear_layer_module_cache(index)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if neuron in {"phase", "mtn"}:
        controller.begin_sequential_calibration(neuron, len(parts.layers))
        controller.statistics = StatisticsStore()
        for item in cached:
            parts.final_norm(item.hidden_states.to(_device(parts.final_norm)))
        _save_target_statistics(controller.statistics, root, neuron, global_only=True)
        materialize_target_state(root, cfg, neuron, global_only=True)
    controller.end_sequential_calibration()
    return {"neuron": neuron, "blocks": len(parts.layers), "samples": len(cached)}
