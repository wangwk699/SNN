from __future__ import annotations

import collections
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .artifacts import write_json
from .data import CausalLMCollator, tokenize_dataset


class PrefixOutlierCollector:
    def __init__(self, eta: float):
        self.eta = float(eta)
        self.current_ids: torch.Tensor | None = None
        self.current_mask: torch.Tensor | None = None
        self.layer_counts: dict[str, list[int]] = collections.defaultdict(list)
        self.token_frequency: collections.Counter[int] = collections.Counter()
        self.handles = []

    def set_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        self.current_ids = input_ids.detach().cpu()
        self.current_mask = attention_mask.detach().cpu()

    def hook(self, name: str):
        def collect(_module, inputs):
            if self.current_ids is None or self.current_mask is None:
                raise RuntimeError("Prefix collector batch context was not set")
            activation = inputs[0].detach().float().cpu()
            token_max = activation.abs().amax(dim=-1)
            for batch_index in range(token_max.shape[0]):
                valid = self.current_mask[batch_index].bool()
                values = token_max[batch_index][valid]
                ids = self.current_ids[batch_index][valid]
                if values.numel() == 0:
                    self.layer_counts[name].append(0)
                    continue
                median = values.median().clamp_min(1e-12)
                outlier = values / median > self.eta
                self.layer_counts[name].append(int(outlier.sum().item()))
                positions = torch.nonzero(outlier, as_tuple=False).flatten()
                for position in positions.tolist():
                    if position == 0:  # PrefixQuant frequency excludes the initial token.
                        continue
                    self.token_frequency[int(ids[position].item())] += 1

        return collect

    def register(self, model: torch.nn.Module) -> None:
        for name, module in model.named_modules():
            if name.endswith("mlp.down_proj"):
                self.handles.append(module.register_forward_pre_hook(self.hook(name)))
        if not self.handles:
            raise RuntimeError("No down_proj modules found for PrefixQuant outlier detection")

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def result(self, tokenizer: Any) -> dict[str, Any]:
        average_by_layer = {
            name: sum(counts) / max(len(counts), 1) for name, counts in self.layer_counts.items()
        }
        outlier_count = int(math.ceil(max(average_by_layer.values(), default=0.0)))
        samples = max((len(counts) for counts in self.layer_counts.values()), default=0)
        if (
            len(self.token_frequency) == 1
            and self.token_frequency.most_common(1)[0][1] < 0.1 * samples
        ):
            outlier_count = max(outlier_count - 1, 0)
            self.token_frequency.clear()
        top = [token for token, _ in self.token_frequency.most_common(outlier_count)]
        bos = tokenizer.bos_token_id
        if bos is None:
            bos = tokenizer.eos_token_id
        if bos is not None and int(bos) not in top:
            top.append(int(bos))
        if not top:
            raise RuntimeError("PrefixQuant found no prefix token and tokenizer has no BOS/EOS")
        return {
            "format_version": 1,
            "eta": self.eta,
            "detection_activation": "down_proj_input",
            "outlier_token_count": outlier_count,
            "prefix_token_ids": top,
            "prefix_text": tokenizer.decode(top),
            "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "unknown")),
            "layer_average_outlier_counts": average_by_layer,
            "token_frequency": {str(key): value for key, value in self.token_frequency.items()},
            "corner_case_filter": "drop sole token if frequency < 10% of calibration samples",
        }


@torch.no_grad()
def discover_prefix_tokens(
    model: torch.nn.Module,
    tokenizer: Any,
    calibration_raw: Any,
    cfg: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    dataset = tokenize_dataset(calibration_raw, tokenizer, cfg, prefix_ids=None)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["calibration"].get("batch_size", 1)),
        shuffle=False,
        collate_fn=CausalLMCollator(tokenizer),
    )
    collector = PrefixOutlierCollector(float(cfg["prefix"]["outlier_threshold"]))
    collector.register(model)
    model.eval()
    device = next(model.parameters()).device
    try:
        for batch in loader:
            collector.set_batch(batch["input_ids"], batch["attention_mask"])
            model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                use_cache=False,
            )
    finally:
        collector.remove()
    state = collector.result(tokenizer)
    write_json(output_path, state)
    return state
