#!/usr/bin/env python3
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "snn2").is_dir():
    raise SystemExit("Run this script from the SNN repository root.")


def replace(path: str, old: str, new: str) -> None:
    p = ROOT / path
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"Expected block not found in {path}:\n{old[:300]}")
    p.write_text(text.replace(old, new), encoding="utf-8")


# ---------------------------------------------------------------------
# 1) New helper: fixed PrefixQuant-style KV cache
# ---------------------------------------------------------------------
(ROOT / "snn2" / "prefix_cache.py").write_text('''from __future__ import annotations

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


def _fresh_dynamic_cache(prefix_key_values, batch_size: int):
    from transformers.cache_utils import DynamicCache

    repeated = tuple(
        (
            key.repeat_interleave(batch_size, dim=0),
            value.repeat_interleave(batch_size, dim=0),
        )
        for key, value in prefix_key_values
    )
    return DynamicCache.from_legacy_cache(repeated)


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


def install_prefix_kv_forward(model: torch.nn.Module, prefix_key_values) -> None:
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

        kwargs["past_key_values"] = _fresh_dynamic_cache(aligned, batch_size)
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
''', encoding="utf-8")

# ---------------------------------------------------------------------
# 2) Prefix discovery behavior
# ---------------------------------------------------------------------
replace(
    "snn2/prefix.py",
    '''class PrefixOutlierCollector:\n    def __init__(self, eta: float):\n        self.eta = float(eta)\n''',
    '''class PrefixOutlierCollector:\n    def __init__(self, eta: float, skip_initial_position: bool = True):\n        self.eta = float(eta)\n        self.skip_initial_position = bool(skip_initial_position)\n''',
)
replace(
    "snn2/prefix.py",
    '''                for position in positions.tolist():\n                    if position == 0:  # PrefixQuant frequency excludes the initial token.\n                        continue\n                    self.token_frequency[int(ids[position].item())] += 1\n''',
    '''                for position in positions.tolist():\n                    if self.skip_initial_position and position == 0:\n                        continue\n                    self.token_frequency[int(ids[position].item())] += 1\n''',
)
replace(
    "snn2/prefix.py",
    '''    def result(self, tokenizer: Any) -> dict[str, Any]:\n''',
    '''    def result(\n        self,\n        tokenizer: Any,\n        *,\n        append_start_token: bool = True,\n    ) -> dict[str, Any]:\n''',
)
replace(
    "snn2/prefix.py",
    '''        top = [token for token, _ in self.token_frequency.most_common(outlier_count)]\n        bos = tokenizer.bos_token_id\n        if bos is None:\n            bos = tokenizer.eos_token_id\n        if bos is not None and int(bos) not in top:\n            top.append(int(bos))\n        if not top:\n            raise RuntimeError("PrefixQuant found no prefix token and tokenizer has no BOS/EOS")\n        return {\n''',
    '''        top = [token for token, _ in self.token_frequency.most_common(outlier_count)]\n        appended_start_token_id = None\n        if append_start_token:\n            bos = tokenizer.bos_token_id\n            if bos is not None and int(bos) not in top:\n                appended_start_token_id = int(bos)\n                top.append(appended_start_token_id)\n        return {\n''',
)
replace(
    "snn2/prefix.py",
    '''            "corner_case_filter": "drop sole token if frequency < 10% of calibration samples",\n''',
    '''            "corner_case_filter": "drop sole token if frequency < 10% of calibration samples",\n            "skip_initial_position_in_frequency": self.skip_initial_position,\n            "appended_start_token_id": appended_start_token_id,\n''',
)
replace(
    "snn2/prefix.py",
    '''    collector = PrefixOutlierCollector(float(cfg["prefix"]["outlier_threshold"]))\n''',
    '''    is_qwen = "qwen" in str(cfg["experiment"]["model_name"]).lower()\n    collector = PrefixOutlierCollector(\n        float(cfg["prefix"]["outlier_threshold"]),\n        skip_initial_position=not is_qwen,\n    )\n''',
)
replace(
    "snn2/prefix.py",
    '''    state = collector.result(tokenizer)\n''',
    '''    state = collector.result(\n        tokenizer,\n        append_start_token=not is_qwen,\n    )\n''',
)

# ---------------------------------------------------------------------
# 2.5) Prevent accidental future token-level Prefix prepending
# ---------------------------------------------------------------------
replace(
    "snn2/data.py",
    '''    prefix_ids = list(prefix_ids or [])\n    if prefix_ids:\n        input_ids = prefix_ids + input_ids\n        labels = [-100] * len(prefix_ids) + labels\n''',
    '''    prefix_ids = list(prefix_ids or [])\n    if prefix_ids:\n        raise ValueError(\n            "Prefix token IDs must not be prepended to input_ids. "\n            "Use the fixed Prefix past_key_values cache instead."\n        )\n''',
)

# ---------------------------------------------------------------------
# 3) Discover and save Prefix KV
# ---------------------------------------------------------------------
replace(
    "scripts/discover_prefix.py",
    '''from snn2.prefix import discover_prefix_tokens\n''',
    '''from snn2.prefix import discover_prefix_tokens\nfrom snn2.prefix_cache import build_prefix_key_values, save_prefix_key_values\n''',
)
replace(
    "scripts/discover_prefix.py",
    '''        state = discover_prefix_tokens(model, tokenizer, bundle.calibration, cfg, output)\n        run.event("prefix_saved", count=len(state["prefix_token_ids"]), ids=state["prefix_token_ids"])\n''',
    '''        state = discover_prefix_tokens(model, tokenizer, bundle.calibration, cfg, output)\n        prefix_key_values = build_prefix_key_values(model, state["prefix_token_ids"])\n        cache_path = layout.prefix_dir / "prefixed_key_values.pt"\n        save_prefix_key_values(cache_path, prefix_key_values)\n        run.event(\n            "prefix_saved",\n            count=len(state["prefix_token_ids"]),\n            ids=state["prefix_token_ids"],\n            kv_cache_saved=prefix_key_values is not None,\n            kv_cache_path=str(cache_path) if prefix_key_values is not None else None,\n        )\n''',
)

# ---------------------------------------------------------------------
# 4) model helper
# ---------------------------------------------------------------------
replace(
    "snn2/modeling.py",
    '''from .rotation import load_rotation_state\n''',
    '''from .rotation import load_rotation_state\nfrom .prefix_cache import load_prefix_key_values\n''',
)
block = '''def prefix_ids(cfg: dict[str, Any], layout: ArtifactLayout) -> list[int]:\n    if not bool(cfg["prefix"]["enabled"]):\n        return []\n    state = read_json(layout.prefix_dir / "prefix_state.json")\n    return [int(value) for value in state["prefix_token_ids"]]\n'''
replace(
    "snn2/modeling.py",
    block,
    block + '''\n\ndef prefix_key_values(cfg: dict[str, Any], layout: ArtifactLayout):\n    ids = prefix_ids(cfg, layout)\n    if not ids:\n        return None\n    path = layout.prefix_dir / "prefixed_key_values.pt"\n    if not path.exists():\n        raise FileNotFoundError(\n            f"Prefix is enabled but fixed KV cache is missing: {path}. "\n            "Re-run scripts/discover_prefix.py."\n        )\n    return load_prefix_key_values(path)\n''',
)

# ---------------------------------------------------------------------
# 5) Calibration: no prepend, fixed past_key_values
# ---------------------------------------------------------------------
replace(
    "snn2/calibration.py",
    '''from .stats import StatisticsStore\n''',
    '''from .stats import StatisticsStore\nfrom .prefix_cache import install_prefix_kv_forward\n''',
)
replace(
    "snn2/calibration.py",
    '''    prefix_ids: list[int],\n    site_root: str | Path,\n) -> dict[str, Any]:\n    dataset = tokenize_dataset(calibration_raw, tokenizer, cfg, prefix_ids=prefix_ids)\n''',
    '''    prefix_key_values: Any,\n    site_root: str | Path,\n) -> dict[str, Any]:\n    dataset = tokenize_dataset(calibration_raw, tokenizer, cfg, prefix_ids=None)\n''',
)
replace(
    "snn2/calibration.py",
    '''    controller.statistics = StatisticsStore(\n        max_channels_by_site={5: int(cfg["data"]["max_seq_length"])}\n    )\n    model.eval()\n''',
    '''    controller.statistics = StatisticsStore(\n        max_channels_by_site={5: int(cfg["data"]["max_seq_length"])}\n    )\n    install_prefix_kv_forward(model, prefix_key_values)\n    model.eval()\n''',
)
replace(
    "scripts/calibrate_sites.py",
    '''    prefix_ids,\n    rotation_state,\n)\n''',
    '''    prefix_key_values,\n    rotation_state,\n)\n''',
)
replace(
    "scripts/calibrate_sites.py",
    '''            prefix_ids(cfg, layout),\n            layout.site_dir,\n''',
    '''            prefix_key_values(cfg, layout),\n            layout.site_dir,\n''',
)

# ---------------------------------------------------------------------
# 6) Training: no prepend, fixed past_key_values
# ---------------------------------------------------------------------
replace(
    "snn2/training.py",
    '''from .modeling import load_model, load_tokenizer, model_source, prefix_ids, rotation_state\n''',
    '''from .modeling import load_model, load_tokenizer, model_source, prefix_ids, prefix_key_values, rotation_state\nfrom .prefix_cache import install_prefix_kv_forward\n''',
)
replace(
    "snn2/training.py",
    '''    bundle = load_selected_raw(cfg, layout)\n    prefixes = prefix_ids(cfg, layout)\n    with arguments.main_process_first(desc="tokenize train and validation datasets"):\n        train_dataset = tokenize_dataset(bundle.train, tokenizer, cfg, prefixes)\n        validation_dataset = tokenize_dataset(bundle.validation, tokenizer, cfg, prefixes)\n''',
    '''    bundle = load_selected_raw(cfg, layout)\n    prefixes = prefix_ids(cfg, layout)\n    install_prefix_kv_forward(model, prefix_key_values(cfg, layout))\n    with arguments.main_process_first(desc="tokenize train and validation datasets"):\n        train_dataset = tokenize_dataset(bundle.train, tokenizer, cfg, prefix_ids=None)\n        validation_dataset = tokenize_dataset(bundle.validation, tokenizer, cfg, prefix_ids=None)\n''',
)
replace(
    "snn2/training.py",
    '''                "prefix_loss_masked": True,\n''',
    '''                "prefix_mode": "fixed_past_key_values",\n                "prefix_token_ids": prefixes,\n                "prefix_loss_masked": "not_applicable_prefix_not_in_labels",\n''',
)

# ---------------------------------------------------------------------
# 7) lm-eval proxy: no explicit token prepend
# ---------------------------------------------------------------------
replace(
    "snn2/evaluation.py",
    '''from .model_integration import temporal_forward\n''',
    '''from .model_integration import temporal_forward\nfrom .prefix_cache import install_prefix_kv_forward\n''',
)
replace(
    "snn2/evaluation.py",
    '''        prefix_ids: list[int],\n''',
    '''        prefix_key_values,\n''',
)
replace(
    "snn2/evaluation.py",
    '''        self.prefix_ids = list(prefix_ids)\n\n        self.config = model.config\n''',
    '''        install_prefix_kv_forward(self.model, prefix_key_values)\n\n        self.config = model.config\n''',
)
p = ROOT / "snn2/evaluation.py"
text = p.read_text(encoding="utf-8")
start = text.index('    def _prefix(\n')
end = text.index('    def forward(\n', start)
text = text[:start] + '''    def _prefix(\n        self,\n        input_ids: torch.Tensor,\n        attention_mask: torch.Tensor | None,\n    ):\n        if attention_mask is None:\n            attention_mask = torch.ones_like(input_ids)\n        return input_ids, attention_mask, 0\n\n''' + text[end:]
p.write_text(text, encoding="utf-8")

replace(
    "scripts/evaluate_lm_harness.py",
    '''    prefix_ids,\n    rotation_state,\n)\n''',
    '''    prefix_ids,\n    prefix_key_values,\n    rotation_state,\n)\n''',
)
replace(
    "scripts/evaluate_lm_harness.py",
    '''            model_prefix_ids,\n        )\n''',
    '''            prefix_key_values(cfg, layout),\n        )\n''',
)

# ---------------------------------------------------------------------
# 8) TL;DR: no prepend; fixed Prefix KV
# ---------------------------------------------------------------------
replace(
    "scripts/evaluate_tldr.py",
    '''from snn2.modeling import load_model, load_tokenizer, model_source, prefix_ids, rotation_state\n''',
    '''from snn2.modeling import (\n    load_model,\n    load_tokenizer,\n    model_source,\n    prefix_ids,\n    prefix_key_values,\n    rotation_state,\n)\nfrom snn2.prefix_cache import install_prefix_kv_forward\n''',
)
replace(
    "scripts/evaluate_tldr.py",
    '''        prefixes = prefix_ids(cfg, layout)\n\n        evaluation = load_selected_raw(cfg, layout).evaluation\n''',
    '''        prefixes = prefix_ids(cfg, layout)\n        install_prefix_kv_forward(model, prefix_key_values(cfg, layout))\n\n        evaluation = load_selected_raw(cfg, layout).evaluation\n''',
)
replace(
    "scripts/evaluate_tldr.py",
    '''                    max_length=max(input_length - len(prefixes), 1),\n                )\n\n                input_ids = prefixes + prompt_ids\n\n                batch_input_ids.append(input_ids)\n''',
    '''                    max_length=input_length,\n                )\n\n                batch_input_ids.append(prompt_ids)\n''',
)

# ---------------------------------------------------------------------
# 9) SNN integration: cached Prefix positions bypass sites 3/4/5 stats/replacement
# ---------------------------------------------------------------------
replace(
    "snn2/model_integration.py",
    '''    r3: HadamardSpec | None = getattr(module, "_snn2_r3", None)\n    if r3 is not None:\n        query = random_hadamard(query, r3)\n        key = random_hadamard(key, r3)\n    query = controller.apply(layer_index, 2, query)\n    key = controller.apply(layer_index, 3, key)\n    value = controller.apply(layer_index, 4, value)\n\n    groups = int(getattr(module, "num_key_value_groups", 1))\n''',
    '''    current_length = int(query.shape[-2])\n    past_length = max(int(key.shape[-2]) - current_length, 0)\n\n    r3: HadamardSpec | None = getattr(module, "_snn2_r3", None)\n    if r3 is not None:\n        query = random_hadamard(query, r3)\n        key = random_hadamard(key, r3)\n    query = controller.apply(layer_index, 2, query)\n\n    if past_length:\n        prefix_key, current_key = key[..., :past_length, :], key[..., past_length:, :]\n        prefix_value, current_value = value[..., :past_length, :], value[..., past_length:, :]\n        current_key = controller.apply(layer_index, 3, current_key)\n        current_value = controller.apply(layer_index, 4, current_value)\n        key = torch.cat((prefix_key, current_key), dim=-2)\n        value = torch.cat((prefix_value, current_value), dim=-2)\n    else:\n        key = controller.apply(layer_index, 3, key)\n        value = controller.apply(layer_index, 4, value)\n\n    groups = int(getattr(module, "num_key_value_groups", 1))\n''',
)
replace(
    "snn2/model_integration.py",
    '''    if controller.mode == "collect":\n        controller.record_saliency(layer_index, 2, query * torch.matmul(qk, key))\n        controller.record_saliency(\n            layer_index, 3, key * torch.matmul(qk.transpose(2, 3), query)\n        )\n''',
    '''    if controller.mode == "collect":\n        controller.record_saliency(layer_index, 2, query * torch.matmul(qk, key))\n        key_score = key * torch.matmul(qk.transpose(2, 3), query)\n        if past_length:\n            key_score = key_score[..., past_length:, :]\n        controller.record_saliency(layer_index, 3, key_score)\n''',
)
replace(
    "snn2/model_integration.py",
    '''    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)\n    weights = controller.apply(layer_index, 5, weights)\n''',
    '''    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)\n    if past_length:\n        prefix_weights = weights[..., :past_length]\n        current_weights = controller.apply(layer_index, 5, weights[..., past_length:])\n        weights = torch.cat((prefix_weights, current_weights), dim=-1)\n    else:\n        weights = controller.apply(layer_index, 5, weights)\n''',
)
replace(
    "snn2/model_integration.py",
    '''        controller.record_saliency(\n            layer_index,\n            4,\n            value * torch.matmul(weights.transpose(2, 3), output),\n        )\n''',
    '''        value_score = value * torch.matmul(weights.transpose(2, 3), output)\n        if past_length:\n            value_score = value_score[..., past_length:, :]\n        controller.record_saliency(layer_index, 4, value_score)\n''',
)
replace(
    "snn2/model_integration.py",
    '''        controller.record_saliency_reduced(\n            layer_index,\n            5,\n            position_score,\n            weights.shape[0] * weights.shape[1] * weights.shape[2],\n        )\n''',
    '''        if past_length:\n            position_score = position_score[past_length:]\n        controller.record_saliency_reduced(\n            layer_index,\n            5,\n            position_score,\n            weights.shape[0] * weights.shape[1] * weights.shape[2],\n        )\n''',
)

# ---------------------------------------------------------------------
# 10) Tests
# ---------------------------------------------------------------------
(ROOT / "tests" / "test_prefix.py").write_text('''import torch\n\nfrom snn2.prefix import PrefixOutlierCollector\n\n\nclass _Tokenizer:\n    bos_token_id = 128000\n    eos_token_id = 128001\n    name_or_path = "dummy"\n\n    def decode(self, ids):\n        return " ".join(map(str, ids))\n\n\ndef _collect(skip_initial_position: bool, activation: torch.Tensor):\n    collector = PrefixOutlierCollector(\n        64.0,\n        skip_initial_position=skip_initial_position,\n    )\n    collector.set_batch(\n        torch.tensor([[11, 12, 13]], dtype=torch.long),\n        torch.ones((1, 3), dtype=torch.long),\n    )\n    collector.hook("model.layers.0.mlp.down_proj")(None, (activation,))\n    return collector\n\n\ndef test_qwen_counts_position_zero_and_does_not_append_start_token():\n    activation = torch.tensor([[[100.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])\n    collector = _collect(False, activation)\n    state = collector.result(_Tokenizer(), append_start_token=False)\n    assert state["prefix_token_ids"] == [11]\n    assert state["skip_initial_position_in_frequency"] is False\n    assert state["appended_start_token_id"] is None\n\n\ndef test_llama_skips_position_zero_and_appends_bos():\n    activation = torch.tensor([[[100.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])\n    collector = _collect(True, activation)\n    state = collector.result(_Tokenizer(), append_start_token=True)\n    assert state["prefix_token_ids"] == [128000]\n    assert state["skip_initial_position_in_frequency"] is True\n    assert state["appended_start_token_id"] == 128000\n\n\ndef test_qwen_can_have_empty_prefix():\n    activation = torch.ones((1, 3, 2))\n    collector = _collect(False, activation)\n    state = collector.result(_Tokenizer(), append_start_token=False)\n    assert state["prefix_token_ids"] == []\n''', encoding="utf-8")

print("Prefix KV-cache fix applied.")
print("Run: python -m py_compile snn2/*.py scripts/*.py")
print("Run: pytest -q tests/test_prefix.py")
