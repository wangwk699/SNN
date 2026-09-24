"""Process-local theoretical operation accounting for fixed length inference."""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from snn2.artifacts import sha256_file, write_json
from snn2.config import (
    final_ann_evaluation_prefix_artifact_stage,
    final_snn_evaluation_prefix_artifact_stage,
)
from snn2.data import load_selected_raw, tokenize_row
from snn2.evaluation import build_evaluation_controller, position_ids_from_attention_mask
from snn2.modeling import load_model, load_tokenizer, model_source_for_stage, prefix_key_values_for_stage, rotation_state
from snn2.model_integration import install_model_integration, temporal_forward
from snn2.neurons import (
    AllLowStaticGIF, IdentityGIF,
    SoftmaxIdentityGIF, StaticGIF, _mask_values, GIF_LOW_QMAX,
)
from snn2.prefix_cache import install_prefix_kv_forward, prefix_length
from snn2.training import validate_recorded_training_artifact_provenance

ENERGY_PROFILER_VERSION = 3
ENERGY_ACCOUNTING_POLICY = "sat_llm_mac_ac_v3_final_deployment_protocol"
SEQUENCE_LENGTH = 512
COLUMNS = ("Model", "Neuron", "T", "MACs (G)", "Synaptic ACs (G)", "Neuron ACs (G)", "Total ACs (G)", "Energy (J)")


@dataclass
class EnergyCounts:
    mac_raw: int = 0
    synaptic_ac_raw: int = 0
    neuron_ac_raw: int = 0

    @property
    def total_ac_raw(self) -> int:
        return self.synaptic_ac_raw + self.neuron_ac_raw

    def add(self, other: "EnergyCounts") -> None:
        self.mac_raw += other.mac_raw
        self.synaptic_ac_raw += other.synaptic_ac_raw
        self.neuron_ac_raw += other.neuron_ac_raw


def raw_to_g(value: int | float) -> float:
    return value / 1_000_000_000


def energy_j_from_raw_counts(mac_raw: int | float, ac_raw: int | float) -> float:
    return 4.6e-12 * mac_raw + 0.9e-12 * ac_raw


def select_validation_positions(length: int, seed: int, num_samples: int) -> list[int]:
    if num_samples < 1 or num_samples > length:
        raise ValueError(f"Profiling sample count {num_samples} exceeds validation length {length} or is nonpositive")
    return sorted(random.Random(seed).sample(range(length), k=num_samples))


def fixed_length_input(row: dict, tokenizer: Any, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    # tokenize_row uses complete TL;DR or conversation semantics and rejects prefix IDs.
    local_cfg = {**cfg, "data": {**cfg["data"], "max_seq_length": SEQUENCE_LENGTH}}
    ids = tokenize_row(row, tokenizer, local_cfg)["input_ids"]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer needs a pad or EOS token for fixed length profiling")
    mask = [1] * len(ids) + [0] * (SEQUENCE_LENGTH - len(ids))
    ids = ids + [int(pad_id)] * (SEQUENCE_LENGTH - len(ids))
    return torch.tensor([ids], dtype=torch.long), torch.tensor([mask], dtype=torch.long)


def gif_integer_multiplicity(module: nn.Module, incoming: torch.Tensor, role: str | None = None) -> torch.Tensor | None:
    if isinstance(module, (IdentityGIF, SoftmaxIdentityGIF)):
        return None
    x = incoming.sum(dim=0)
    if isinstance(module, AllLowStaticGIF):
        _, low, _ = module._quantize(x, module.low_scale, module.low_zero, qmin=0, qmax=GIF_LOW_QMAX)
        return torch.stack((low.to(torch.int32), torch.zeros_like(low, dtype=torch.int32)))
    if not isinstance(module, StaticGIF):
        raise TypeError(type(module))
    _, low, _ = module._quantize(x, module.low_scale, module.low_zero, qmin=0, qmax=GIF_LOW_QMAX)
    _, high, _ = module._quantize(x, module.high_scale, module.high_zero, qmin=0, qmax=module.high_qmax)
    mask = _mask_values(x, module._mask(role), module.layout)
    chunk0, chunk1 = module.integer_chunks(high)
    return torch.stack((torch.where(mask, low, chunk0), torch.where(mask, torch.zeros_like(low), chunk1))).to(torch.int32)


def linear_counts(x: torch.Tensor, out_features: int, multiplicity: torch.Tensor | None = None) -> EnergyCounts:
    if multiplicity is None:
        return EnergyCounts(mac_raw=x.numel() * out_features)
    return EnergyCounts(synaptic_ac_raw=int(multiplicity.sum().item()) * out_features)


def _pair(a: torch.Tensor, b: torch.Tensor) -> int:
    """Count event pairs by reducing the shared feature dimension."""
    if a.shape[:-2] != b.shape[:-2] or a.shape[-1] != b.shape[-2]:
        raise ValueError(f"Incompatible pair shapes: {tuple(a.shape)} and {tuple(b.shape)}")
    left = a.to(torch.int64).sum(dim=-2)
    right = b.to(torch.int64).sum(dim=-1)
    value = int((left * right).sum().item())
    if value < 0:
        raise OverflowError("Event pair count overflowed int64")
    return value


def temporal_product_counts(a: torch.Tensor, b: torch.Tensor, ma: torch.Tensor | None, mb: torch.Tensor | None) -> EnergyCounts:
    if a.shape != b.shape or (ma is not None and ma.shape != a.shape) or (mb is not None and mb.shape != b.shape):
        raise ValueError("Temporal product operands and multiplicities must have the same shape")
    if ma is None and mb is None:
        return EnergyCounts(mac_raw=2 * a.numel())
    if ma is not None and mb is not None:
        left = ma.to(torch.int64)
        right = mb.to(torch.int64)
        count = int((left * right.sum(dim=0, keepdim=True) + left.sum(dim=0, keepdim=True) * right).sum().item())
        return EnergyCounts(synaptic_ac_raw=count)
    # A dense dynamic value, including its temporal sum, is one operand.
    events = mb if ma is None else ma
    return EnergyCounts(mac_raw=(int(a.shape[0]) + 1) * int(events.sum().item()))


class EnergyInstrumentation:
    """Only observes forward values; restores controller methods and module references."""

    def __init__(self, model: nn.Module, controller: Any, neuron: str, *, prefix_tokens: int = 0):
        self.model, self.controller, self.neuron = model, controller, neuron
        self.prefix_tokens = prefix_tokens
        self.counts = EnergyCounts()
        self._handles: list[Any] = []
        self._originals: list[tuple[Any, str, Any]] = []
        self.events: dict[tuple[int, int, str | None], torch.Tensor | None] = {}
        self._matmul_call = 0
        self._hadamard_layer = 0

    def reset(self) -> None:
        self.counts = EnergyCounts()
        self.events.clear()
        self._matmul_call = 0
        self._hadamard_layer = 0

    def _patch(self, obj: Any, name: str, replacement: Any) -> None:
        self._originals.append((obj, name, getattr(obj, name)))
        setattr(obj, name, replacement)

    def _event(self, layer: int, site: int, role: str | None = None) -> torch.Tensor | None:
        return self.events.get((layer, site, role))

    def __enter__(self):
        import snn2.temporal_model as temporal_model
        import snn2.model_integration as integration
        original_apply = self.controller.apply
        original_final = self.controller.apply_final_norm_neuron

        def apply(layer, site, x, *, gif_role=None):
            out = original_apply(layer, site, x, gif_role=gif_role)
            if self.neuron != "ann":
                steps = int(self.controller.temporal_steps)
                incoming = x.reshape(steps, x.shape[0] // steps, *x.shape[1:])
                temporal_out = out.reshape(steps, out.shape[0] // steps, *out.shape[1:])
                if self.neuron == "gif":
                    module = self.controller._load(layer, site)["gif"]
                    mult = gif_integer_multiplicity(module, incoming, gif_role)
                else:
                    mult = (temporal_out != 0).to(torch.int32)
                    self.counts.neuron_ac_raw += (incoming[0].numel() if self.neuron == "phase" else incoming.numel()) + int(mult.sum().item())
                self.events[(layer, site, gif_role)] = mult
            return out

        def final(x):
            out = original_final(x)
            if self.neuron in {"phase", "mtn"}:
                steps = int(self.controller.temporal_steps)
                mult = (out.reshape(steps, out.shape[0] // steps, *out.shape[1:]) != 0).to(torch.int32)
                self.counts.neuron_ac_raw += (x.numel() // steps if self.neuron == "phase" else x.numel()) + int(mult.sum().item())
                self.events[(-1, 0, None)] = mult
            return out

        self._patch(self.controller, "apply", apply)
        self._patch(self.controller, "apply_final_norm_neuron", final)

        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            def hook(mod, inputs, output, path=name):
                mult = None
                if self.neuron != "ann":
                    parts = path.split(".")
                    layer = next((int(parts[i + 1]) for i, p in enumerate(parts[:-1]) if p == "layers" and parts[i + 1].isdigit()), None)
                    leaf = parts[-1]
                    site_role = {"q_proj": (1, "q"), "k_proj": (1, "k"), "v_proj": (1, "v"), "o_proj": (6, None), "gate_proj": (7, "gate"), "up_proj": (7, "up"), "down_proj": (10, None)}.get(leaf)
                    if layer is not None and site_role is not None:
                        site, role = site_role
                        mult = self._event(layer, site, role)
                        if mult is None:
                            mult = self._event(layer, site, None)
                    elif leaf == "lm_head":
                        mult = self._event(-1, 0)
                self.counts.add(linear_counts(inputs[0], mod.out_features, mult))
                if self.neuron != "ann":
                    if layer is not None and site_role is not None:
                        if leaf in {"v_proj", "up_proj", "o_proj", "down_proj"}:
                            site = site_role[0]
                            for key in [key for key in self.events if key[0] == layer and key[1] == site]:
                                self.events.pop(key, None)
                    elif leaf == "lm_head":
                        self.events.pop((-1, 0, None), None)
            self._handles.append(module.register_forward_hook(hook))

        if self.neuron != "ann":
            original_matmul = temporal_model.temporal_seq_matmul
            original_hadamard = integration.temporal_symmetric_hadamard
            def matmul(a, b):
                out = original_matmul(a, b)
                call = self._matmul_call
                self._matmul_call += 1
                layer, kind = divmod(call, 2)
                if kind == 0:
                    ma, mb = self._event(layer, 2), self._event(layer, 3)
                    if mb is not None:
                        mb = mb.reshape(mb.shape[0], mb.shape[1], mb.shape[2], a.shape[2], a.shape[-1]).transpose(2, 3).transpose(-2, -1)
                else:
                    ma, mb = self._event(layer, 5), self._event(layer, 4)
                    if mb is not None:
                        mb = mb.reshape(mb.shape[0], mb.shape[1], mb.shape[2], b.shape[2], b.shape[-1]).transpose(2, 3)
                if ma is not None and ma.shape != a.shape:
                    if kind == 0:
                        ma = ma if ma.shape == a.shape else ma.reshape(a.shape)
                if mb is not None and mb.shape != b.shape:
                    mb = mb.reshape(b.shape)
                self.counts.add(count_temporal_matmul(a, b, ma, mb))
                if kind == 1:
                    for site in (2, 3, 4, 5):
                        self.events.pop((layer, site, None), None)
                return out
            def hadamard(a, b):
                out = original_hadamard(a, b)
                layer = self._hadamard_layer
                self._hadamard_layer += 1
                ma, mb = self._event(layer, 8), self._event(layer, 9)
                self.counts.add(temporal_product_counts(a, b, ma, mb))
                self.events.pop((layer, 8, None), None)
                self.events.pop((layer, 9, None), None)
                return out
            self._patch(temporal_model, "temporal_seq_matmul", matmul)
            self._patch(integration, "temporal_symmetric_hadamard", hadamard)
        return self

    def __exit__(self, *_):
        for handle in self._handles:
            handle.remove()
        for obj, name, original in reversed(self._originals):
            setattr(obj, name, original)
        self.events.clear()


def count_temporal_matmul(a: torch.Tensor, b: torch.Tensor, ma: torch.Tensor | None, mb: torch.Tensor | None) -> EnergyCounts:
    if a.shape[:-2] != b.shape[:-2] or a.shape[-1] != b.shape[-2]:
        raise ValueError("Temporal matmul operand shapes are incompatible")
    if (ma is not None and ma.shape != a.shape) or (mb is not None and mb.shape != b.shape):
        raise ValueError("Temporal event multiplicity shape differs from its operand")
    if ma is None and mb is None:
        return EnergyCounts(mac_raw=3 * a.numel() * int(b.shape[-1]))
    if ma is not None and mb is not None:
        count = _pair(ma.cumsum(0), mb) + _pair(ma, mb.cumsum(0)) + _pair(ma, mb)
        return EnergyCounts(synaptic_ac_raw=count)
    if ma is None:
        current = int(a.shape[-2]) * int(mb.sum().item())
        cumulative = int(a.shape[-2]) * int(mb.cumsum(0).sum().item())
        return EnergyCounts(mac_raw=2 * current + cumulative)
    current = int(b.shape[-1]) * int(ma.sum().item())
    cumulative = int(b.shape[-1]) * int(ma.cumsum(0).sum().item())
    return EnergyCounts(mac_raw=cumulative + 2 * current)


def validate_energy_deployment_protocol(cfg: dict, neuron: str) -> None:
    """Require the source checkpoint used in the final deployment Energy table."""
    mode = cfg["experiment"]["ann_mode"]
    if neuron == "ann":
        if mode != "vanilla":
            raise ValueError(
                f"Final ANN Energy requires ann_mode=vanilla, got {mode!r}"
            )
        return
    if neuron in {"phase", "gif", "mtn"}:
        if mode not in {"phase_aware", "gif_aware"}:
            raise ValueError(
                "Final SNN Energy requires a selected phase_aware or gif_aware "
                f"checkpoint, got ann_mode={mode!r}"
            )
        return
    raise ValueError(f"Unknown Energy neuron: {neuron!r}")


def profile_energy(cfg: dict, layout: Any, *, neuron: str) -> dict:
    validate_energy_deployment_protocol(cfg, neuron)
    source = model_source_for_stage(cfg, layout, stage="post_finetuning")
    if neuron == "ann" and cfg["experiment"]["ann_mode"] in {"phase_aware", "gif_aware"}:
        validate_recorded_training_artifact_provenance(cfg, layout)
    controller, _ = build_evaluation_controller(cfg, layout, neuron=neuron)
    model = load_model(cfg, source, training=False)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = load_tokenizer(cfg, source)
    if neuron != "ann" or cfg["rotation"]["enabled"]:
        install_model_integration(model, controller, rotation_state(cfg, layout))
    prefix_stage = "final_ann_evaluation" if neuron == "ann" else "final_snn_evaluation"
    cache = prefix_key_values_for_stage(cfg, layout, stage=prefix_stage)
    prefix_tokens = prefix_length(cache) if cache is not None else 0
    if cache is not None:
        install_prefix_kv_forward(model, cache, controller=controller)
    model.eval()
    bundle = load_selected_raw(cfg, layout)
    seed, n = int(cfg["calibration"]["seed"]), int(cfg["calibration"]["num_samples"])
    positions = select_validation_positions(len(bundle.validation), seed, n)
    manifest_path = layout.data_dir / "validation_manifest.json"
    manifest_hash = sha256_file(manifest_path)
    total = EnergyCounts()
    device = next(model.parameters()).device
    with torch.no_grad(), EnergyInstrumentation(model, controller, neuron, prefix_tokens=prefix_tokens) as instrument:
        for position in positions:
            instrument.reset()
            ids, mask = fixed_length_input(bundle.validation[position], tokenizer, cfg)
            ids, mask = ids.to(device), mask.to(device)
            pos = position_ids_from_attention_mask(mask)
            if neuron == "ann":
                model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False)
                # Dense attention and MLP products are not nn.Linear modules.
                for layer in model.model.layers:
                    attention = layer.self_attn
                    heads = int(attention.config.num_attention_heads) if hasattr(attention, "config") else int(model.config.num_attention_heads)
                    dim = int(attention.head_dim)
                    instrument.counts.mac_raw += 2 * heads * SEQUENCE_LENGTH * (SEQUENCE_LENGTH + prefix_tokens) * dim
                    instrument.counts.mac_raw += SEQUENCE_LENGTH * int(layer.mlp.gate_proj.out_features)
            else:
                temporal_forward(model, controller, ids, mask, position_ids=pos)
            total.add(instrument.counts)
            instrument.events.clear()
    mean_mac = total.mac_raw / n
    mean_syn = total.synaptic_ac_raw / n
    mean_neuron = total.neuron_ac_raw / n
    result = dict(zip(COLUMNS, (cfg["experiment"]["model_name"], neuron, None if neuron == "ann" else int(controller.temporal_steps), raw_to_g(mean_mac), raw_to_g(mean_syn), raw_to_g(mean_neuron), raw_to_g(mean_syn + mean_neuron), energy_j_from_raw_counts(mean_mac, mean_syn + mean_neuron))))
    output = layout.energy_dir(neuron)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "energy_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerow(result)
    write_json(output / "energy_results.json", result)
    write_json(output / "energy_sample_manifest.json", {"source": "validation_manifest", "sequence_length": SEQUENCE_LENGTH, "num_samples": n, "selection_seed_source": "calibration.seed", "selection_seed": seed, "sampling": "seeded_random_without_replacement", "positions_in_validation": positions, "validation_manifest_sha256": manifest_hash})
    metadata = {"energy_profiler_version": ENERGY_PROFILER_VERSION, "energy_accounting_policy": ENERGY_ACCOUNTING_POLICY, "experiment_id": cfg["experiment"].get("id"), "task": cfg["experiment"]["task"], "model_name": cfg["experiment"]["model_name"], "ann_mode": cfg["experiment"]["ann_mode"], "source_ann_mode": cfg["experiment"]["ann_mode"], "energy_deployment_protocol": "vanilla_ann_vs_single_selected_aware_checkpoint_snn", "energy_deployment_role": "ann_baseline" if neuron == "ann" else "selected_aware_snn", "neuron": neuron, "deployment_T": result["T"], "phase_base": cfg["phase"]["base"] if neuron == "phase" else None, "mtn_K": cfg["mtn"]["K"] if neuron == "mtn" else None, "conversion_use_post_finetuning_artifacts": cfg["conversion"]["use_post_finetuning_artifacts"], "evaluation_prefix_enabled": cfg["evaluation"]["prefix_enabled"], "prefix_artifact_stage": final_ann_evaluation_prefix_artifact_stage(cfg) if neuron == "ann" else final_snn_evaluation_prefix_artifact_stage(cfg), "prefix_length": prefix_tokens, "profile_sequence_length": SEQUENCE_LENGTH, "profile_num_samples": n, "profile_seed": seed, "selected_validation_positions": positions, "validation_manifest_path": str(manifest_path), "validation_manifest_sha256": manifest_hash, "checkpoint_source": source, "controller_mode": controller.mode, "mac_energy_pj": 4.6, "ac_energy_pj": 0.9, "energy_scope": "mac_synaptic_ac_neuron_ac_only", "memory_energy_included": False, "dense_residual_bias_ac_included": False, "dense_elementwise_additions_included": False, "special_function_energy_included": False, "hadamard_rotation_energy_included": False, "fixed_neuron_coefficient_policy": "prefold_or_fixed_event_lookup", "gif_unit_event_policy": "integer_code_expanded_to_unit_events", "gif_zero_point_compensation_policy": "fixed_prefolded_or_bias_like_compensation_not_counted", "gif_asymmetric_zero_point_energy_included": False, "energy_path_profile_num_samples": n, "energy_path_prefix_enabled": layout.energy_prefix_enabled(neuron), "gif_neuron_ac_policy": "no_recurrent_membrane_ac_in_current_static_gif_temporal_impl", "total_mac_raw": total.mac_raw, "total_synaptic_ac_raw": total.synaptic_ac_raw, "total_neuron_ac_raw": total.neuron_ac_raw, "mean_mac_raw": mean_mac, "mean_synaptic_ac_raw": mean_syn, "mean_neuron_ac_raw": mean_neuron}
    write_json(output / "energy_metadata.json", metadata)
    return result
