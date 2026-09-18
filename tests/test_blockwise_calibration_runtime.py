from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import yaml

from snn2.calibration import materialize_target_state
from snn2.config import resolve_config
from snn2.controller import SiteController
from snn2.model_integration import prepare_temporal_model_inputs, repeat_kv
from snn2.sites import SITE_IDS, site_key
from snn2.temporal_model import deployment_attention_forward


ROOT = Path(__file__).resolve().parents[1]


def _cfg() -> dict:
    raw = yaml.safe_load((ROOT / "configs" / "experiment_matrix.yaml").read_text())[
        "experiments"
    ][0]["config"]
    raw["experiment"]["ann_mode"] = "gif_aware"
    return resolve_config(copy.deepcopy(raw))


def test_temporal_collect_preserves_head_layout_and_excludes_prefix():
    controller = SiteController(
        mode="collect", phase_T=2, mtn_T=2, mtn_K=2, mtn_threshold_factor=0.75
    )
    controller.begin_sequential_calibration("gif", 0)
    module = SimpleNamespace(training=False, num_key_value_groups=1, scaling=0.5)
    # T=2, B=1; K/V include two Prefix positions and two token positions.
    query = torch.randn(2, 2, 2, 3)
    key = torch.randn(2, 2, 4, 3)
    value = torch.randn_like(key)
    mask = torch.zeros(2, 1, 2, 4)

    deployment_attention_forward(
        module, query, key, value, mask, scaling=None, dropout=0.0,
        controller=controller, layer_index=0, repeat_kv=repeat_kv, softcap=None,
    )

    for site in (3, 4):
        stats = controller.statistics.items[site_key(0, site)]
        assert stats.layout_kind == "attention_head"
        assert stats.num_heads == 2
        assert stats.channels_per_head == 3
        # One logical sample, two current positions; Prefix positions are excluded.
        assert stats.row_count.item() == 2
        assert bool(torch.all(stats.saliency_row_count_by_role["default"] == 2))


def test_target_state_uses_canonical_site_directories_and_fails_fast(tmp_path):
    cfg = _cfg()
    cfg["calibration"]["phase_previous_layers_snn"] = True
    cfg["calibration"]["group_size"] = -1
    for site in SITE_IDS:
        directory = tmp_path / site_key(0, site)
        directory.mkdir(parents=True)
        activation = (
            torch.ones(1, 2, 2, 3)
            if site in {2, 3, 4}
            else (torch.ones(1, 2, 2, 2) if site == 5 else torch.ones(1, 2, 3))
        )
        probe = SiteController(mode="collect")
        probe.record_activation(0, site, activation)
        state = probe.statistics.items[site_key(0, site)].state_dict()
        torch.save(state, directory / "phase_statistics.pt")

    materialize_target_state(tmp_path, cfg, "phase", layer_index=0)
    assert (tmp_path / site_key(0, 1) / "phase_state.pt").exists()

    (tmp_path / site_key(0, 2) / "phase_statistics.pt").unlink()
    try:
        materialize_target_state(tmp_path, cfg, "phase", layer_index=0)
    except FileNotFoundError as error:
        assert error.filename is None or "site_02_q_post_rope_r3" in str(error)
    else:
        raise AssertionError("missing target statistics must fail fast")


def test_temporal_input_preparation_is_shared_and_time_major():
    ids = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(ids)
    position_ids = torch.tensor([[4, 5, 6]])
    repeated_ids, repeated_mask, kwargs = prepare_temporal_model_inputs(
        ids, mask, steps=3, position_ids=position_ids,
        cache_position=torch.tensor([4, 5, 6]),
    )
    assert torch.equal(repeated_ids, ids.repeat(3, 1))
    assert torch.equal(repeated_mask, mask.repeat(3, 1))
    assert torch.equal(kwargs["position_ids"], position_ids.repeat(3, 1))

class _ToyBlock(nn.Module):
    def __init__(self, controller: SiteController, index: int):
        super().__init__()
        self.controller, self.index = controller, index
        self.anchor = nn.Parameter(torch.ones(()))
        self.collect_inputs: list[torch.Tensor] = []

    def forward(self, hidden_states, **_kwargs):
        if self.controller.collecting_statistics:
            self.collect_inputs.append(hidden_states.detach().clone())
        batch, length, width = hidden_states.shape
        for site in SITE_IDS:
            if site in {2, 3, 4}:
                activation = hidden_states.reshape(batch, 1, length, width)
            elif site == 5:
                activation = torch.ones(batch, 1, length, length, device=hidden_states.device)
            else:
                activation = hidden_states
            transformed = self.controller.apply(self.index, site, activation)
            if site == 1:
                hidden_states = transformed
        return (hidden_states,)


class _ToyFinalNorm(nn.Module):
    def __init__(self, controller: SiteController):
        super().__init__()
        self.controller = controller
        self.anchor = nn.Parameter(torch.ones(()))

    def forward(self, hidden_states):
        return self.controller.apply_final_norm_neuron(hidden_states)


class _ToyModel(nn.Module):
    def __init__(self, controller: SiteController):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(16, 4)
        with torch.no_grad():
            self.model.embed_tokens.weight.fill_(2.0)
        self.model.layers = nn.ModuleList([_ToyBlock(controller, 0), _ToyBlock(controller, 1)])
        self.model.norm = _ToyFinalNorm(controller)
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.config = SimpleNamespace(num_hidden_layers=2)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden, attention_mask=attention_mask, **kwargs)[0]
        hidden = self.model.norm(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


def test_blockwise_runner_e2e_uses_fresh_phase_state_and_clears_stale(monkeypatch, tmp_path):
    """Exercises bootstrap, block collect/materialize/deploy, and next-block input."""
    from snn2 import blockwise_calibration as blockwise

    cfg = _cfg()
    cfg["calibration"].update({"phase_previous_layers_snn": True, "group_size": -1})
    cfg["phase"]["T"] = 2
    controller = SiteController(
        mode="collect", site_root=tmp_path, phase_T=2, phase_base=2.0, mtn_T=2,
        mtn_K=2, mtn_threshold_factor=0.75,
    )
    model = _ToyModel(controller)
    stale = tmp_path / site_key(1, 6)
    stale.mkdir(parents=True)
    torch.save({"stale": 999}, stale / "phase_statistics.pt")
    torch.save({"stale": 999}, stale / "phase_state.pt")

    monkeypatch.setattr(blockwise, "tokenize_dataset", lambda *_args, **_kwargs: [
        {"input_ids": torch.tensor([1, 2, 3]), "attention_mask": torch.ones(3, dtype=torch.long)}
    ])
    monkeypatch.setattr(blockwise, "CausalLMCollator", lambda _tokenizer: lambda rows: {
        "input_ids": rows[0]["input_ids"].unsqueeze(0),
        "attention_mask": rows[0]["attention_mask"].unsqueeze(0),
    })
    monkeypatch.setattr(blockwise, "install_prefix_kv_forward", lambda *_args, **_kwargs: None)

    result = blockwise.collect_blockwise_snn_conditioned_statistics(
        model, controller, object(), object(), cfg, None, tmp_path, neuron="phase"
    )
    assert result == {"neuron": "phase", "blocks": 2, "samples": 1}
    assert (tmp_path / site_key(0, 1) / "phase_statistics.pt").exists()
    assert (tmp_path / site_key(1, 6) / "phase_state.pt").exists()
    assert not torch.equal(model.model.layers[0].collect_inputs[0], model.model.layers[1].collect_inputs[0])

@pytest.mark.parametrize("neuron", ("phase", "gif", "mtn"))
def test_clear_target_trajectory_artifacts_preserves_common_and_other_neurons(tmp_path, neuron):
    from snn2.blockwise_calibration import clear_target_trajectory_artifacts

    directory = tmp_path / site_key(0, 1)
    directory.mkdir(parents=True)
    for name in ("statistics.pt", "phase_statistics.pt", "phase_state.pt", "gif_statistics.pt", "gif_state.pt", "mtn_statistics.pt", "mtn_state.pt"):
        (directory / name).write_bytes(b"state")
    global_directory = tmp_path / "_global" / "final_rmsnorm"
    global_directory.mkdir(parents=True)
    for name in ("phase_statistics.pt", "phase_state.pt", "mtn_statistics.pt", "mtn_state.pt"):
        (global_directory / name).write_bytes(b"state")

    clear_target_trajectory_artifacts(tmp_path, neuron)
    assert (directory / "statistics.pt").exists()
    for other in {"phase", "gif", "mtn"} - {neuron}:
        assert (directory / f"{other}_statistics.pt").exists()
        assert (directory / f"{other}_state.pt").exists()
    assert not (directory / f"{neuron}_statistics.pt").exists()
    assert not (directory / f"{neuron}_state.pt").exists()
    if neuron in {"phase", "mtn"}:
        assert not (global_directory / f"{neuron}_statistics.pt").exists()
        assert not (global_directory / f"{neuron}_state.pt").exists()



def test_gif_mse_blockwise_order_is_statistics_histogram_state_deploy(
    monkeypatch, tmp_path,
):
    from snn2 import blockwise_calibration as blockwise

    cfg = _cfg()
    cfg["calibration"].update({"gif_previous_layers_snn": True, "group_size": -1})
    cfg["gif"]["mse_scale_refinement"] = True
    cfg["gif"]["mse_refinement"]["histogram_bins"] = 16
    controller = SiteController(
        mode="collect", site_root=tmp_path, phase_T=2, phase_base=2.0, mtn_T=2,
        mtn_K=2, mtn_threshold_factor=0.75,
    )
    model = _ToyModel(controller)
    cached = [blockwise._LayerInput(torch.ones(1, 2, 4), {})]
    events = []

    monkeypatch.setattr(blockwise, "tokenize_dataset", lambda *_args, **_kwargs: [{}])
    monkeypatch.setattr(blockwise, "CausalLMCollator", lambda _tokenizer: lambda rows: rows[0])
    monkeypatch.setattr(blockwise, "install_prefix_kv_forward", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(blockwise, "_bootstrap", lambda *_args, **_kwargs: cached)
    monkeypatch.setattr(
        blockwise, "_run",
        lambda _layer, item, **_kwargs: item.hidden_states + 1,
    )
    monkeypatch.setattr(
        blockwise, "_save_target_statistics",
        lambda _store, _root, _neuron, *, layer_index=None, **_kwargs:
            events.append(f"statistics_{layer_index}"),
    )
    monkeypatch.setattr(
        blockwise, "create_histogram_store",
        lambda _root, _cfg, *, statistics_name, layer_index:
            events.append(f"histogram_{layer_index}") or SimpleNamespace(),
    )
    monkeypatch.setattr(
        blockwise, "save_histogram_store",
        lambda _store, _root, _metadata, **_kwargs: events.append("histogram_saved"),
    )
    monkeypatch.setattr(
        blockwise, "materialize_target_state",
        lambda _root, _cfg, _neuron, *, layer_index=None, **_kwargs:
            events.append(f"state_{layer_index}"),
    )
    monkeypatch.setattr(blockwise, "read_json", lambda _path: {})
    original_deploy = controller.begin_sequential_deployment

    def tracked_deploy(layer_index):
        events.append(f"deploy_{layer_index}")
        original_deploy(layer_index)

    monkeypatch.setattr(controller, "begin_sequential_deployment", tracked_deploy)
    result = blockwise.collect_blockwise_snn_conditioned_statistics(
        model, controller, object(), object(), cfg, None, tmp_path, neuron="gif"
    )
    assert result == {"neuron": "gif", "blocks": 2, "samples": 1}
    assert events == [
        "statistics_0", "histogram_0", "histogram_saved", "state_0", "deploy_0",
        "statistics_1", "histogram_1", "histogram_saved", "state_1", "deploy_1",
    ]
