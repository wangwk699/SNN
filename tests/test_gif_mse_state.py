from __future__ import annotations

from copy import deepcopy

import pytest

import torch

from snn2.calibration import build_gif_state
from snn2.config import load_config, validate_config
from snn2.gif_mse_calibration import GIFMSEHistogramStore, runtime_fake_quant
from snn2.gif_mse_integration import histogram_provenance, save_histogram_store
from snn2.gif_mse_provenance import validate_histogram_provenance
from snn2.gif_mse_state import build_histogram_spec
from snn2.gif_mse_validation import validate_gif_mse_state
from snn2.phase_statistics import (
    PHASE_TAU_ACCUMULATOR_DTYPE,
    PHASE_TAU_CALIBRATION,
    PHASE_TAU_CHANNEL_POLICY,
    PHASE_TAU_EMA_FACTOR,
    PHASE_TAU_REDUCTION_POLICY,
)
from snn2.temporal_ops import GIF_SCALE_MIN, STATISTICS_FORMAT_VERSION


def _statistics(width: int = 4):
    return {
        "format_version": STATISTICS_FORMAT_VERSION,
        "site_index": 2,
        "layout_kind": "last_dim",
        "num_heads": None,
        "channels_per_head": None,
        "channels": width,
        "value_min": torch.full((width,), -1.0, dtype=torch.float64),
        "value_max": torch.full((width,), 10.0, dtype=torch.float64),
        "saliency_row_count_by_role": {},
        "saliency_sum_by_role": {},
        "saliency_rule_by_role": {},
        "saliency_accumulator_dtype_by_role": {},
        "phase_ema_abs_max": torch.ones(width),
        "phase_ema_updates": torch.ones(width, dtype=torch.long),
        "phase_tau_calibration": PHASE_TAU_CALIBRATION,
        "phase_tau_ema_factor": PHASE_TAU_EMA_FACTOR,
        "phase_tau_accumulator_dtype": PHASE_TAU_ACCUMULATOR_DTYPE,
        "phase_tau_channel_policy": PHASE_TAU_CHANNEL_POLICY,
        "phase_tau_reduction_policy": PHASE_TAU_REDUCTION_POLICY,
    }


def test_histogram_materializes_fixed_refined_state_and_preserves_direct_baseline():
    cfg = load_config("configs/generated/exp2_llama3_8b_tulu3__gif_aware.yaml")
    cfg["calibration"]["group_size"] = -1
    cfg["gif"]["mse_scale_refinement"] = True
    cfg["gif"]["mse_refinement"]["histogram_bins"] = 128
    direct_cfg = deepcopy(cfg)
    direct_cfg["gif"]["mse_scale_refinement"] = False
    statistics = _statistics()
    direct = build_gif_state(statistics, direct_cfg)
    spec = build_histogram_spec(statistics, direct, configured_group_size=-1)
    store = GIFMSEHistogramStore({"layer_000/site_02": spec}, histogram_bins=128)
    values = torch.cat((torch.linspace(-1.0, 1.0, 4000), torch.tensor([10.0])))
    activation = values[:, None].repeat(1, 4)
    store.update(0, 2, activation)
    histogram = store.state_for(
        "layer_000/site_02",
        histogram_provenance(cfg, {}, trajectory_source="ann_common"),
    )
    refined = build_gif_state(statistics, cfg, histogram)
    assert refined["qparam_calibration_method"] == "offline_static_mse"
    assert refined["runtime_quantization"] == "static"
    torch.testing.assert_close(refined["direct_low_scale"], direct["low_scale"])
    torch.testing.assert_close(refined["direct_low_zero"], direct["low_zero"])
    diagnostic = refined["mse_diagnostics"]["low"]
    assert torch.all(diagnostic["refined_mse"] <= diagnostic["baseline_mse"] + 1e-12)
    assert torch.all(refined["low_scale"] > 0)
    assert torch.equal(refined["low_zero"], refined["low_zero"].round())


class _ProbeCollector:
    def __init__(self):
        self.calls = []

    def update(self, layer, site, value, *, role=None):
        self.calls.append((layer, site, value.clone(), role))


def test_controller_histogram_mode_is_identity_and_does_not_touch_statistics():
    from snn2.controller import SiteController

    controller = SiteController(mode="collect")
    probe = _ProbeCollector()
    controller.gif_mse_collector = probe
    controller.begin_gif_mse_collection()
    before = dict(controller.statistics.items)
    value = torch.randn(2, 3, 4)
    output = controller.apply(0, 8, value)
    assert output is value
    assert controller.statistics.items == before
    assert len(probe.calls) == 1


def test_sequential_histogram_mode_collects_logical_sum_without_deployment():
    from snn2.controller import SiteController
    controller = SiteController(mode="collect")
    probe = _ProbeCollector()
    controller.gif_mse_collector = probe
    controller.begin_gif_mse_collection(block_index=0)
    logical = torch.randn(1, 2, 4)
    repeated = torch.cat((logical * 0.25, logical * 0.75), dim=0)
    output = controller.apply(0, 8, repeated)
    assert output is repeated
    torch.testing.assert_close(probe.calls[0][2], logical)



def test_common_calibration_runs_one_or_two_passes_with_identical_sample_order(
    monkeypatch, tmp_path,
):
    from types import SimpleNamespace

    from torch import nn

    from snn2 import calibration
    from snn2.controller import SiteController

    class ToyModel(nn.Module):
        def __init__(self, controller):
            super().__init__()
            self.anchor = nn.Parameter(torch.ones(()))
            self.config = SimpleNamespace(num_hidden_layers=1)
            self.model = nn.Module()
            self.model.norm = nn.Identity()
            self.model.layers = nn.ModuleList([nn.Identity()])
            self.model.embed_tokens = nn.Embedding(32, 4)
            self.lm_head = nn.Linear(4, 32, bias=False)
            self.controller = controller
            self.seen = []

        def forward(self, input_ids, attention_mask=None, **_kwargs):
            self.seen.extend(input_ids[:, 0].tolist())
            value = torch.ones(input_ids.shape[0], input_ids.shape[1], 4)
            for site in range(1, 11):
                if site in {2, 3, 4}:
                    activation = value.reshape(value.shape[0], 1, value.shape[1], 4)
                elif site == 5:
                    activation = torch.ones(
                        value.shape[0], 1, value.shape[1], value.shape[1]
                    )
                else:
                    activation = value
                self.controller.apply(0, site, activation)
            return SimpleNamespace(logits=value)

    rows = [
        {"input_ids": torch.tensor([11, 1]), "attention_mask": torch.ones(2, dtype=torch.long)},
        {"input_ids": torch.tensor([22, 2]), "attention_mask": torch.ones(2, dtype=torch.long)},
    ]
    monkeypatch.setattr(calibration, "tokenize_dataset", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(
        calibration, "CausalLMCollator",
        lambda _tokenizer: lambda batch: {
            "input_ids": batch[0]["input_ids"].unsqueeze(0),
            "attention_mask": batch[0]["attention_mask"].unsqueeze(0),
        },
    )
    monkeypatch.setattr(
        calibration, "install_prefix_kv_forward", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        calibration, "create_histogram_store", lambda *_args, **_kwargs: _ProbeCollector()
    )
    monkeypatch.setattr(
        calibration, "save_histogram_store", lambda *_args, **_kwargs: None
    )

    for enabled, expected in ((False, [11, 22]), (True, [11, 22, 11, 22])):
        cfg = load_config("configs/generated/exp2_llama3_8b_tulu3__gif_aware.yaml")
        cfg["calibration"]["gif_previous_layers_snn"] = False
        cfg["gif"]["mse_scale_refinement"] = enabled
        controller = SiteController(mode="collect")
        model = ToyModel(controller)
        calibration.collect_site_statistics(
            model, controller, object(), object(), cfg, None,
            tmp_path / str(enabled), purpose="ann_training_calibration",
            materialize_states=False,
        )
        assert model.seen == expected



def test_runtime_fake_quant_matches_staticgif_fp32_forward():
    from snn2.neurons import gif_module_from_state

    cfg = load_config("configs/generated/exp2_llama3_8b_tulu3__gif_aware.yaml")
    cfg["calibration"]["group_size"] = -1
    direct_cfg = deepcopy(cfg)
    direct_cfg["gif"]["mse_scale_refinement"] = False
    statistics = _statistics()
    statistics["site_index"] = 10
    statistics.update({
        "saliency_sum_by_role": {"default": torch.ones(4)},
        "saliency_row_count_by_role": {"default": torch.ones(4, dtype=torch.long)},
        "saliency_rule_by_role": {"default": "test"},
        "saliency_accumulator_dtype_by_role": {"default": "float64"},
    })
    state = build_gif_state(statistics, direct_cfg)
    state["mask_low"] = torch.ones_like(state["mask_low"], dtype=torch.bool)
    module = gif_module_from_state(state)
    scale = float(state["low_scale"].reshape(-1)[0])
    zero = int(state["low_zero"].reshape(-1)[0])
    values = torch.tensor([-1.0, -0.5 * scale, 0.5 * scale, 1.5 * scale,
                           -1.5 * scale, -0.49 * scale, 1.49 * scale, 3.0]).reshape(2, 4)
    expected = runtime_fake_quant(values, scale, zero, qmin=0, qmax=15).float()
    assert torch.equal(module(values), expected)


def test_mse_fixed_boolean_options_reject_false():
    cfg = load_config("configs/generated/exp2_llama3_8b_tulu3__gif_aware.yaml")
    for key in ("preserve_zero", "fallback_to_direct_min_max"):
        bad = deepcopy(cfg)
        bad["gif"]["mse_refinement"][key] = False
        with pytest.raises(ValueError, match="fixed and must be true"):
            validate_config(bad)



def test_histogram_binds_current_source_statistics_hash(tmp_path):
    class Store:
        specs = {"layer_000/site_02": {}}

        @staticmethod
        def state_for(_key, metadata):
            return {"format_version": 2, "configured_group_size": 128, **metadata}

    cfg = load_config("configs/generated/exp2_llama3_8b_tulu3__gif_aware.yaml")
    cfg["calibration"]["gif_previous_layers_snn"] = False
    directory = tmp_path / "layer_000" / "site_02"
    directory.mkdir(parents=True)
    source = directory / "statistics.pt"
    torch.save({"value": 1}, source)
    manifest = {
        "calibration_data_manifest_sha256": None,
        "prefix_state_sha256": None,
        "prefix_kv_sha256": None,
        "rotation_state_sha256": None,
    }
    save_histogram_store(
        Store(), tmp_path, histogram_provenance(cfg, manifest, trajectory_source="ann_common"),
        statistics_name="statistics.pt",
    )
    histogram_path = directory / "gif_mse_histogram.pt"
    histogram = torch.load(histogram_path, weights_only=False)
    validate_histogram_provenance(histogram, manifest, cfg, site_directory=directory)
    torch.save({"value": 2}, source)
    with pytest.raises(ValueError, match="source statistics hash"):
        validate_histogram_provenance(histogram, manifest, cfg, site_directory=directory)



def test_sequential_histogram_binds_gif_statistics_hash(tmp_path):
    class Store:
        specs = {"layer_000/site_02": {}}

        @staticmethod
        def state_for(_key, metadata):
            return {"format_version": 2, "configured_group_size": 128, **metadata}

    cfg = load_config("configs/generated/exp2_llama3_8b_tulu3__gif_aware.yaml")
    cfg["calibration"]["gif_previous_layers_snn"] = True
    directory = tmp_path / "layer_000" / "site_02"
    directory.mkdir(parents=True)
    source = directory / "gif_statistics.pt"
    torch.save({"value": 1}, source)
    manifest = {
        "calibration_data_manifest_sha256": None,
        "prefix_state_sha256": None,
        "prefix_kv_sha256": None,
        "rotation_state_sha256": None,
    }
    save_histogram_store(
        Store(), tmp_path,
        histogram_provenance(cfg, manifest, trajectory_source="sequential_temporal_gif"),
        statistics_name="gif_statistics.pt",
    )
    histogram = torch.load(directory / "gif_mse_histogram.pt", weights_only=False)
    validate_histogram_provenance(histogram, manifest, cfg, site_directory=directory)
    torch.save({"value": 2}, source)
    with pytest.raises(ValueError, match="source statistics hash"):
        validate_histogram_provenance(histogram, manifest, cfg, site_directory=directory)


def test_common_calibration_cleanup_removes_stale_histogram(tmp_path):
    from snn2.calibration import clear_common_gif_mse_histograms

    stale = tmp_path / "layer_000" / "site_02" / "gif_mse_histogram.pt"
    stale.parent.mkdir(parents=True)
    torch.save({"sentinel": True}, stale)
    clear_common_gif_mse_histograms(tmp_path)
    assert not stale.exists()



def test_gif_scale_floor_matches_mse_ann_and_temporal_paths():
    from snn2.neurons import gif_module_from_state

    cfg = load_config("configs/generated/exp2_llama3_8b_tulu3__gif_aware.yaml")
    cfg["calibration"]["group_size"] = -1
    statistics = _statistics()
    statistics["site_index"] = 10
    statistics.update({
        "saliency_sum_by_role": {"default": torch.ones(4)},
        "saliency_row_count_by_role": {"default": torch.ones(4, dtype=torch.long)},
        "saliency_rule_by_role": {"default": "test"},
        "saliency_accumulator_dtype_by_role": {"default": "float64"},
    })
    state = build_gif_state(statistics, cfg)
    values = torch.tensor([
        [-1.5e-8, -0.5e-8, 0.5e-8, 1.5e-8],
        [2.5e-8, -2.5e-8, 7.5e-8, -7.5e-8],
    ])
    for low_branch, qmax in ((True, 15), (False, 30)):
        branch_state = deepcopy(state)
        branch_state["mask_low"] = torch.full_like(branch_state["mask_low"], low_branch)
        branch_state["low_scale"] = torch.full_like(branch_state["low_scale"], 0.95 * GIF_SCALE_MIN)
        branch_state["high_scale"] = torch.full_like(branch_state["high_scale"], 0.95 * GIF_SCALE_MIN)
        module = gif_module_from_state(branch_state)
        scale_key, zero_key = ("low_scale", "low_zero") if low_branch else ("high_scale", "high_zero")
        zero = int(branch_state[zero_key].reshape(-1)[0])
        mse = runtime_fake_quant(values, 0.95 * GIF_SCALE_MIN, zero, qmin=0, qmax=qmax).float()
        ann = module(values)
        temporal = module.temporal(torch.stack((values, torch.zeros_like(values)))).sum(dim=0)
        assert torch.equal(mse, ann)
        torch.testing.assert_close(ann, temporal, rtol=0.0, atol=1e-14)



def test_mse_state_rejects_scale_below_runtime_floor(tmp_path):
    cfg = load_config("configs/generated/exp2_llama3_8b_tulu3__gif_aware.yaml")
    cfg["calibration"]["group_size"] = -1
    cfg["gif"]["mse_scale_refinement"] = True
    cfg["gif"]["mse_refinement"]["histogram_bins"] = 16
    direct_cfg = deepcopy(cfg)
    direct_cfg["gif"]["mse_scale_refinement"] = False
    statistics = _statistics()
    direct = build_gif_state(statistics, direct_cfg)
    spec = build_histogram_spec(statistics, direct, configured_group_size=-1)
    store = GIFMSEHistogramStore({"layer_000/site_02": spec}, histogram_bins=16)
    store.update(0, 2, torch.zeros(2, 4))
    histogram = store.state_for(
        "layer_000/site_02", histogram_provenance(cfg, {}, trajectory_source="ann_common")
    )
    state = build_gif_state(statistics, cfg, histogram)
    path = tmp_path / "gif_state.pt"
    torch.save(histogram, tmp_path / "gif_mse_histogram.pt")
    state["low_scale"] = torch.full_like(state["low_scale"], 0.95 * GIF_SCALE_MIN)
    with pytest.raises(ValueError, match="below runtime GIF_SCALE_MIN"):
        validate_gif_mse_state(state, cfg, path=path)
