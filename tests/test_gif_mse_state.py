from __future__ import annotations

from copy import deepcopy

import torch

from snn2.calibration import build_gif_state
from snn2.config import load_config
from snn2.gif_mse_calibration import GIFMSEHistogramStore
from snn2.gif_mse_integration import histogram_provenance
from snn2.gif_mse_state import build_histogram_spec
from snn2.phase_statistics import (
    PHASE_TAU_ACCUMULATOR_DTYPE,
    PHASE_TAU_CALIBRATION,
    PHASE_TAU_CHANNEL_POLICY,
    PHASE_TAU_EMA_FACTOR,
    PHASE_TAU_REDUCTION_POLICY,
)
from snn2.temporal_ops import STATISTICS_FORMAT_VERSION


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
        cfg["gif"]["mse_scale_refinement"] = enabled
        controller = SiteController(mode="collect")
        model = ToyModel(controller)
        calibration.collect_site_statistics(
            model, controller, object(), object(), cfg, None,
            tmp_path / str(enabled), purpose="ann_training_calibration",
            materialize_states=False,
        )
        assert model.seen == expected
