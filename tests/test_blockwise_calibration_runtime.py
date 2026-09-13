from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import torch
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
