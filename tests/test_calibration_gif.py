import torch

from snn2.calibration import build_site_states
from snn2.phase_statistics import (
    PHASE_TAU_ACCUMULATOR_DTYPE, PHASE_TAU_CALIBRATION,
    PHASE_TAU_CHANNEL_POLICY, PHASE_TAU_EMA_FACTOR, PHASE_TAU_REDUCTION_POLICY,
)
from snn2.temporal_ops import STATISTICS_FORMAT_VERSION


def test_gif_high_qparams_use_two_chunk_capacity():
    statistics = {
        "format_version": STATISTICS_FORMAT_VERSION, "site_index": 1,
        "layout_kind": "last_dim", "num_heads": None,
        "channels_per_head": None, "channels": 4,
        "value_min": torch.full((4,), -1.0),
        "value_max": torch.full((4,), 1.0),
        "saliency_row_count_by_role": {role: torch.ones(4, dtype=torch.long) for role in ("q", "k", "v")},
        "saliency_sum_by_role": {role: torch.arange(4, dtype=torch.float32) for role in ("q", "k", "v")},
        "saliency_rule_by_role": {role: "spikellm_linear_fp32" for role in ("q", "k", "v")},
        "saliency_accumulator_dtype_by_role": {role: "float32" for role in ("q", "k", "v")},
        "phase_ema_abs_max": torch.ones(4),
        "phase_ema_updates": torch.ones(4, dtype=torch.long),
        "phase_tau_calibration": PHASE_TAU_CALIBRATION,
        "phase_tau_ema_factor": PHASE_TAU_EMA_FACTOR,
        "phase_tau_accumulator_dtype": PHASE_TAU_ACCUMULATOR_DTYPE,
        "phase_tau_channel_policy": PHASE_TAU_CHANNEL_POLICY,
        "phase_tau_reduction_policy": PHASE_TAU_REDUCTION_POLICY,
    }
    cfg = {
        "calibration": {"group_size": -1},
        "phase": {"T": 4, "base": 2.0, "surrogate_slope": 1.0},
        "gif": {"base_bits": 4, "add_bits": 1, "low_ratio": 0.5},
        "mtn": {"T": 4, "K": 6, "threshold_factor": 0.75},
    }

    state = build_site_states(statistics, cfg)["gif"]

    assert state["high_qmax"] == 30
    torch.testing.assert_close(
        state["high_scale"], torch.tensor([2.0 / 30.0]), rtol=1e-6, atol=1e-8
    )
    torch.testing.assert_close(state["high_zero"], torch.tensor([15.0]))
    assert state["integer_decomposition"] == "two_unsigned_chunks_each_0_to_15_high_qmax_30"


def test_spikellm_threshold_preserves_ties():
    from snn2.calibration import spikellm_mask_low
    score = torch.tensor([0.0, 1.0, 1.0, 2.0])
    mask = spikellm_mask_low(score, 0.5)
    assert torch.equal(mask, torch.tensor([True, True, True, False]))


def test_site3_global_threshold_is_not_per_head():
    from snn2.calibration import spikellm_mask_low
    score = torch.tensor([[0.0, 1.0], [100.0, 101.0]])
    mask = spikellm_mask_low(score, 0.5)
    assert torch.equal(mask, torch.tensor([[True, True], [False, False]]))
