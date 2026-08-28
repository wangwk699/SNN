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
        "saliency_row_count": torch.ones(4, dtype=torch.long),
        "saliency_sum": torch.arange(4, dtype=torch.float64),
        "phase_ema_abs_max": torch.ones(4),
        "phase_ema_updates": torch.ones(4, dtype=torch.long),
        "phase_tau_calibration": PHASE_TAU_CALIBRATION,
        "phase_tau_ema_factor": PHASE_TAU_EMA_FACTOR,
        "phase_tau_accumulator_dtype": PHASE_TAU_ACCUMULATOR_DTYPE,
        "phase_tau_channel_policy": PHASE_TAU_CHANNEL_POLICY,
        "phase_tau_reduction_policy": PHASE_TAU_REDUCTION_POLICY,
    }
    cfg = {
        "calibration": {"group_size": -1, "num_samples": 128, "seed": 42},
        "phase": {"T": 4, "base": 2.0, "surrogate_slope": 1.0},
        "gif": {"base_bits": 4, "add_bits": 1, "low_ratio": 0.5, "salient_ratio": 0.5},
        "mtn": {"T": 4, "K": 6, "threshold_factor": 0.75},
    }

    state = build_site_states(statistics, cfg, include_clip=False)["gif"]

    assert state["high_qmax"] == 30
    torch.testing.assert_close(
        state["high_scale"], torch.tensor([2.0 / 30.0]), rtol=1e-6, atol=1e-8
    )
    torch.testing.assert_close(state["high_zero"], torch.tensor([15.0]))
    assert state["integer_decomposition"] == "two_unsigned_chunks_each_0_to_15_high_qmax_30"
