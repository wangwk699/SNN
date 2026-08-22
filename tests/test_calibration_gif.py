import torch

from snn2.calibration import build_site_states


def test_gif_high_qparams_use_two_chunk_capacity():
    statistics = {
        "channels": 4,
        "value_min": torch.full((4,), -1.0),
        "value_max": torch.full((4,), 1.0),
        "saliency_row_count": torch.ones(4, dtype=torch.long),
        "saliency_sum": torch.arange(4, dtype=torch.float64),
    }
    cfg = {
        "calibration": {"group_size": -1},
        "phase": {"T": 4, "base": 2.0, "surrogate_slope": 4.0, "max_spikes": 4},
        "gif": {"base_bits": 4, "add_bits": 1, "low_ratio": 0.5},
        "mtn": {"T": 4, "K": 6, "threshold_factor": 0.75},
    }

    state = build_site_states(statistics, cfg)["gif"]

    assert state["high_qmax"] == 30
    torch.testing.assert_close(
        state["high_scale"], torch.tensor([2.0 / 30.0]), rtol=1e-6, atol=1e-8
    )
    torch.testing.assert_close(state["high_zero"], torch.tensor([15.0]))
    assert state["integer_decomposition"].startswith("unsigned_q_capped")
