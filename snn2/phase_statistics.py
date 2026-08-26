from __future__ import annotations


PHASE_TAU_CALIBRATION = "ema_channel_abs_max_then_group_max"
PHASE_TAU_EMA_FACTOR = 0.99
PHASE_TAU_ACCUMULATOR_DTYPE = "float32"
PHASE_TAU_CHANNEL_POLICY = "native_site_layout_per_channel"
PHASE_TAU_REDUCTION_POLICY = "within_group_max_after_channel_ema"
