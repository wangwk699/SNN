from __future__ import annotations


PHASE_TAU_CALIBRATION = "ema_channel_abs_max_then_group_max"
PHASE_TAU_EMA_FACTOR = 0.99
PHASE_TAU_ACCUMULATOR_DTYPE = "float32"
PHASE_TAU_CHANNEL_POLICY = "native_site_layout_per_channel"

# Shared frozen parameter-statistics contract for Phase and MTN.
PARAMETER_CALIBRATION = "ema_channel_abs_max_then_group_max"
PARAMETER_EMA_FACTOR = 0.99
PARAMETER_ACCUMULATOR_DTYPE = "float32"
PARAMETER_CHANNEL_POLICY = "native_site_layout_per_channel"
PARAMETER_REDUCTION_POLICY = "within_group_max_after_channel_ema"
PARAMETER_CONSUMERS = ["phase_tau", "mtn_base_scale"]
NEURON_PARAMETER_CLAMP_MIN = 5e-4
NEURON_PARAMETER_CLAMP_MAX = 1e4
NEURON_PARAMETER_CLAMP_POLICY = "materialize_clamped_state"
MTN_BASE_SCALE_CALIBRATION = "ema_channel_abs_max_then_group_max_then_times_2"
MTN_BASE_SCALE_MULTIPLIER = 2.0
PHASE_TAU_REDUCTION_POLICY = "within_group_max_after_channel_ema"
