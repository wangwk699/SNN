import torch

from snn2.calibration import build_mtn_state, build_phase_state
from snn2.phase_statistics import (
    PHASE_TAU_ACCUMULATOR_DTYPE,
    PHASE_TAU_CALIBRATION,
    PHASE_TAU_CHANNEL_POLICY,
    PHASE_TAU_EMA_FACTOR,
    PHASE_TAU_REDUCTION_POLICY,
    MTN_BASE_SCALE_CALIBRATION,
    NEURON_PARAMETER_CLAMP_MAX,
    NEURON_PARAMETER_CLAMP_MIN,
    NEURON_PARAMETER_CLAMP_POLICY,
    PARAMETER_ACCUMULATOR_DTYPE,
)
from snn2.stats import StatisticsStore
from snn2.temporal_ops import STATISTICS_FORMAT_VERSION


def _cfg(group_size=-1):
    return {
        "calibration": {"group_size": group_size},
        "phase": {"T": 4, "base": 2.0},
    }


def test_last_dim_statistics_use_fp32_ordered_ema():
    store = StatisticsStore()
    store.update(0, 1, torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]))
    store.update(0, 1, torch.tensor([[[2.0, 4.0, 6.0, 8.0]]]))
    state = next(iter(store.items.values())).state_dict()
    assert state["layout_kind"] == "last_dim"
    assert state["phase_ema_abs_max"].dtype == torch.float32
    assert torch.allclose(state["phase_ema_abs_max"], torch.tensor([1.01, 2.02, 3.03, 4.04]))


def test_attention_statistics_preserve_heads_and_do_not_cross_contaminate():
    store = StatisticsStore()
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]], [[10.0, 20.0, 30.0, 40.0]]]])
    store.update(0, 2, x)
    state = next(iter(store.items.values())).state_dict()
    assert state["layout_kind"] == "attention_head"
    assert state["num_heads"] == 2
    assert state["channels_per_head"] == 4
    assert state["phase_ema_abs_max"].shape == (2, 4)
    assert torch.equal(state["phase_ema_abs_max"], x[0, :, 0])


def test_attention_grouped_phase_tau_is_head_local():
    store = StatisticsStore()
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]], [[10.0, 20.0, 30.0, 40.0]]]])
    store.update(0, 2, x)
    statistics = next(iter(store.items.values())).state_dict()
    grouped = build_phase_state(statistics, _cfg(2))
    assert torch.equal(grouped["tau"], torch.tensor([[2.0, 4.0], [20.0, 40.0]]))
    assert grouped["parameter_layout"] == "attention_head_grouped"
    per_head = build_phase_state(statistics, _cfg(-1))
    assert torch.equal(per_head["tau"], torch.tensor([[4.0], [40.0]]))


def test_site5_statistics_are_per_head_and_allow_variable_qk():
    store = StatisticsStore()
    store.update(0, 5, torch.rand(1, 2, 3, 4))
    store.update(0, 5, torch.rand(1, 2, 1, 7))
    state = next(iter(store.items.values())).state_dict()
    assert state["layout_kind"] == "attention_softmax"
    assert state["value_min"].shape == (2,)
    assert state["phase_ema_abs_max"].shape == (2,)
    assert state["saliency_sum_by_role"] == {}
    phase = build_phase_state(state, _cfg(3))
    assert phase["tau"].shape == (2, 1)
    assert phase["group_size"] == -1


def test_statistics_schema_and_manifest_are_versioned(tmp_path):
    store = StatisticsStore()
    store.update(0, 1, torch.ones(1, 2, 4))
    store.update_global("final_rmsnorm", torch.ones(1, 2, 4))
    manifest = store.reduce_and_save(tmp_path)
    state = torch.load(next(tmp_path.glob("layer_*/site_*/statistics.pt")), weights_only=False)
    assert manifest["format_version"] == STATISTICS_FORMAT_VERSION
    assert state["format_version"] == STATISTICS_FORMAT_VERSION
    assert state["phase_tau_calibration"] == PHASE_TAU_CALIBRATION
    assert state["phase_tau_ema_factor"] == PHASE_TAU_EMA_FACTOR
    assert state["phase_tau_accumulator_dtype"] == PHASE_TAU_ACCUMULATOR_DTYPE
    assert state["phase_tau_channel_policy"] == PHASE_TAU_CHANNEL_POLICY
    assert state["phase_tau_reduction_policy"] == PHASE_TAU_REDUCTION_POLICY


def test_role_saliency_preserves_accumulator_precision():
    store = StatisticsStore()
    store.update(0, 1, torch.ones(1, 2, 4))
    store.update_saliency(0, 1, torch.ones(1, 2, 4, dtype=torch.float32), role="q", source="spikellm_linear_fp32")
    linear = next(iter(store.items.values()))
    assert linear.saliency_sum_by_role["q"].dtype == torch.float32

    store = StatisticsStore()
    store.update(0, 3, torch.ones(1, 2, 3, 4))
    store.update_saliency(0, 3, torch.ones(1, 2, 3, 4, dtype=torch.float64), source="spikellm_qk_k_fp64")
    matmul = next(iter(store.items.values()))
    assert matmul.saliency_sum_by_role["default"].dtype == torch.float64


def test_phase_and_mtn_materialize_shared_ema_with_post_multiplier_clamp():
    store = StatisticsStore()
    store.update(0, 1, torch.ones(1, 1, 3))
    statistics = next(iter(store.items.values())).state_dict()
    statistics["phase_ema_abs_max"] = torch.tensor([1e-4, 1.0, 2e4], dtype=torch.float32)
    statistics["phase_ema_updates"] = torch.ones(3, dtype=torch.long)
    # Deliberately disagree with extrema so MTN cannot silently use that legacy path.
    statistics["value_min"] = torch.tensor([-9e5, -9e5, -9e5])
    statistics["value_max"] = torch.tensor([9e5, 9e5, 9e5])
    phase = build_phase_state(statistics, _cfg(1))
    mtn = build_mtn_state(statistics, _cfg(1))
    torch.testing.assert_close(phase["tau"], torch.tensor([
        NEURON_PARAMETER_CLAMP_MIN, 1.0, NEURON_PARAMETER_CLAMP_MAX
    ]))
    torch.testing.assert_close(mtn["base_scale"], torch.tensor([
        NEURON_PARAMETER_CLAMP_MIN, 2.0, NEURON_PARAMETER_CLAMP_MAX
    ]))
    assert phase["tau_clamp_policy"] == NEURON_PARAMETER_CLAMP_POLICY
    assert mtn["base_scale_calibration"] == MTN_BASE_SCALE_CALIBRATION
    assert mtn["base_scale_accumulator_dtype"] == PARAMETER_ACCUMULATOR_DTYPE
    assert mtn["base_scale_clamp_policy"] == NEURON_PARAMETER_CLAMP_POLICY
