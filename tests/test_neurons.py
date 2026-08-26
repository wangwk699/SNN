import pytest
import torch

from snn2.neurons import (
    Clipper,
    MultiThresholdNeuron,
    PhaseSurrogate,
    SoftmaxFixedGIF,
    StaticGIF,
    gif_module_from_state,
)
from snn2.phase_statistics import (
    PHASE_TAU_ACCUMULATOR_DTYPE,
    PHASE_TAU_CALIBRATION,
    PHASE_TAU_CHANNEL_POLICY,
    PHASE_TAU_EMA_FACTOR,
    PHASE_TAU_REDUCTION_POLICY,
)
from snn2.temporal_ops import (
    GIF_INTEGER_DECOMPOSITION,
    SITE_STATE_FORMAT_VERSION,
    TEMPORAL_IMPLEMENTATION_VERSION,
)


def _header(kind):
    return {
        "state_kind": kind,
        "format_version": SITE_STATE_FORMAT_VERSION,
        "temporal_implementation_version": TEMPORAL_IMPLEMENTATION_VERSION,
    }


def _layout(kind="last_dim_grouped"):
    if kind == "last_dim_grouped":
        return dict(parameter_layout=kind, configured_group_size=2, group_size=2, num_heads=None, channels_per_head=4, groups_per_head=2)
    if kind == "attention_head_grouped":
        return dict(parameter_layout=kind, configured_group_size=2, group_size=2, num_heads=2, channels_per_head=4, groups_per_head=2)
    return dict(parameter_layout="attention_head_scalar", configured_group_size=-1, group_size=-1, num_heads=2, channels_per_head=None, groups_per_head=1)


def _phase_state(kind="last_dim_grouped"):
    layout = _layout(kind)
    shape = (2,) if kind == "last_dim_grouped" else (2, 2) if kind == "attention_head_grouped" else (2, 1)
    tau = torch.full(shape, 2.0)
    return {
        **_header("phase"), **layout, "T": 4, "base": 2.0, "max_spikes": 2,
        "tau": tau, "v0": tau / 32,
        "tau_calibration": PHASE_TAU_CALIBRATION,
        "tau_ema_factor": PHASE_TAU_EMA_FACTOR,
        "tau_accumulator_dtype": PHASE_TAU_ACCUMULATOR_DTYPE,
        "tau_channel_policy": PHASE_TAU_CHANNEL_POLICY,
        "tau_reduction_policy": PHASE_TAU_REDUCTION_POLICY,
    }


def _gif_state(kind="last_dim_grouped"):
    layout = _layout(kind)
    shape = (2,) if kind == "last_dim_grouped" else (2, 2)
    mask_shape = (4,) if kind == "last_dim_grouped" else (2, 4)
    return {
        **_header("gif"), **layout, "gif_policy": "ordinary_grouped_qmax30",
        "base_bits": 4, "add_bits": 1, "low_qmin": 0, "low_qmax": 15,
        "high_qmin": 0, "high_qmax": 30, "temporal_steps": 2,
        "per_step_qmin": 0, "per_step_qmax": 15,
        "integer_decomposition": GIF_INTEGER_DECOMPOSITION,
        "low_scale": torch.full(shape, 0.1), "low_zero": torch.zeros(shape),
        "high_scale": torch.full(shape, 0.05), "high_zero": torch.zeros(shape),
        "mask_low": torch.zeros(mask_shape, dtype=torch.bool),
    }


def _softmax_gif_state():
    return {
        **_header("gif"), "parameter_layout": "softmax_fixed_range",
        "configured_group_size": 32, "group_size": -1,
        "group_size_source": "site5_fixed_override", "num_heads": 2,
        "channels_per_head": None, "groups_per_head": 1,
        "gif_policy": "softmax_fixed_range_u16", "range_min": 0.0,
        "range_max": 1.0, "quantization_bits": 16, "qmin": 0,
        "qmax": 65535, "scale": 1.0 / 65535.0, "zero_point": 0,
        "temporal_steps": 2, "temporal_policy": "quantized_cumulative_difference",
    }


@pytest.mark.parametrize("kind,shape", [("last_dim_grouped", (2, 3, 4)), ("attention_head_grouped", (1, 2, 3, 4)), ("attention_head_scalar", (1, 2, 3, 7))])
def test_phase_supports_all_grouped_layouts(kind, shape):
    module = PhaseSurrogate(_phase_state(kind), surrogate_slope=1.0)
    x = torch.randn(*shape)
    assert module(x).shape == x.shape


def test_phase_hard_forward_and_temporal_sum_match():
    module = PhaseSurrogate(_phase_state())
    x = torch.randn(2, 3, 4)
    incoming = x.unsqueeze(0).expand(4, *x.shape) / 4
    assert torch.equal(module(x), module.temporal(incoming).sum(0))


def test_phase_rejects_wrong_head_or_parameter_shape():
    module = PhaseSurrogate(_phase_state("attention_head_grouped"))
    with pytest.raises(ValueError, match="runtime shape"):
        module(torch.randn(1, 3, 2, 4))
    state = _phase_state("attention_head_grouped")
    state["tau"] = torch.ones(2, 1)
    with pytest.raises(ValueError, match="tau shape"):
        PhaseSurrogate(state)


@pytest.mark.parametrize("kind,shape", [("last_dim_grouped", (2, 3, 4)), ("attention_head_grouped", (1, 2, 3, 4))])
def test_ordinary_gif_grouped_forward_and_temporal(kind, shape):
    module = StaticGIF(_gif_state(kind))
    x = torch.rand(*shape)
    incoming = torch.stack((x, torch.zeros_like(x)))
    assert torch.allclose(module(x), module.temporal(incoming).sum(0))


def test_gif_mask_shape_is_strict_without_padding_or_truncation():
    state = _gif_state()
    state["mask_low"] = torch.zeros(3, dtype=torch.bool)
    with pytest.raises(ValueError, match="mask_low shape"):
        StaticGIF(state)


def test_softmax_fixed_gif_is_exact_q16_and_factory_selects_it():
    module = gif_module_from_state(_softmax_gif_state())
    assert isinstance(module, SoftmaxFixedGIF)
    x = torch.tensor([[[[0.0, 0.1, 0.5, 1.0]], [[0.2, 0.3, 0.4, 0.1]]]])
    expected = torch.round(x * 65535) / 65535
    assert torch.equal(module(x), expected)


def test_softmax_fixed_gif_temporal_is_quantized_cumulative_difference():
    module = SoftmaxFixedGIF(_softmax_gif_state())
    incoming = torch.rand(2, 1, 2, 3, 5) / 2
    output = module.temporal(incoming)
    expected = torch.round(incoming.sum(0).clamp(0, 1) * 65535) / 65535
    assert output.shape == incoming.shape
    assert torch.allclose(output.sum(0), expected)


def test_mtn_and_clip_support_attention_grouped_parameters():
    layout = _layout("attention_head_grouped")
    mtn = MultiThresholdNeuron({**_header("mtn"), **layout, "T": 2, "K": 2, "threshold_factor": 0.75, "base_scale": torch.ones(2, 2)})
    incoming = torch.rand(2, 1, 2, 3, 4)
    assert mtn.temporal(incoming).shape == incoming.shape
    clip = Clipper({
        **_header("clip"), **layout, "gif_high_qmax": 30,
        "gif_per_step_qmax": 15, "gif_integer_decomposition": GIF_INTEGER_DECOMPOSITION,
        "lower": -torch.ones(2, 2), "upper": torch.ones(2, 2),
        "gif_low_range": (-torch.ones(2, 2), torch.ones(2, 2)),
        "gif_high_range": (-torch.ones(2, 2), torch.ones(2, 2)),
    })
    x = torch.full((1, 2, 3, 4), 2.0)
    assert torch.equal(clip(x), torch.ones_like(x))


def test_legacy_state_versions_and_metadata_are_rejected():
    state = _phase_state()
    state["format_version"] -= 1
    with pytest.raises(ValueError, match="legacy"):
        PhaseSurrogate(state)
    state = _phase_state()
    state["surrogate_slope"] = 1.0
    with pytest.raises(ValueError, match="surrogate_slope"):
        PhaseSurrogate(state)
