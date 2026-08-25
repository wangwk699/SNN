import pytest
import torch

from snn2.neurons import (
    Clipper,
    HeavisideSigmoid,
    MultiThresholdNeuron,
    PhaseSurrogate,
    StaticGIF,
)
from snn2.temporal_ops import (
    GIF_INTEGER_DECOMPOSITION, PHASE_TAU_CALIBRATION, PHASE_TAU_EMA_FACTOR,
    SITE_STATE_FORMAT_VERSION, TEMPORAL_IMPLEMENTATION_VERSION,
)

def _header(kind):
    return {
        "state_kind": kind,
        "format_version": SITE_STATE_FORMAT_VERSION,
        "temporal_implementation_version": TEMPORAL_IMPLEMENTATION_VERSION,
    }


def _gif_policy():
    return {
        **_header("gif"),
        "base_bits": 4,
        "add_bits": 1,
        "low_qmin": 0,
        "low_qmax": 15,
        "high_qmin": 0,
        "high_qmax": 30,
        "temporal_steps": 2,
        "per_step_qmin": 0,
        "per_step_qmax": 15,
        "integer_decomposition": GIF_INTEGER_DECOMPOSITION,
    }


def _clip_state():
    return {
        **_header("clip"),
        "gif_high_qmax": 30,
        "gif_per_step_qmax": 15,
        "gif_integer_decomposition": GIF_INTEGER_DECOMPOSITION,
        "group_size": -1,
        "lower": torch.tensor([-0.75]),
        "upper": torch.tensor([0.75]),
        "gif_low_range": (torch.tensor([-1.0]), torch.tensor([1.0])),
        "gif_high_range": (torch.tensor([-1.0]), torch.tensor([1.0])),
    }


def test_phase_training_output_is_static():
    module = PhaseSurrogate(
        {
            **_header("phase"),
            "T": 4,
            "base": 2.0,
            "group_size": -1,
            "max_spikes": 2,
            "tau": torch.tensor([2.0]),
            "v0": torch.tensor([0.0625]),
            "tau_calibration": PHASE_TAU_CALIBRATION,
            "tau_ema_factor": PHASE_TAU_EMA_FACTOR,
            "tau_accumulator_dtype": "float32",
            "tau_channel_policy": "spikingllm_flatten_attention_heads_before_channel_ema",
            "tau_reduction_policy": "per_channel_ema_then_global_max",
            "phase_statistical_view": "spikingllm_identity_input_layout",
            "phase_statistical_view_version": 1,
        },
        surrogate_slope=1.0,
    )
    x = torch.randn(2, 5, 16, requires_grad=True)
    output = module(x)
    assert output.shape == x.shape
    output.sum().backward()
    assert x.grad is not None


def test_phase_surrogate_keeps_hard_forward_and_uses_unit_slope_backward():
    x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    output = HeavisideSigmoid.apply(x, 1.0)
    torch.testing.assert_close(output, torch.tensor([0.0, 0.0, 1.0]))
    output.sum().backward()
    expected = torch.sigmoid(x.detach()) * (1.0 - torch.sigmoid(x.detach()))
    torch.testing.assert_close(x.grad, expected)


def test_static_gif_unsigned_temporal_chunks_sum_to_fake_quant():
    state = {
        **_gif_policy(),
        "group_size": -1,
        "low_scale": torch.tensor([0.1]),
        "low_zero": torch.tensor([7.0]),
        "high_scale": torch.tensor([0.05]),
        "high_zero": torch.tensor([13.0]),
        "mask_low": torch.tensor([True, False, True, False]),
    }
    module = StaticGIF(state)
    x = torch.tensor([[[[-0.7, 0.9, 0.5, -0.65]]], [[[0.0, 0.0, 0.0, 0.0]]]])
    temporal = module.temporal(x)
    torch.testing.assert_close(temporal.sum(dim=0), module(x.sum(dim=0)))


def test_static_gif_caps_high_q_to_two_base_bit_chunks():
    state = {
        **_gif_policy(),
        "group_size": -1,
        "low_scale": torch.tensor([1.0]),
        "low_zero": torch.tensor([0.0]),
        "high_scale": torch.tensor([1.0]),
        "high_zero": torch.tensor([0.0]),
        "mask_low": torch.tensor([False]),
    }
    module = StaticGIF(state)

    temporal = module.temporal(torch.tensor([[[[31.0]]], [[[0.0]]]]))

    assert temporal[:, 0, 0, 0].tolist() == [15.0, 15.0]
    torch.testing.assert_close(temporal.sum(dim=0), module(torch.tensor([[[31.0]]])))


def test_mtn_rejects_wrong_temporal_length():
    module = MultiThresholdNeuron(
        {
            **_header("mtn"),
            "T": 4,
            "K": 6,
            "group_size": -1,
            "threshold_factor": 0.75,
            "base_scale": torch.tensor([2.0]),
        }
    )
    try:
        module.temporal(torch.zeros(3, 1, 2, 4))
    except ValueError as exc:
        assert "expects T=4" in str(exc)
    else:
        raise AssertionError("MTN must reject a mismatched temporal dimension")

@pytest.mark.parametrize("code", [0, 15, 16, 29, 30])
def test_gif_integer_boundaries_use_two_unsigned_chunks(code):
    chunk0, chunk1 = StaticGIF.integer_chunks(torch.tensor(float(code)))
    assert 0 <= chunk0.item() <= 15
    assert 0 <= chunk1.item() <= 15
    assert chunk0.item() + chunk1.item() == code
    if code == 30:
        assert [chunk0.item(), chunk1.item()] == [15, 15]


def test_gif_rejects_legacy_qmax_31_state():
    state = {
        **_gif_policy(),
        "high_qmax": 31,
        "group_size": -1,
        "low_scale": torch.tensor([1.0]),
        "low_zero": torch.tensor([0.0]),
        "high_scale": torch.tensor([1.0]),
        "high_zero": torch.tensor([0.0]),
        "mask_low": torch.tensor([False]),
    }
    with pytest.raises(ValueError, match="legacy GIF"):
        StaticGIF(state)



def test_clipper_remains_available_only_for_static_ann_tensors():
    clip = Clipper(_clip_state())
    output = clip(torch.tensor([[-2.0, 0.0, 2.0]]))
    torch.testing.assert_close(output, torch.tensor([[-0.75, 0.0, 0.75]]))
    assert not hasattr(clip, "temporal")


@pytest.mark.parametrize("kind", ["phase", "mtn", "clip"])
def test_non_gif_neurons_reject_format_v1(kind):
    if kind == "phase":
        state = {
            **_header("phase"), "T": 2, "base": 2.0, "group_size": -1,
            "tau": torch.tensor([1.0]),
            "v0": torch.tensor([0.125]),
            "tau_calibration": PHASE_TAU_CALIBRATION,
            "tau_ema_factor": PHASE_TAU_EMA_FACTOR,
            "tau_accumulator_dtype": "float32",
            "tau_channel_policy": "spikingllm_flatten_attention_heads_before_channel_ema",
            "tau_reduction_policy": "per_channel_ema_then_global_max",
            "phase_statistical_view": "spikingllm_identity_input_layout",
            "phase_statistical_view_version": 1,
        }
        factory = PhaseSurrogate
    elif kind == "mtn":
        state = {
            **_header("mtn"), "T": 2, "K": 2, "group_size": -1,
            "base_scale": torch.tensor([1.0]),
        }
        factory = MultiThresholdNeuron
    else:
        state = _clip_state()
        factory = Clipper
    state["format_version"] = 1
    with pytest.raises(ValueError, match="legacy"):
        factory(state)


def test_clip_rejects_missing_gif_range_metadata():
    state = _clip_state()
    del state["gif_high_range"]
    with pytest.raises(ValueError, match="gif_high_range"):
        Clipper(state)
