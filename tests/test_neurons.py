import torch

from snn2.neurons import MultiThresholdNeuron, PhaseSurrogate, StaticGIF


def test_phase_training_output_is_static():
    module = PhaseSurrogate(
        {
            "T": 4,
            "base": 2.0,
            "group_size": -1,
            "surrogate_slope": 4.0,
            "max_spikes": 2,
            "tau": torch.tensor([2.0]),
            "v0": torch.tensor([0.0625]),
        }
    )
    x = torch.randn(2, 5, 16, requires_grad=True)
    output = module(x)
    assert output.shape == x.shape
    output.sum().backward()
    assert x.grad is not None


def test_static_gif_unsigned_temporal_chunks_sum_to_fake_quant():
    state = {
        "base_bits": 4,
        "add_bits": 1,
        "group_size": -1,
        "low_scale": torch.tensor([0.1]),
        "low_zero": torch.tensor([7.0]),
        "high_scale": torch.tensor([0.05]),
        "high_zero": torch.tensor([13.0]),
        "mask_low": torch.tensor([True, False, True, False]),
    }
    module = StaticGIF(state)
    x = torch.tensor([[[[-0.7, 0.9, 0.5, -0.65]]]])
    temporal = module.temporal(x)
    torch.testing.assert_close(temporal.sum(dim=0), module(x.sum(dim=0)))


def test_static_gif_caps_high_q_to_two_base_bit_chunks():
    state = {
        "base_bits": 4,
        "add_bits": 1,
        "high_qmax": 30,
        "group_size": -1,
        "low_scale": torch.tensor([1.0]),
        "low_zero": torch.tensor([0.0]),
        "high_scale": torch.tensor([1.0]),
        "high_zero": torch.tensor([0.0]),
        "mask_low": torch.tensor([False]),
    }
    module = StaticGIF(state)

    temporal = module.temporal(torch.tensor([[[[31.0]]]]))

    assert temporal[:, 0, 0, 0].tolist() == [15.0, 15.0]
    torch.testing.assert_close(temporal.sum(dim=0), module(torch.tensor([[[31.0]]])))


def test_mtn_rejects_wrong_temporal_length():
    module = MultiThresholdNeuron(
        {
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
