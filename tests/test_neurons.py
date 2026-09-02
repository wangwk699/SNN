import pytest
import torch

from snn2.neurons import (
    Clipper,
    MultiThresholdNeuron,
    PhaseSurrogate,
    SoftmaxIdentityGIF,
    StaticGIF,
    _mask_values,
    gif_module_from_state,
)
from snn2.phase_statistics import (
    PHASE_TAU_ACCUMULATOR_DTYPE,
    MTN_BASE_SCALE_CALIBRATION,
    MTN_BASE_SCALE_MULTIPLIER,
    NEURON_PARAMETER_CLAMP_MAX,
    NEURON_PARAMETER_CLAMP_MIN,
    NEURON_PARAMETER_CLAMP_POLICY,
    PARAMETER_ACCUMULATOR_DTYPE,
    PARAMETER_CHANNEL_POLICY,
    PHASE_TAU_CALIBRATION,
    PHASE_TAU_CHANNEL_POLICY,
    PHASE_TAU_EMA_FACTOR,
    PHASE_TAU_REDUCTION_POLICY,
)
from snn2.temporal_ops import (
    GIF_INTEGER_DECOMPOSITION,
    GIF_LOW_QMAX,
    SITE_STATE_FORMAT_VERSION,
    SOFTMAX_SITE5_GIF_POLICY,
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
        **_header("phase"), **layout,
        "tau": tau,
        "tau_calibration": PHASE_TAU_CALIBRATION,
        "tau_ema_factor": PHASE_TAU_EMA_FACTOR,
        "tau_accumulator_dtype": PHASE_TAU_ACCUMULATOR_DTYPE,
        "tau_channel_policy": PHASE_TAU_CHANNEL_POLICY,
        "tau_reduction_policy": PHASE_TAU_REDUCTION_POLICY,
        "tau_clamp_min": NEURON_PARAMETER_CLAMP_MIN,
        "tau_clamp_max": NEURON_PARAMETER_CLAMP_MAX,
        "tau_clamp_policy": NEURON_PARAMETER_CLAMP_POLICY,
    }


def _gif_state(kind="last_dim_grouped"):
    layout = _layout(kind)
    shape = (2,) if kind == "last_dim_grouped" else (2, 2)
    mask_shape = (4,) if kind == "last_dim_grouped" else (2, 4)
    return {
        **_header("gif"), **layout, "gif_policy": "ordinary_salient_static_qmax30",
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
        **_header("gif"), "parameter_layout": "softmax_identity",
        "configured_group_size": 32, "group_size": -1,
        "group_size_source": "site5_identity_override", "num_heads": 2,
        "channels_per_head": None, "groups_per_head": 1,
        "gif_policy": SOFTMAX_SITE5_GIF_POLICY,
        "reference_n_bits": 16, "reference_metric": "fix0to1",
        "quantization_applied": False, "temporal_steps": 2,
        "temporal_policy": "identity",
    }


@pytest.mark.parametrize("kind,shape", [("last_dim_grouped", (2, 3, 4)), ("attention_head_grouped", (1, 2, 3, 4)), ("attention_head_scalar", (1, 2, 3, 7))])
def test_phase_supports_all_grouped_layouts(kind, shape):
    module = PhaseSurrogate(_phase_state(kind), T=4, surrogate_slope=1.0)
    x = torch.randn(*shape)
    assert module(x).shape == x.shape


def test_phase_hard_forward_and_temporal_sum_match():
    module = PhaseSurrogate(_phase_state(), T=4)
    x = torch.randn(2, 3, 4)
    incoming = x.unsqueeze(0).expand(4, *x.shape) / 4
    assert torch.equal(module(x), module.temporal(incoming).sum(0))


def test_phase_rejects_wrong_head_or_parameter_shape():
    module = PhaseSurrogate(_phase_state("attention_head_grouped"), T=4)
    with pytest.raises(ValueError, match="runtime shape"):
        module(torch.randn(1, 3, 2, 4))
    state = _phase_state("attention_head_grouped")
    state["tau"] = torch.ones(2, 1)
    with pytest.raises(ValueError, match="tau shape"):
        PhaseSurrogate(state, T=4)


@pytest.mark.parametrize("kind,shape", [("last_dim_grouped", (2, 3, 4)), ("attention_head_grouped", (1, 2, 3, 4))])
def test_ordinary_gif_grouped_forward_and_temporal(kind, shape):
    module = StaticGIF(_gif_state(kind))
    x = torch.rand(*shape)
    incoming = torch.stack((x, torch.zeros_like(x)))
    assert torch.allclose(module(x), module.temporal(incoming).sum(0))


def test_gif_mask_shape_is_strict_without_padding_or_truncation():
    state = _gif_state()
    state["mask_low"] = torch.zeros(3, dtype=torch.bool)
    with pytest.raises(ValueError, match="masks must have shape"):
        StaticGIF(state)


def test_softmax_gif_is_exact_identity_and_factory_selects_it():
    module = gif_module_from_state(_softmax_gif_state())
    assert isinstance(module, SoftmaxIdentityGIF)
    x = torch.tensor([[[[-0.1, 0.1, 0.5, 1.1]], [[0.2, 0.3, 0.4, 0.1]]]])
    output = module(x)
    assert output is x
    assert torch.equal(output, x)


def test_softmax_gif_temporal_is_exact_identity():
    module = SoftmaxIdentityGIF(_softmax_gif_state())
    incoming = torch.randn(2, 1, 2, 3, 5)
    output = module.temporal(incoming)
    assert output is incoming
    assert torch.equal(output, incoming)


def test_softmax_gif_rejects_legacy_q16_state():
    state = _softmax_gif_state()
    state.update(
        {
            "parameter_layout": "softmax_fixed_range",
            "gif_policy": "softmax_fixed_range_u16",
            "temporal_policy": "quantized_cumulative_difference",
        }
    )
    with pytest.raises(ValueError, match="identity state"):
        SoftmaxIdentityGIF(state)

    state = _softmax_gif_state()
    state["qmax"] = 65535
    with pytest.raises(ValueError, match="forbidden"):
        SoftmaxIdentityGIF(state)


def test_mtn_and_clip_support_attention_grouped_parameters():
    layout = _layout("attention_head_grouped")
    mtn = MultiThresholdNeuron({**_header("mtn"), **layout, "base_scale": torch.ones(2, 2),
        "base_scale_calibration": MTN_BASE_SCALE_CALIBRATION,
        "base_scale_ema_factor": PHASE_TAU_EMA_FACTOR,
        "base_scale_accumulator_dtype": PARAMETER_ACCUMULATOR_DTYPE,
        "base_scale_channel_policy": PARAMETER_CHANNEL_POLICY,
        "base_scale_reduction_policy": PHASE_TAU_REDUCTION_POLICY,
        "base_scale_multiplier": MTN_BASE_SCALE_MULTIPLIER,
        "base_scale_clamp_min": NEURON_PARAMETER_CLAMP_MIN,
        "base_scale_clamp_max": NEURON_PARAMETER_CLAMP_MAX,
        "base_scale_clamp_policy": NEURON_PARAMETER_CLAMP_POLICY,
    }, T=2, K=2, threshold_factor=0.75)
    incoming = torch.rand(2, 1, 2, 3, 4)
    assert mtn.temporal(incoming).shape == incoming.shape
    clip = Clipper({
        **_header("clip"), **layout, "ordinary_gif_high_qmax": 30,
        "ordinary_gif_per_step_qmax": 15, "gif_integer_decomposition": GIF_INTEGER_DECOMPOSITION,
        "clip_role_policy": "single", "lower": -torch.ones(2, 2), "upper": torch.ones(2, 2),
        "gif_low_range": (-torch.ones(2, 2), torch.ones(2, 2)),
        "gif_high_range": (-torch.ones(2, 2), torch.ones(2, 2)),
    })
    x = torch.full((1, 2, 3, 4), 2.0)
    assert torch.equal(clip(x), torch.ones_like(x))


def test_legacy_state_versions_and_metadata_are_rejected():
    state = _phase_state()
    state["format_version"] -= 1
    with pytest.raises(ValueError, match="legacy"):
        PhaseSurrogate(state, T=4)
    state = _phase_state()
    state["surrogate_slope"] = 1.0
    with pytest.raises(ValueError, match="surrogate_slope"):
        PhaseSurrogate(state, T=4)


def test_attention_head_grouped_parameters_support_merged_runtime():
    module = PhaseSurrogate(_phase_state("attention_head_grouped"), T=4)
    x = torch.randn(2, 3, 8)
    assert module(x).shape == x.shape


def test_multi_role_gif_selects_role_and_fails_fast():
    state = _gif_state()
    state.pop("mask_low")
    state.update({
        "mask_policy": "multi_role",
        "mask_roles": ["q", "k", "v"],
        "mask_low_by_role": {
            "q": torch.ones(4, dtype=torch.bool),
            "k": torch.zeros(4, dtype=torch.bool),
            "v": torch.tensor([True, False, True, False]),
        },
    })
    module = StaticGIF(state)
    x = torch.tensor([[[0.17, 0.17, 0.17, 0.17]]])
    assert not torch.equal(module(x, role="q"), module(x, role="k"))
    with pytest.raises(ValueError, match="role must be"):
        module(x)
    with pytest.raises(ValueError, match="role must be"):
        module(x, role="invalid")


def test_all_low_and_identity_gif_temporal_policies():
    all_low = {
        **_header("gif"), **_layout(),
        "gif_policy": "all_low_static_qmax15",
        "base_bits": 4, "add_bits": 1,
        "low_qmin": 0, "low_qmax": 15, "temporal_steps": 2,
        "per_step_qmin": 0, "per_step_qmax": 15,
        "quantization_path": "low_only", "quantization_applied": True,
        "saliency_enabled": False, "temporal_policy": "low_at_t0_zero_at_t1",
        "low_scale": torch.full((2,), 0.1),
        "low_zero": torch.zeros(2),
    }
    module = gif_module_from_state(all_low)
    incoming = torch.randn(2, 1, 3, 4)
    temporal = module.temporal(incoming)
    assert torch.count_nonzero(temporal[1]) == 0
    torch.testing.assert_close(temporal.sum(0), module(incoming.sum(0)))

    identity = {
        **_header("gif"), "gif_policy": "identity",
        "quantization_applied": False, "temporal_steps": 2,
    }
    module = gif_module_from_state(identity)
    x = torch.randn(2, 1, 3, 4)
    assert module.temporal(x) is x
    frame = x[0]
    assert module(frame) is frame
    assert torch.equal(module(frame), frame)


def _all_low_state():
    return {
        **_header("gif"), **_layout("attention_head_grouped"),
        "gif_policy": "all_low_static_qmax15",
        "base_bits": 4, "add_bits": 1,
        "low_qmin": 0, "low_qmax": 15, "temporal_steps": 2,
        "per_step_qmin": 0, "per_step_qmax": 15,
        "quantization_path": "low_only", "quantization_applied": True,
        "saliency_enabled": False, "temporal_policy": "low_at_t0_zero_at_t1",
        "low_scale": torch.full((2, 2), 0.1),
        "low_zero": torch.zeros(2, 2),
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("low_qmax", 14),
        ("temporal_steps", 4),
        ("per_step_qmax", 14),
        ("low_qmin", 1),
        ("per_step_qmin", 1),
        ("base_bits", 3),
        ("add_bits", 2),
        ("quantization_path", "low_high"),
        ("quantization_applied", False),
        ("saliency_enabled", True),
        ("temporal_policy", "ordinary"),
    ],
)
def test_all_low_gif_rejects_corrupted_policy(field, value):
    state = _all_low_state()
    state[field] = value
    with pytest.raises(ValueError, match="Invalid all-low GIF state"):
        gif_module_from_state(state)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("high_scale", torch.ones(2, 2)),
        ("high_zero", torch.zeros(2, 2)),
        ("high_qmax", 30),
        ("integer_decomposition", GIF_INTEGER_DECOMPOSITION),
        ("mask_low", torch.ones(2, 4, dtype=torch.bool)),
        ("mask_low_by_role", {"default": torch.ones(2, 4, dtype=torch.bool)}),
        ("mask_roles", ["default"]),
        ("saliency_score", torch.ones(2, 4)),
    ],
)
def test_all_low_gif_rejects_high_or_mask_fields(field, value):
    state = _all_low_state()
    state[field] = value
    with pytest.raises(ValueError, match="Invalid all-low GIF state"):
        gif_module_from_state(state)


def _legacy_static_gif_forward(module, x, *, role=None):
    low, _, _ = module._quantize(
        x, module.low_scale, module.low_zero, qmin=0, qmax=GIF_LOW_QMAX
    )
    high, _, _ = module._quantize(
        x, module.high_scale, module.high_zero, qmin=0, qmax=module.high_qmax
    )
    return torch.where(_mask_values(x, module._mask(role), module.layout), low, high)


@pytest.mark.parametrize(
    ("kind", "shape"),
    [
        ("last_dim_grouped", (2, 3, 4)),
        ("attention_head_grouped", (1, 2, 3, 4)),
        ("attention_head_grouped", (2, 3, 8)),
    ],
)
def test_phase_ann_streaming_matches_legacy_forward_and_input_gradient(kind, shape):
    torch.manual_seed(17)
    module = PhaseSurrogate(_phase_state(kind), T=4, surrogate_slope=1.0)
    x = torch.randn(*shape)
    x_reference = x.detach().clone().requires_grad_(True)
    x_optimized = x.detach().clone().requires_grad_(True)
    reference = module.encode(x_reference, return_temporal=False)
    optimized = module(x_optimized)
    torch.testing.assert_close(optimized, reference, rtol=1e-6, atol=1e-7)
    grad = torch.randn_like(reference)
    reference.backward(grad)
    optimized.backward(grad)
    torch.testing.assert_close(x_optimized.grad, x_reference.grad, rtol=1e-6, atol=1e-7)


def test_phase_temporal_still_matches_legacy_encode_reference():
    torch.manual_seed(23)
    module = PhaseSurrogate(_phase_state("attention_head_grouped"), T=4)
    incoming = torch.randn(4, 1, 2, 3, 4)
    assert torch.equal(
        module.temporal(incoming),
        module.encode(incoming.sum(dim=0), return_temporal=True),
    )


@pytest.mark.parametrize(
    ("kind", "shape"),
    [
        ("last_dim_grouped", (2, 3, 4)),
        ("attention_head_grouped", (1, 2, 3, 4)),
        ("attention_head_grouped", (2, 3, 8)),
    ],
)
def test_static_gif_ann_mixed_quant_matches_legacy_forward_and_input_gradient(kind, shape):
    torch.manual_seed(29)
    state = _gif_state(kind)
    if kind == "last_dim_grouped":
        state["mask_low"] = torch.tensor([True, False, True, False])
    else:
        state["mask_low"] = torch.tensor(
            [[True, False, True, False], [False, True, False, True]]
        )
    module = StaticGIF(state)
    x = torch.tensor([-2.1, -0.04, 0.14, 0.76, 2.4], dtype=torch.float32)
    x = x.repeat((int(torch.tensor(shape).prod().item()) + x.numel() - 1) // x.numel())
    x = x[: int(torch.tensor(shape).prod().item())].reshape(shape)
    x_reference = x.detach().clone().requires_grad_(True)
    x_optimized = x.detach().clone().requires_grad_(True)
    reference = _legacy_static_gif_forward(module, x_reference)
    optimized = module(x_optimized)
    torch.testing.assert_close(optimized, reference, rtol=1e-6, atol=1e-7)
    grad = torch.randn_like(reference)
    reference.backward(grad)
    optimized.backward(grad)
    torch.testing.assert_close(x_optimized.grad, x_reference.grad, rtol=1e-6, atol=1e-7)


def test_static_gif_ann_mixed_quant_matches_legacy_multi_role_masks():
    state = _gif_state()
    state.pop("mask_low")
    state.update(
        {
            "mask_policy": "multi_role",
            "mask_roles": ["q", "k"],
            "mask_low_by_role": {
                "q": torch.tensor([True, False, True, False]),
                "k": torch.tensor([False, True, False, True]),
            },
        }
    )
    module = StaticGIF(state)
    x = torch.tensor([[[-0.2, 0.13, 0.84, 2.2]]])
    for role in ("q", "k"):
        torch.testing.assert_close(
            module(x, role=role), _legacy_static_gif_forward(module, x, role=role)
        )


def test_static_gif_temporal_still_matches_legacy_reference():
    torch.manual_seed(31)
    module = StaticGIF(_gif_state())
    incoming = torch.randn(2, 2, 3, 4)
    x = incoming.sum(dim=0)
    _, low_q, low_zero = module._quantize(
        x, module.low_scale, module.low_zero, qmin=0, qmax=GIF_LOW_QMAX
    )
    _, high_q, high_zero = module._quantize(
        x, module.high_scale, module.high_zero, qmin=0, qmax=module.high_qmax
    )
    mask = _mask_values(x, module._mask(None), module.layout)
    scale_low = module.low_scale.repeat_interleave(2).view(1, 1, 4)
    scale_high = module.high_scale.repeat_interleave(2).view(1, 1, 4)
    outputs = []
    for timestep, chunk in enumerate(module.integer_chunks(high_q)):
        high_output = chunk * scale_high
        if timestep == 0:
            high_output = high_output - high_zero * scale_high
            low_output = (low_q - low_zero) * scale_low
        else:
            low_output = torch.zeros_like(x, dtype=torch.float32)
        outputs.append(torch.where(mask, low_output, high_output))
    reference = torch.stack(outputs, dim=0).to(x.dtype)
    torch.testing.assert_close(module.temporal(incoming), reference, rtol=1e-6, atol=1e-7)


def test_static_gif_ann_mixed_quant_matches_legacy_clamp_boundary_gradients():
    state = _gif_state()
    state["mask_low"] = torch.tensor([True, False, True, False])
    module = StaticGIF(state)
    x = torch.tensor(
        [
            [[-0.10, -0.05, 0.00, 0.00]],
            [[0.70, 0.75, 1.50, 1.50]],
            [[1.60, 1.55, 1.50, 1.50]],
        ],
        dtype=torch.float32,
    )
    x_reference = x.detach().clone().requires_grad_(True)
    x_optimized = x.detach().clone().requires_grad_(True)
    reference = _legacy_static_gif_forward(module, x_reference)
    optimized = module(x_optimized)
    torch.testing.assert_close(optimized, reference, rtol=0, atol=0)
    grad = torch.tensor(
        [
            [[0.5, -1.0, 2.0, -0.5]],
            [[1.5, 0.25, -0.75, 2.0]],
            [[-2.0, 1.25, 0.75, -1.5]],
        ],
        dtype=torch.float32,
    )
    reference.backward(grad)
    optimized.backward(grad)
    torch.testing.assert_close(x_optimized.grad, x_reference.grad, rtol=0, atol=0)
