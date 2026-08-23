from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from snn2.temporal_ops import (
    CALIBRATION_MANIFEST_FORMAT_VERSION,
    CONVERSION_METADATA_FORMAT_VERSION,
    SITE_STATE_FORMAT_VERSION,
    TEMPORAL_IMPLEMENTATION_VERSION,
    from_temporal,
    temporal_bias_once,
    temporal_rmsnorm,
    temporal_seq_matmul,
    temporal_silu,
    temporal_softmax,
    temporal_symmetric_hadamard,
    to_temporal,
)


@pytest.mark.parametrize("steps,batch", [(2, 1), (2, 3), (4, 1), (4, 3)])
@pytest.mark.parametrize("tail", [(5, 7), (2, 3, 5, 7)])
def test_temporal_layout_round_trip_is_time_major(steps, batch, tail):
    x = torch.arange(steps * batch * torch.tensor(tail).prod().item()).reshape(
        steps * batch, *tail
    )
    temporal = to_temporal(x, steps)
    torch.testing.assert_close(from_temporal(temporal), x)
    for timestep in range(steps):
        for sample in range(batch):
            torch.testing.assert_close(
                temporal[timestep, sample], x[timestep * batch + sample]
            )


@pytest.mark.parametrize("steps,batch", [(2, 1), (2, 2), (4, 1), (4, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_temporal_seq_matmul_reconstructs_every_prefix(steps, batch, dtype):
    torch.manual_seed(7)
    a = torch.randn(steps, batch, 3, 2, 4).to(dtype)
    b = torch.randn(steps, batch, 3, 4, 5).to(dtype)
    output = temporal_seq_matmul(a, b)
    tolerance = 3e-2 if dtype == torch.bfloat16 else 1e-5
    for timestep in range(steps):
        reference = torch.matmul(
            a[: timestep + 1].float().sum(0),
            b[: timestep + 1].float().sum(0),
        )
        torch.testing.assert_close(
            output[: timestep + 1].float().sum(0),
            reference,
            rtol=tolerance,
            atol=tolerance,
        )


class _RMSNorm(torch.nn.Module):
    def __init__(self, hidden: int, epsilon: float):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(0.5, 1.5, hidden))
        self.variance_epsilon = epsilon


@pytest.mark.parametrize("shape", [(4, 2, 3, 7), (4, 2, 3, 5, 7)])
@pytest.mark.parametrize("epsilon", [1e-6, 1e-5])
def test_temporal_rmsnorm_reconstructs_every_prefix(shape, epsilon):
    torch.manual_seed(11)
    x = torch.randn(shape)
    module = _RMSNorm(shape[-1], epsilon)
    output = temporal_rmsnorm(x, module)
    for timestep in range(shape[0]):
        cumulative = x[: timestep + 1].sum(0)
        reference = cumulative * torch.rsqrt(
            cumulative.square().mean(-1, keepdim=True) + epsilon
        )
        reference = reference * module.weight
        torch.testing.assert_close(
            output[: timestep + 1].sum(0), reference, rtol=1e-5, atol=1e-5
        )


def test_temporal_softmax_uses_one_fixed_mask_with_prefix_and_softcap():
    torch.manual_seed(13)
    steps, batch, heads, query, prefix, current = 4, 2, 2, 3, 2, 3
    scores = torch.randn(steps, batch, heads, query, prefix + current)
    mask = torch.zeros(batch, 1, query, prefix + current)
    causal = torch.triu(torch.ones(query, current, dtype=torch.bool), diagonal=1)
    mask[..., prefix:] = mask[..., prefix:].masked_fill(causal, float("-inf"))
    mask[1, ..., -1] = float("-inf")
    flattened_mask = mask.unsqueeze(0).expand(steps, -1, -1, -1, -1).reshape(
        steps * batch, 1, query, prefix + current
    )
    output = temporal_softmax(scores, flattened_mask, softcap=4.0)
    for timestep in range(steps):
        cumulative = scores[: timestep + 1].sum(0)
        capped = torch.tanh(cumulative / 4.0) * 4.0
        reference = F.softmax(capped + mask, dim=-1)
        reconstructed = output[: timestep + 1].sum(0)
        torch.testing.assert_close(reconstructed, reference, rtol=1e-5, atol=1e-6)
        assert torch.all(reconstructed.masked_select(torch.isneginf(mask)) == 0)


def test_temporal_softmax_rejects_different_masks_between_frames():
    scores = torch.randn(2, 1, 1, 2, 2)
    masks = torch.zeros(2, 1, 2, 2)
    masks[1, ..., -1] = float("-inf")
    with pytest.raises(ValueError, match="identical"):
        temporal_softmax(scores, masks)


def test_temporal_silu_reconstructs_every_prefix():
    x = torch.tensor(
        [[[[-2.0, 1.0]]], [[[3.0, -2.0]]], [[[-1.5, 4.0]]]]
    )
    output = temporal_silu(x)
    for timestep in range(x.shape[0]):
        torch.testing.assert_close(
            output[: timestep + 1].sum(0),
            F.silu(x[: timestep + 1].sum(0)),
        )


def test_temporal_symmetric_hadamard_matches_explicit_formula_and_total():
    torch.manual_seed(17)
    a = torch.randn(4, 2, 3, 5)
    b = torch.randn_like(a)
    output = temporal_symmetric_hadamard(a, b)
    explicit = []
    for timestep in range(a.shape[0]):
        value = a[timestep] * b[timestep]
        for other in range(a.shape[0]):
            if other != timestep:
                value = value + 0.5 * (
                    a[timestep] * b[other] + a[other] * b[timestep]
                )
        explicit.append(value)
    torch.testing.assert_close(output, torch.stack(explicit), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        output.sum(0), a.sum(0) * b.sum(0), rtol=1e-5, atol=1e-5
    )


def test_temporal_linear_bias_is_kept_only_at_timestep_zero():
    torch.manual_seed(19)
    steps, batch = 4, 3
    linear = torch.nn.Linear(5, 7, bias=True)
    x = torch.randn(steps, batch, 2, 5)
    repeated = linear(from_temporal(x))
    output = to_temporal(temporal_bias_once(repeated, linear.bias, steps), steps)
    reference = linear(x.sum(0))
    torch.testing.assert_close(output.sum(0), reference, rtol=1e-5, atol=1e-5)
    without_bias = F.linear(x, linear.weight, None)
    torch.testing.assert_close(output[0], without_bias[0] + linear.bias)
    torch.testing.assert_close(output[1:], without_bias[1:])


def test_artifact_schema_versions_do_not_change_temporal_arithmetic():
    assert CALIBRATION_MANIFEST_FORMAT_VERSION == 4
    assert CONVERSION_METADATA_FORMAT_VERSION == 5
    assert SITE_STATE_FORMAT_VERSION == 3
    assert TEMPORAL_IMPLEMENTATION_VERSION == 3
