import pytest
import torch

from snn2.hadamard import (
    HadamardSpec,
    make_spec,
    materialize_random_hadamard_matrix,
    paley_hadamard,
    random_hadamard,
    structured_hadamard,
    transform_weight_left_transpose_fp64_dense,
    transform_weight_right_fp32_fht,
    transform_weight_right_fp64_dense,
)


def test_paley_orders_are_orthogonal():
    for order in (12, 28):
        matrix = paley_hadamard(order)
        expected = torch.eye(order, dtype=torch.float64) * order
        torch.testing.assert_close(matrix @ matrix.T, expected, rtol=0, atol=0)


def test_random_hadamard_round_trip_supported_dimensions():
    for dimension in (32, 12 * 8, 28 * 4):
        spec = make_spec("test", dimension, 42)
        x = torch.randn(3, dimension)
        reconstructed = random_hadamard(random_hadamard(x, spec), spec, transpose=True)
        torch.testing.assert_close(reconstructed, x, rtol=1e-5, atol=1e-5)


def test_random_hadamard_is_du_and_transpose_is_q_transpose():
    dimension = 8
    spec = make_spec("test", dimension, 42)
    identity = torch.eye(dimension, dtype=torch.float64)
    structured = structured_hadamard(identity, spec.factor_k)
    diagonal = torch.diag(spec.signs.to(torch.float64))
    expected_q = diagonal @ structured
    x = torch.randn(3, dimension, dtype=torch.float64)

    torch.testing.assert_close(random_hadamard(x, spec), x @ expected_q)
    torch.testing.assert_close(random_hadamard(x, spec, transpose=True), x @ expected_q.T)
    assert not torch.allclose(x @ expected_q, x @ (structured @ diagonal))


def test_materialized_random_hadamard_matrix_is_fp64_du():
    spec = make_spec("test", 8, 42)
    matrix = materialize_random_hadamard_matrix(spec, device="cpu")
    structured = structured_hadamard(torch.eye(8, dtype=torch.float64), spec.factor_k)
    expected = torch.diag(spec.signs.to(torch.float64)) @ structured

    assert matrix.dtype == torch.float64
    torch.testing.assert_close(matrix, expected)


def test_hadamard_spec_rejects_old_orientation():
    state = make_spec("test", 8, 42).state_dict()
    state["orientation"] = "UD"
    with pytest.raises(ValueError, match="Q = D U"):
        HadamardSpec.from_state_dict(state)


def test_fp64_dense_helpers_compute_in_fp64_and_cast_back(monkeypatch):
    spec = make_spec("test", 8, 42)
    matrix = materialize_random_hadamard_matrix(spec, device="cpu")
    weight = torch.randn(8, 8, dtype=torch.float32)
    calls = []
    original_matmul = torch.matmul

    def recorded_matmul(left, right):
        calls.append((left.dtype, right.dtype))
        return original_matmul(left, right)

    monkeypatch.setattr(torch, "matmul", recorded_matmul)
    right = transform_weight_right_fp64_dense(weight, matrix)
    left = transform_weight_left_transpose_fp64_dense(weight, matrix)

    assert calls == [(torch.float64, torch.float64), (torch.float64, torch.float64)]
    assert right.dtype == weight.dtype
    assert left.dtype == weight.dtype


def test_fp32_fht_helper_passes_fp32_to_hadamard(monkeypatch):
    seen = []
    spec = make_spec("test", 8, 42)

    def recorded(value, actual_spec, transpose=False):
        seen.append(value.dtype)
        return value

    monkeypatch.setattr("snn2.hadamard.random_hadamard", recorded)
    weight = torch.randn(2, 8, dtype=torch.float64)
    output = transform_weight_right_fp32_fht(weight, spec, "cpu")

    assert seen == [torch.float32]
    assert output.dtype == torch.float64
