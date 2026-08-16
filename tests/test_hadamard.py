import torch

from snn2.hadamard import make_spec, paley_hadamard, random_hadamard


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
