from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import torch


@dataclass
class HadamardSpec:
    name: str
    dimension: int
    seed: int
    signs: torch.Tensor
    factor_k: int
    generator: str = "paley_or_sylvester"
    orientation: str = "DU"

    def __post_init__(self) -> None:
        if self.orientation != "DU":
            raise ValueError(
                f"Unsupported random Hadamard orientation {self.orientation!r}; "
                "current artifacts must use Q = D U"
            )

    def state_dict(self) -> dict[str, Any]:
        state = asdict(self)
        state["signs"] = self.signs.cpu()
        return state

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "HadamardSpec":
        return cls(**state)


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def choose_factor_k(n: int) -> int:
    if is_power_of_two(n):
        return 1
    for k in (12, 28):
        if n % k == 0 and is_power_of_two(n // k):
            return k
    raise ValueError(
        f"No supported exact Hadamard factorization for dimension={n}. "
        "Supported dimensions are powers of two, 12*2^m, and 28*2^m."
    )


def make_spec(name: str, dimension: int, seed: int) -> HadamardSpec:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    signs = torch.randint(0, 2, (dimension,), generator=generator, dtype=torch.int8)
    signs = signs.mul(2).sub(1)
    return HadamardSpec(name, dimension, seed, signs, choose_factor_k(dimension))


def _gf27_add(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple((x + y) % 3 for x, y in zip(a, b))  # type: ignore[return-value]


def _gf27_neg(a: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple((-x) % 3 for x in a)  # type: ignore[return-value]


def _gf27_mul(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[int, int, int]:
    # GF(3)[x] / (x^3 - x - 1), hence x^3 = x + 1.
    raw = [0] * 5
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            raw[i + j] = (raw[i + j] + ai * bj) % 3
    for degree in (4, 3):
        coefficient = raw[degree] % 3
        if coefficient:
            raw[degree] = 0
            raw[degree - 3] = (raw[degree - 3] + coefficient) % 3
            raw[degree - 2] = (raw[degree - 2] + coefficient) % 3
    return raw[0] % 3, raw[1] % 3, raw[2] % 3


@lru_cache(maxsize=2)
def paley_hadamard(order: int) -> torch.Tensor:
    if order == 12:
        q = 11
        elements: list[Any] = list(range(q))

        def sub(a: int, b: int) -> int:
            return (a - b) % q

        squares = {(i * i) % q for i in range(1, q)}
        zero: Any = 0
    elif order == 28:
        elements = [(a, b, c) for a in range(3) for b in range(3) for c in range(3)]
        zero = (0, 0, 0)

        def sub(a, b):
            return _gf27_add(a, _gf27_neg(b))

        squares = {_gf27_mul(value, value) for value in elements if value != zero}
    else:
        raise ValueError(f"Paley generator only supports orders 12 and 28, got {order}")

    q = order - 1

    def chi(value: Any) -> int:
        if value == zero:
            return 0
        return 1 if value in squares else -1

    core = torch.empty((q, q), dtype=torch.float64)
    for i, left in enumerate(elements):
        for j, right in enumerate(elements):
            core[i, j] = chi(sub(left, right)) - (1 if i == j else 0)
    had = torch.ones((order, order), dtype=torch.float64)
    had[1:, 1:] = core
    identity = torch.eye(order, dtype=torch.float64) * order
    if not torch.equal(had @ had.T, identity):
        raise RuntimeError(f"Generated H{order} failed orthogonality check")
    return had


def _pure_fht(x: torch.Tensor, normalized: bool = True) -> torch.Tensor:
    n = x.shape[-1]
    if not is_power_of_two(n):
        raise ValueError(f"FHT dimension must be a power of two, got {n}")
    y = x.contiguous().reshape(-1, n)
    width = 1
    while width < n:
        y = y.reshape(-1, n // (2 * width), 2, width)
        left = y[:, :, 0, :]
        right = y[:, :, 1, :]
        y = torch.cat((left + right, left - right), dim=-1)
        width *= 2
    y = y.reshape(x.shape)
    return y / math.sqrt(n) if normalized else y


def _fast_fht(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        return _pure_fht(x, normalized=True)

    try:
        from fast_hadamard_transform import hadamard_transform
    except ImportError as exc:
        raise RuntimeError(
            "CUDA Hadamard requires fast-hadamard-transform. "
            "Install the local package with: "
            "python -m pip install -e ./fast-hadamard-transform --no-build-isolation"
        ) from exc

    try:
        return hadamard_transform(
            x.contiguous(),
            scale=1.0 / math.sqrt(x.shape[-1]),
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "fast-hadamard-transform failed on CUDA; "
            "refusing to silently fall back to the PyTorch implementation."
        ) from exc


def structured_hadamard(x: torch.Tensor, factor_k: int, transpose: bool = False) -> torch.Tensor:
    n = x.shape[-1]
    if factor_k == 1:
        return _fast_fht(x)
    if n % factor_k != 0 or not is_power_of_two(n // factor_k):
        raise ValueError(f"Invalid Hadamard factorization n={n}, K={factor_k}")
    power = n // factor_k
    original_shape = x.shape
    y = x.reshape(-1, factor_k, power)
    y = _fast_fht(y)
    had_k = paley_hadamard(factor_k)
    if transpose:
        had_k = had_k.T
    y = torch.einsum("ij,bjk->bik", had_k.to(device=y.device, dtype=y.dtype), y)
    y = y / math.sqrt(factor_k)
    return y.reshape(original_shape)


def random_hadamard(x: torch.Tensor, spec: HadamardSpec, transpose: bool = False) -> torch.Tensor:
    if x.shape[-1] != spec.dimension:
        raise ValueError(
            f"{spec.name} expects last dimension {spec.dimension}, got {x.shape[-1]}"
        )
    signs = spec.signs.to(device=x.device, dtype=x.dtype)
    # Q = D U, where D = diag(signs) and U is the normalized structured
    # Hadamard transform. Row-vector xQ applies signs first; xQ^T applies U^T
    # first and signs last.
    if transpose:
        return structured_hadamard(x, spec.factor_k, transpose=True) * signs
    return structured_hadamard(x * signs, spec.factor_k, transpose=False)


def materialize_random_hadamard_matrix(
    spec: HadamardSpec,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Materialize Q = D U; intended for small/R1 FP64 dense transforms."""
    eye = torch.eye(spec.dimension, dtype=torch.float64, device="cpu")
    structured = structured_hadamard(eye, spec.factor_k, transpose=False)
    signs = spec.signs.to(device=structured.device, dtype=torch.float64)
    matrix = signs[:, None] * structured
    return matrix.to(device=device, dtype=dtype)


def transform_weight_right_fp64_dense(
    weight: torch.Tensor,
    matrix: torch.Tensor,
) -> torch.Tensor:
    """Compute weight @ Q in FP64 and cast back to the original placement."""
    original_device, original_dtype = weight.device, weight.dtype
    work = weight.to(device=matrix.device, dtype=torch.float64)
    transformed = torch.matmul(work, matrix.to(dtype=torch.float64))
    return transformed.to(device=original_device, dtype=original_dtype)


def transform_weight_left_transpose_fp64_dense(
    weight: torch.Tensor,
    matrix: torch.Tensor,
) -> torch.Tensor:
    """Compute Q^T @ weight in FP64 and cast back to original placement."""
    original_device, original_dtype = weight.device, weight.dtype
    work = weight.to(device=matrix.device, dtype=torch.float64)
    transformed = torch.matmul(matrix.to(dtype=torch.float64).T, work)
    return transformed.to(device=original_device, dtype=original_dtype)


def transform_weight_right_fp32_fht(
    weight: torch.Tensor,
    spec: HadamardSpec,
    device: torch.device | str,
) -> torch.Tensor:
    """Compute weight @ Q through the FP32 FHT path and cast back."""
    original_device, original_dtype = weight.device, weight.dtype
    work = weight.to(device=device, dtype=torch.float32)
    transformed = random_hadamard(work, spec)
    return transformed.to(device=original_device, dtype=original_dtype)

