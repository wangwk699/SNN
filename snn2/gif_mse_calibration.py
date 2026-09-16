from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

import torch


MSE_REFINEMENT_DEFAULTS: dict[str, Any] = {
    "version": "static_mse_v1",
    "objective": "elementwise_mse",
    "histogram_bins": 4096,
    "coarse_alpha_values": [
        1.0, 0.995, 0.99, 0.98, 0.97, 0.95, 0.925,
        0.90, 0.875, 0.85, 0.80, 0.75, 0.70,
    ],
    "fine_alpha_radius": 0.03,
    "fine_alpha_step": 0.005,
    "fine_alpha_min": 0.70,
    "fine_alpha_max": 1.0,
    "local_scale_ratio_min": 0.95,
    "local_scale_ratio_max": 1.05,
    "local_scale_ratio_step": 0.01,
    "local_zero_radius": 2,
    "preserve_zero": True,
    "fallback_to_direct_min_max": True,
}


def normalize_mse_refinement_config(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is not None and not isinstance(value, dict):
        raise ValueError("gif.mse_refinement must be a mapping")
    result = deepcopy(MSE_REFINEMENT_DEFAULTS)
    result.update(deepcopy(value or {}))
    return result


def mse_refinement_signature(value: dict[str, Any]) -> str:
    normalized = normalize_mse_refinement_config(value)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_mse_refinement_config(value: dict[str, Any]) -> None:
    cfg = normalize_mse_refinement_config(value)
    if cfg["version"] != "static_mse_v1":
        raise ValueError("gif.mse_refinement.version must be static_mse_v1")
    if cfg["objective"] != "elementwise_mse":
        raise ValueError("gif.mse_refinement.objective must be elementwise_mse")
    bins = cfg["histogram_bins"]
    if not isinstance(bins, int) or isinstance(bins, bool) or bins <= 0:
        raise ValueError("gif.mse_refinement.histogram_bins must be a positive integer")
    alphas = cfg["coarse_alpha_values"]
    if not isinstance(alphas, list) or not alphas:
        raise ValueError("gif.mse_refinement.coarse_alpha_values must be a non-empty list")
    numeric_alphas = []
    for alpha in alphas:
        try:
            alpha = float(alpha)
        except (TypeError, ValueError) as exc:
            raise ValueError("coarse alpha values must be finite numbers") from exc
        if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ValueError("coarse alpha values must satisfy 0 < alpha <= 1")
        numeric_alphas.append(alpha)
    if len(set(numeric_alphas)) != len(numeric_alphas):
        raise ValueError("gif.mse_refinement.coarse_alpha_values must not contain duplicates")
    for key in ("fine_alpha_radius", "fine_alpha_step", "local_scale_ratio_step"):
        number = float(cfg[key])
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"gif.mse_refinement.{key} must be positive and finite")
    fine_min, fine_max = float(cfg["fine_alpha_min"]), float(cfg["fine_alpha_max"])
    if not (math.isfinite(fine_min) and math.isfinite(fine_max) and 0.0 < fine_min <= fine_max <= 1.0):
        raise ValueError("gif.mse_refinement fine alpha bounds must satisfy 0 < min <= max <= 1")
    ratio_min, ratio_max = float(cfg["local_scale_ratio_min"]), float(cfg["local_scale_ratio_max"])
    if not (math.isfinite(ratio_min) and math.isfinite(ratio_max) and 0.0 < ratio_min <= ratio_max):
        raise ValueError("gif.mse_refinement scale ratio bounds must satisfy 0 < min <= max")
    radius = cfg["local_zero_radius"]
    if not isinstance(radius, int) or isinstance(radius, bool) or radius < 0:
        raise ValueError("gif.mse_refinement.local_zero_radius must be a non-negative integer")
    for key in ("preserve_zero", "fallback_to_direct_min_max"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"gif.mse_refinement.{key} must be true or false")


def runtime_fake_quant(values: torch.Tensor, scale: float, zero: int, *, qmin: int, qmax: int) -> torch.Tensor:
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError("scale must be positive and finite")
    if int(zero) != zero or not qmin <= int(zero) <= qmax:
        raise ValueError("zero must be integer-valued and inside the quantization range")
    work = values.to(torch.float64)
    return float(scale) * (torch.round(work / float(scale)).clamp(qmin - int(zero), qmax - int(zero)))


def evaluate_histogram_mse(
    counts: torch.Tensor, bin_edges: torch.Tensor, scale: float, zero: int,
    *, qmin: int, qmax: int, sum_sq: float | None = None,
) -> dict[str, float]:
    counts = counts.to(torch.float64)
    edges = bin_edges.to(torch.float64)
    if counts.ndim != 1 or edges.shape != (counts.numel() + 1,):
        raise ValueError("Histogram counts/edges have incompatible shapes")
    if not torch.isfinite(edges).all() or torch.any(edges[1:] <= edges[:-1]) or torch.any(counts < 0):
        raise ValueError("Histogram is invalid")
    total = float(counts.sum())
    if total <= 0.0:
        return {"mse": 0.0, "nmse": 0.0, "clip_ratio_low": 0.0, "clip_ratio_high": 0.0}
    centers = (edges[:-1] + edges[1:]) * 0.5
    reconstructed = runtime_fake_quant(centers, scale, zero, qmin=qmin, qmax=qmax)
    squared_error = float((counts * (centers - reconstructed).square()).sum())
    denominator = float((counts * centers.square()).sum()) if sum_sq is None else float(sum_sq)
    lower, upper = (qmin - int(zero)) * float(scale), (qmax - int(zero)) * float(scale)
    return {
        "mse": squared_error / total,
        "nmse": squared_error / (denominator + 1e-12),
        "clip_ratio_low": float(counts[centers < lower].sum()) / total,
        "clip_ratio_high": float(counts[centers > upper].sum()) / total,
    }


def _float_grid(start: float, stop: float, step: float) -> list[float]:
    count = int(math.floor((stop - start) / step + 1e-10))
    return sorted({round(start + index * step, 12) for index in range(count + 1)} | {round(stop, 12)})


@dataclass(frozen=True)
class _Candidate:
    mse: float
    scale: float
    zero: int
    lower: float
    upper: float
    label: str


def optimize_static_qparams(
    histogram: dict[str, Any], *, direct_scale: float, direct_zero: int,
    qmin: int, qmax: int, config: dict[str, Any],
    qparams_fn: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
) -> dict[str, Any]:
    cfg = normalize_mse_refinement_config(config)
    validate_mse_refinement_config(cfg)
    counts = torch.as_tensor(histogram["counts"], dtype=torch.int64)
    edges = torch.as_tensor(histogram["bin_edges"], dtype=torch.float64)
    sample_count = int(histogram.get("sample_count", int(counts.sum())))
    if sample_count != int(counts.sum()):
        raise ValueError("Histogram sample_count does not match counts")
    if sample_count == 0:
        return {
            "scale": float(direct_scale), "zero": int(direct_zero),
            "refinement_applied": False, "fallback_reason": "empty_branch",
            "sample_count": 0,
        }
    if qparams_fn is None:
        from .calibration import _qparams as qparams_fn
    minimum, maximum = float(edges[0]), float(edges[-1])
    direct_metrics = evaluate_histogram_mse(
        counts, edges, direct_scale, direct_zero, qmin=qmin, qmax=qmax,
        sum_sq=float(histogram.get("sum_sq", 0.0)),
    )
    candidates: list[_Candidate] = []

    def add(lower: float, upper: float, label: str, *, scale: float | None = None, zero: int | None = None) -> None:
        if scale is None or zero is None:
            s, z, _, _ = qparams_fn(
                torch.tensor(lower, dtype=torch.float64), torch.tensor(upper, dtype=torch.float64),
                qmin=qmin, qmax=qmax,
            )
            scale, zero = float(s), int(z)
        metrics = evaluate_histogram_mse(counts, edges, scale, zero, qmin=qmin, qmax=qmax)
        candidates.append(_Candidate(metrics["mse"], float(scale), int(zero), float(lower), float(upper), label))

    add((qmin - direct_zero) * direct_scale, (qmax - direct_zero) * direct_scale,
        "direct_min_max", scale=direct_scale, zero=direct_zero)

    def ranges(alpha_values: list[float]):
        if minimum < 0.0 < maximum:
            for left in alpha_values:
                for right in alpha_values:
                    yield left * minimum, right * maximum, left, right
        elif minimum >= 0.0:
            for right in alpha_values:
                yield 0.0, right * maximum, 1.0, right
        else:
            for left in alpha_values:
                yield left * minimum, 0.0, left, 1.0

    coarse = [float(value) for value in cfg["coarse_alpha_values"]]
    coarse_candidates: list[tuple[_Candidate, float, float]] = []
    start = len(candidates)
    for lower, upper, left, right in ranges(coarse):
        add(lower, upper, "coarse")
        coarse_candidates.append((candidates[-1], left, right))
    _, best_left, best_right = min(coarse_candidates, key=lambda item: item[0].mse)
    fine_radius, fine_step = float(cfg["fine_alpha_radius"]), float(cfg["fine_alpha_step"])
    fine_min, fine_max = float(cfg["fine_alpha_min"]), float(cfg["fine_alpha_max"])
    left_values = _float_grid(max(fine_min, best_left - fine_radius), min(fine_max, best_left + fine_radius), fine_step)
    right_values = _float_grid(max(fine_min, best_right - fine_radius), min(fine_max, best_right + fine_radius), fine_step)
    for lower, upper, _, _ in ranges(sorted(set(left_values + right_values))):
        # The Cartesian generator is filtered to preserve independent left/right neighborhoods.
        left = lower / minimum if minimum < 0 else 1.0
        right = upper / maximum if maximum > 0 else 1.0
        if left in left_values and right in right_values:
            add(lower, upper, "fine")
    clip_best = min(candidates[start:], key=lambda item: item.mse)
    ratios = _float_grid(
        float(cfg["local_scale_ratio_min"]), float(cfg["local_scale_ratio_max"]),
        float(cfg["local_scale_ratio_step"]),
    )
    zero_radius = int(cfg["local_zero_radius"])
    for ratio in ratios:
        scale = clip_best.scale * ratio
        for zero in range(max(qmin, clip_best.zero - zero_radius), min(qmax, clip_best.zero + zero_radius) + 1):
            add((qmin - zero) * scale, (qmax - zero) * scale, "local", scale=scale, zero=zero)

    direct_range = (qmax - qmin) * direct_scale
    def tie_key(candidate: _Candidate):
        covered = candidate.upper - candidate.lower
        return (
            abs(covered - direct_range), -covered,
            abs(candidate.scale - direct_scale), abs(candidate.zero - direct_zero),
            candidate.scale, candidate.zero, candidate.lower, candidate.upper, candidate.label,
        )
    best_mse = min(candidate.mse for candidate in candidates)
    best = min((candidate for candidate in candidates if candidate.mse <= best_mse + 1e-12), key=tie_key)
    if best.mse > direct_metrics["mse"] + 1e-12:
        if not cfg["fallback_to_direct_min_max"]:
            raise RuntimeError("MSE refinement produced a result worse than direct min-max")
        best = candidates[0]
    # Persisted scales are FP32. Re-evaluate that exact value so diagnostics and
    # the fallback guarantee describe the qparams that deployment will consume.
    persisted_scale = float(torch.tensor(best.scale, dtype=torch.float32))
    final_metrics = evaluate_histogram_mse(
        counts, edges, persisted_scale, best.zero, qmin=qmin, qmax=qmax,
        sum_sq=float(histogram.get("sum_sq", 0.0)),
    )
    applied = (
        best.label != "direct_min_max"
        and final_metrics["mse"] < direct_metrics["mse"] - 1e-12
    )
    if not applied:
        best = candidates[0]
        persisted_scale = float(direct_scale)
        final_metrics = direct_metrics
    return {
        "scale": persisted_scale, "zero": best.zero, "sample_count": sample_count,
        "baseline_mse": direct_metrics["mse"], "refined_mse": final_metrics["mse"],
        "baseline_nmse": direct_metrics["nmse"], "refined_nmse": final_metrics["nmse"],
        "optimized_lower": best.lower, "optimized_upper": best.upper,
        "representable_lower": (qmin - best.zero) * persisted_scale,
        "representable_upper": (qmax - best.zero) * persisted_scale,
        "clip_ratio_low": final_metrics["clip_ratio_low"],
        "clip_ratio_high": final_metrics["clip_ratio_high"],
        "clip_ratio_total": final_metrics["clip_ratio_low"] + final_metrics["clip_ratio_high"],
        "refinement_applied": applied,
        "fallback_reason": None if applied else "direct_min_max_is_best",
    }


def _activation_rows(activation: torch.Tensor, layout: str) -> torch.Tensor:
    value = activation.detach().to(torch.float64).cpu()
    if layout == "last_dim":
        return value.reshape(-1, value.shape[-1])
    if layout == "attention_head":
        if value.ndim != 4:
            raise ValueError(f"attention_head histogram expects [B,H,L,D], got {tuple(value.shape)}")
        return value.permute(0, 2, 1, 3).reshape(-1, value.shape[1], value.shape[3])
    raise ValueError(f"Unsupported GIF histogram layout: {layout}")


class GIFMSEHistogramStore:
    """Calibration-only fixed-bin histograms for one or more GIF sites."""

    def __init__(self, specs: dict[str, dict[str, Any]], *, histogram_bins: int):
        self.specs = specs
        self.histogram_bins = int(histogram_bins)
        self.data: dict[str, dict[str, Any]] = {}
        for key, spec in specs.items():
            branches = {}
            for branch in spec["branches"]:
                lower, upper = spec[f"{branch}_lower"].double(), spec[f"{branch}_upper"].double()
                same = upper <= lower
                upper = torch.where(same, lower + 1e-8, upper)
                steps = torch.linspace(0.0, 1.0, self.histogram_bins + 1, dtype=torch.float64)
                edges = lower.unsqueeze(-1) + (upper - lower).unsqueeze(-1) * steps
                branches[branch] = {
                    "counts": torch.zeros((*lower.shape, self.histogram_bins), dtype=torch.int64),
                    "bin_edges": edges, "sample_count": torch.zeros(lower.shape, dtype=torch.int64),
                    "sum_sq": torch.zeros(lower.shape, dtype=torch.float64),
                }
            self.data[key] = branches

    @torch.no_grad()
    def update(self, layer_index: int, site_index: int, activation: torch.Tensor, *, role: str | None = None) -> None:
        key = f"layer_{int(layer_index):03d}/site_{int(site_index):02d}"
        spec = self.specs.get(key)
        if spec is None:
            return
        rows = _activation_rows(activation, spec["layout_kind"])
        group_size = int(spec["effective_group_size"])
        grouped = rows.reshape(*rows.shape[:-1], rows.shape[-1] // group_size, group_size)
        if spec["layout_kind"] == "last_dim":
            grouped = grouped.reshape(-1, grouped.shape[-2], group_size)
        else:
            grouped = grouped.reshape(-1, grouped.shape[-3], grouped.shape[-2], group_size)
        roles = [role] if role is not None else list(spec["mask_low_by_role"])
        for active_role in roles:
            if active_role not in spec["mask_low_by_role"]:
                raise ValueError(f"Unexpected GIF histogram role {active_role!r} for {key}")
            mask = spec["mask_low_by_role"][active_role].bool().reshape(*grouped.shape[1:])
            for branch, branch_mask in (("low", mask), ("high", ~mask)):
                if branch not in self.data[key]:
                    continue
                target = self.data[key][branch]
                flat_groups = grouped.reshape(grouped.shape[0], -1, group_size)
                flat_mask = branch_mask.reshape(-1, group_size)
                flat_edges = target["bin_edges"].reshape(-1, self.histogram_bins + 1)
                valid = flat_mask.unsqueeze(0).expand(flat_groups.shape[0], -1, -1)
                if not torch.any(valid):
                    continue
                lower = flat_edges[:, 0].reshape(1, -1, 1)
                upper = flat_edges[:, -1].reshape(1, -1, 1)
                widths = (upper - lower) / self.histogram_bins
                bin_indices = torch.floor((flat_groups - lower) / widths).to(torch.int64)
                bin_indices.clamp_(0, self.histogram_bins - 1)
                offsets = (
                    torch.arange(flat_groups.shape[1], dtype=torch.int64)
                    .reshape(1, -1, 1)
                    .mul(self.histogram_bins)
                )
                linear_indices = (bin_indices + offsets)[valid]
                histogram_counts = torch.bincount(
                    linear_indices, minlength=flat_groups.shape[1] * self.histogram_bins
                ).reshape_as(target["counts"])
                target["counts"].add_(histogram_counts)
                target["sample_count"].add_(
                    valid.sum(dim=(0, 2)).reshape_as(target["sample_count"])
                )
                target["sum_sq"].add_(
                    torch.where(valid, flat_groups.square(), 0.0)
                    .sum(dim=(0, 2))
                    .reshape_as(target["sum_sq"])
                )

    def state_for(self, key: str, metadata: dict[str, Any]) -> dict[str, Any]:
        spec = self.specs[key]
        return {
            "format_version": 1,
            "refinement_version": "static_mse_v1",
            "configured_group_size": spec["configured_group_size"],
            "effective_group_size": spec["effective_group_size"],
            "parameter_layout": spec["parameter_layout"],
            "histogram_bins": self.histogram_bins,
            "low": self.data[key].get("low"),
            "high": self.data[key].get("high"),
            **metadata,
        }
