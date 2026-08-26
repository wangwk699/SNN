from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from .artifacts import ArtifactLayout, read_json, sha256_file
from .conversion import validate_conversion_metadata
from .neurons import PhaseSurrogate
from .sites import SITE_IDS, site_key
from .temporal_ops import (
    from_temporal,
    temporal_bias_once,
    temporal_rmsnorm,
    temporal_seq_matmul,
    temporal_silu,
    temporal_softmax,
    temporal_symmetric_hadamard,
    to_temporal,
)


def tensor_metrics(reference: torch.Tensor, test: torch.Tensor, *, name: str) -> dict[str, Any]:
    reference = reference.detach().float().cpu()
    test = test.detach().float().cpu()
    if reference.shape != test.shape:
        raise ValueError(f"Checkpoint {name} shape mismatch: {reference.shape} != {test.shape}")
    ref_flat = reference.reshape(-1)
    test_flat = test.reshape(-1)
    diff = test_flat - ref_flat
    reference_l2 = float(torch.linalg.vector_norm(ref_flat))
    test_l2 = float(torch.linalg.vector_norm(test_flat))
    diff_l2 = float(torch.linalg.vector_norm(diff))
    denominator = max(reference_l2, 1e-12)
    cosine = 1.0
    if reference_l2 > 0.0 and test_l2 > 0.0:
        cosine = float(F.cosine_similarity(ref_flat, test_flat, dim=0))
    elif reference_l2 != test_l2:
        cosine = 0.0
    return {
        "name": name,
        "shape": list(reference.shape),
        "reference_l2": reference_l2,
        "test_l2": test_l2,
        "diff_l2": diff_l2,
        "relative_l2_error": diff_l2 / denominator,
        "mean_abs_error": float(diff.abs().mean()) if diff.numel() else 0.0,
        "max_abs_error": float(diff.abs().max()) if diff.numel() else 0.0,
        "cosine_similarity": cosine,
        "reference_zero_fraction": float((ref_flat == 0).float().mean()) if ref_flat.numel() else 0.0,
        "test_zero_fraction": float((test_flat == 0).float().mean()) if test_flat.numel() else 0.0,
    }


def logits_metrics(reference: torch.Tensor, test: torch.Tensor, *, name: str) -> dict[str, Any]:
    result = tensor_metrics(reference, test, name=name)
    reference_last = reference.detach().float().cpu()[0, -1]
    test_last = test.detach().float().cpu()[0, -1]
    ref_top1 = int(reference_last.argmax())
    test_top1 = int(test_last.argmax())
    ref_top5 = set(torch.topk(reference_last, min(5, reference_last.numel())).indices.tolist())
    test_top5 = set(torch.topk(test_last, min(5, test_last.numel())).indices.tolist())
    result.update(
        {
            "last_token_top1_equal": ref_top1 == test_top1,
            "last_token_top1_id_ref": ref_top1,
            "last_token_top1_id_test": test_top1,
            "last_token_top5_overlap": len(ref_top5 & test_top5),
        }
    )
    return result


class PhaseConversionRegressionRecorder:
    """In-memory float32 checkpoint recorder for one fixed forward graph."""

    def __init__(self, graph: str, temporal_steps: int | None = None):
        self.graph = str(graph)
        self.temporal_steps = int(temporal_steps) if temporal_steps is not None else None
        self.tensors: OrderedDict[str, torch.Tensor] = OrderedDict()

    def record(self, name: str, value: torch.Tensor, *, temporal: bool = False) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Checkpoint {name} is not a tensor")
        captured = value.detach()
        if temporal:
            if self.temporal_steps is None:
                raise RuntimeError("Temporal checkpoint reduction requires temporal_steps")
            captured = to_temporal(captured, self.temporal_steps).sum(dim=0)
        self.tensors[name] = captured.float().cpu().clone()

    def compare(self, test: "PhaseConversionRegressionRecorder") -> list[dict[str, Any]]:
        missing = [name for name in self.tensors if name not in test.tensors]
        unexpected = [name for name in test.tensors if name not in self.tensors]
        if missing or unexpected:
            raise ValueError(f"Checkpoint sets differ (missing={missing}, unexpected={unexpected})")
        rows = [
            tensor_metrics(reference, test.tensors[name], name=name)
            for name, reference in self.tensors.items()
        ]
        by_name = {row["name"]: row for row in rows}
        for row in rows:
            if not row["name"].endswith("/post"):
                continue
            pre_name = row["name"][:-4] + "pre"
            if pre_name in by_name:
                row["local_error_amplification"] = row["relative_l2_error"] / max(
                    by_name[pre_name]["relative_l2_error"], 1e-12
                )
        return rows


def summarize_first_divergence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def first_over(threshold: float):
        return next((row for row in rows if row["relative_l2_error"] > threshold), None)

    maximum = max(rows, key=lambda row: row["relative_l2_error"], default=None)
    amplified = max(
        (row for row in rows if "local_error_amplification" in row),
        key=lambda row: row["local_error_amplification"],
        default=None,
    )
    return {
        "first_relative_l2_gt_1e-3": first_over(1e-3),
        "first_relative_l2_gt_1e-2": first_over(1e-2),
        "maximum_relative_l2": maximum,
        "maximum_local_error_amplification": amplified,
    }


def validate_phase_conversion_artifacts(
    cfg: dict[str, Any], layout: ArtifactLayout
) -> dict[str, Any]:
    prerequisites = {
        "experiment.ann_mode": cfg["experiment"].get("ann_mode"),
        "replacement.common_clip_enabled": cfg["replacement"].get("common_clip_enabled"),
        "ann_training.prefix_enabled": cfg.get("ann_training", {}).get("prefix_enabled"),
    }
    expected_prerequisites = {
        "experiment.ann_mode": "phase_aware",
        "replacement.common_clip_enabled": False,
        "ann_training.prefix_enabled": True,
    }
    mismatched = {
        key: {"expected": expected, "actual": prerequisites[key]}
        for key, expected in expected_prerequisites.items()
        if prerequisites[key] != expected
    }
    if mismatched:
        raise ValueError(f"Phase conversion regression prerequisites failed: {mismatched}")

    prefix_state = layout.ann_training_prefix_dir / "prefix_state.json"
    prefix_ids = [int(value) for value in read_json(prefix_state).get("prefix_token_ids", [])] if prefix_state.exists() else []
    prefix_kv = layout.ann_training_prefix_dir / "prefixed_key_values.pt"
    manifest = layout.ann_training_site_dir / "calibration_state_manifest.json"
    conversion_path = layout.snn_conversion_dir("phase") / "conversion_metadata.json"
    required = [layout.ann_checkpoint_dir, prefix_state, manifest, conversion_path]
    if prefix_ids:
        required.append(prefix_kv)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Phase conversion regression artifacts are missing: {missing}")

    ann_config_path = layout.ann_checkpoint_dir / "config.json"
    training_result_path = layout.ann_dir / "training_result.json"
    if not ann_config_path.exists() or not training_result_path.exists():
        raise FileNotFoundError(ann_config_path if not ann_config_path.exists() else training_result_path)
    ann_config = read_json(ann_config_path)
    training_result = read_json(training_result_path)
    expected_ann = {
        "snn2_ann_mode": "phase_aware",
        "snn2_ann_common_clip_enabled": False,
    }
    expected_training = {
        "ann_training_common_clip_enabled": False,
        "ann_training_common_clip_applied": False,
        "ann_training_common_clip_state_required": True,
        "ann_training_prefix_state_sha256": sha256_file(prefix_state),
        "ann_training_prefix_kv_sha256": sha256_file(prefix_kv) if prefix_ids else None,
        "ann_training_calibration_manifest_sha256": sha256_file(manifest),
        "final_model_checkpoint": str(layout.ann_checkpoint_dir.resolve()),
    }
    ann_mismatch = {
        key: {"expected": value, "actual": ann_config.get(key)}
        for key, value in expected_ann.items()
        if ann_config.get(key) != value
    }
    training_mismatch = {
        key: {"expected": value, "actual": training_result.get(key)}
        for key, value in expected_training.items()
        if training_result.get(key) != value
    }
    if ann_mismatch or training_mismatch:
        raise ValueError(
            "ANN training provenance mismatch: "
            f"ann_config={ann_mismatch}, training_result={training_mismatch}"
        )

    conversion = validate_conversion_metadata(cfg, layout, "phase")
    expected_conversion = {
        "deployment_neuron": "phase",
        "reused_ann_training_artifacts": True,
        "post_finetuning_recalibration": False,
        "source_ann_common_clip_enabled": False,
        "snn_clip_applied": False,
        "source_ann_checkpoint": str(layout.ann_checkpoint_dir.resolve()),
        "prefix_state_sha256": expected_training["ann_training_prefix_state_sha256"],
        "prefix_kv_sha256": expected_training["ann_training_prefix_kv_sha256"],
        "calibration_state_manifest_sha256": expected_training[
            "ann_training_calibration_manifest_sha256"
        ],
    }
    conversion_mismatch = {
        key: {"expected": value, "actual": conversion.get(key)}
        for key, value in expected_conversion.items()
        if conversion.get(key) != value
    }
    if conversion_mismatch:
        raise ValueError(f"Conversion provenance mismatch: {conversion_mismatch}")
    return {
        "validated": True,
        "prerequisites": prerequisites,
        "paths": {
            "ann_checkpoint": str(layout.ann_checkpoint_dir.resolve()),
            "prefix_state": str(prefix_state.resolve()),
            "prefix_kv": str(prefix_kv.resolve()) if prefix_ids else None,
            "calibration_manifest": str(manifest.resolve()),
            "conversion_metadata": str(conversion_path.resolve()),
        },
        "sha256": {
            "ann_config": sha256_file(ann_config_path),
            "prefix_state": sha256_file(prefix_state),
            "prefix_kv": sha256_file(prefix_kv) if prefix_ids else None,
            "calibration_manifest": sha256_file(manifest),
        },
        "prefix_token_ids": prefix_ids,
        "conversion": conversion,
    }


def _selected_phase_state_paths(site_root: Path, num_layers: int) -> list[Path]:
    layers = sorted({0, num_layers // 2, num_layers - 1})
    paths = [site_root / site_key(layer, site) / "phase_state.pt" for layer in layers for site in SITE_IDS]
    paths.append(site_root / "_global" / "final_rmsnorm" / "phase_state.pt")
    return paths


def run_phase_neuron_micro_regression(
    site_root: str | Path, num_layers: int, *, seed: int = 42
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    cases = []
    for path in _selected_phase_state_paths(Path(site_root), num_layers):
        state = torch.load(path, map_location="cpu", weights_only=False)
        module = PhaseSurrogate(state).eval()
        layout = state["parameter_layout"]
        if layout == "last_dim_grouped":
            x = torch.randn(
                1, 3, int(state["channels_per_head"]), generator=generator
            )
        elif layout == "attention_head_grouped":
            x = torch.randn(
                1,
                int(state["num_heads"]),
                3,
                int(state["channels_per_head"]),
                generator=generator,
            )
        elif layout == "attention_head_scalar":
            x = torch.randn(
                1, int(state["num_heads"]), 3, 5, generator=generator
            )
        else:
            raise ValueError(f"Unsupported Phase regression layout: {layout}")
        static = module(x)
        steps = int(module.T)
        decompositions = {
            "first_frame": torch.cat((x.unsqueeze(0), torch.zeros(steps - 1, *x.shape)), dim=0),
            "uniform": x.unsqueeze(0).expand(steps, *x.shape) / steps,
        }
        random_frames = torch.randn(steps - 1, *x.shape, generator=generator)
        decompositions["random_sum_preserving"] = torch.cat(
            (random_frames, (x - random_frames.sum(0)).unsqueeze(0)), dim=0
        )
        for decomposition, incoming in decompositions.items():
            temporal_sum = module.temporal(incoming).sum(0)
            metric = tensor_metrics(static, temporal_sum, name=f"{path.parent}/{decomposition}")
            metric["exact_equal"] = bool(torch.equal(static, temporal_sum))
            metric["passed"] = metric["relative_l2_error"] <= 1e-7 and metric["max_abs_error"] <= 1e-7
            cases.append(metric)
    return {"threshold": 1e-7, "passed": all(case["passed"] for case in cases), "cases": cases}


class _RegressionRMSNorm(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(0.7, 1.3, hidden))
        self.variance_epsilon = 1e-6


def run_temporal_primitive_regression(*, steps: int = 4, seed: int = 42) -> dict[str, Any]:
    results: dict[str, Any] = {"fp32_threshold": 1e-6, "dtypes": {}}
    for dtype in (torch.float32, torch.bfloat16):
        generator = torch.Generator().manual_seed(seed)
        cases = []

        def add(name: str, reference: torch.Tensor, test: torch.Tensor) -> None:
            metric = tensor_metrics(reference, test, name=name)
            metric["passed"] = metric["relative_l2_error"] <= (1e-6 if dtype == torch.float32 else 5e-2)
            cases.append(metric)

        x = torch.randn(steps, 2, 3, 8, generator=generator).to(dtype)
        norm = _RegressionRMSNorm(8)
        total = x.float().sum(0)
        norm_ref = total * torch.rsqrt(total.square().mean(-1, keepdim=True) + norm.variance_epsilon) * norm.weight
        add("temporal_rmsnorm", norm_ref, temporal_rmsnorm(x, norm).sum(0))
        add("temporal_silu", F.silu(total), temporal_silu(x).sum(0))

        a = torch.randn(steps, 2, 2, 3, 5, generator=generator).to(dtype)
        b = torch.randn(steps, 2, 2, 5, 4, generator=generator).to(dtype)
        add("temporal_seq_matmul", torch.matmul(a.float().sum(0), b.float().sum(0)), temporal_seq_matmul(a, b).sum(0))

        scores = torch.randn(steps, 2, 2, 3, 5, generator=generator).to(dtype)
        mask = torch.zeros(2, 1, 3, 5)
        mask[..., 3:] = float("-inf")
        softmax_ref = F.softmax(scores.float().sum(0) + mask, dim=-1)
        add("temporal_softmax_causal_prefix", softmax_ref, temporal_softmax(scores, mask).sum(0))

        h1 = torch.randn(steps, 2, 3, 8, generator=generator).to(dtype)
        h2 = torch.randn(steps, 2, 3, 8, generator=generator).to(dtype)
        add("temporal_symmetric_hadamard", h1.float().sum(0) * h2.float().sum(0), temporal_symmetric_hadamard(h1, h2).sum(0))

        linear = torch.nn.Linear(8, 6, bias=True).to(dtype)
        linear_input = torch.randn(steps, 2, 3, 8, generator=generator).to(dtype)
        repeated = linear(from_temporal(linear_input))
        add("temporal_bias_once", linear(linear_input.sum(0)), to_temporal(temporal_bias_once(repeated, linear.bias, steps), steps).sum(0))

        embedding = torch.randn(2, 3, 8, generator=generator).to(dtype)
        add("embedding_divide_by_T", embedding, (embedding.unsqueeze(0).expand(steps, -1, -1, -1) / steps).sum(0))
        prefix = torch.randn(2, 2, 3, 8, generator=generator).to(dtype)
        add("prefix_kv_divide_by_T", prefix, (prefix.unsqueeze(0).expand(steps, -1, -1, -1, -1) / steps).sum(0))

        results["dtypes"][str(dtype).removeprefix("torch.")] = {
            "passed": all(case["passed"] for case in cases),
            "cases": cases,
        }
    results["passed"] = results["dtypes"]["float32"]["passed"]
    return results
