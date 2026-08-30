from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import torch

from .artifacts import sha256_file
from .neurons import Clipper, MultiThresholdNeuron, PhaseSurrogate, gif_module_from_state
from .sites import (
    GIF_ALL_LOW_SITE_IDS, GIF_IDENTITY_SITE_IDS, GIF_MULTI_MASK_ROLES,
    GIF_SALIENT_SITE_IDS, SITE_IDS, is_softmax_site, site_supports_clip,
    validate_site_topology,
)
from .temporal_ops import (
    CALIBRATION_MANIFEST_FORMAT_VERSION,
    GIF_BASE_BITS,
    GIF_ADD_BITS,
    GIF_HIGH_QMAX,
    GIF_LOW_QMAX,
    GIF_LOCAL_STEPS,
    GIF_SALIENT_POLICY,
    GIF_ALL_LOW_POLICY,
    GIF_IDENTITY_POLICY,
    GIF_STEP_QMAX,
    CALIBRATION_GROUPING_POLICY,
    SOFTMAX_SITE5_CLIP_POLICY,
    SOFTMAX_SITE5_GIF_POLICY,
    STATISTICS_FORMAT_VERSION,
    TEMPORAL_IMPLEMENTATION_VERSION,
    validate_temporal_policy,
)


_FACTORIES = {
    "phase": lambda state: PhaseSurrogate(state, T=1),
    "gif": gif_module_from_state,
    "mtn": lambda state: MultiThresholdNeuron(state, T=1, K=1, threshold_factor=0.75),
    "clip": Clipper,
}

ClipBundlePolicy = Literal["forbid_all"]
CLIP_BUNDLE_POLICIES = frozenset({"forbid_all"})


def _forbidden_manifest_paths(value: Any, forbidden: set[str], prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in forbidden:
                paths.append(path)
            paths.extend(_forbidden_manifest_paths(item, forbidden, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_forbidden_manifest_paths(item, forbidden, f"{prefix}[{index}]"))
    return paths

def load_calibration_manifest(site_root: str | Path) -> dict[str, Any]:
    path = Path(site_root) / "calibration_state_manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != CALIBRATION_MANIFEST_FORMAT_VERSION:
        raise ValueError(
            "Incompatible legacy calibration manifest schema; calibration manifest "
            f"v{CALIBRATION_MANIFEST_FORMAT_VERSION} with temporal implementation "
            f"v{TEMPORAL_IMPLEMENTATION_VERSION} is required. "
            "Re-materialize calibration states and conversion descriptors before "
            "SNN evaluation"
        )
    forbidden_paths = _forbidden_manifest_paths(
        manifest, {"phase_T", "mtn_T", "mtn_K", "max_spikes", "v0"}
    )
    if forbidden_paths:
        raise ValueError(
            "Stage A manifest contains runtime-dependent fields: "
            + ", ".join(forbidden_paths)
        )
    validate_temporal_policy(manifest, context=str(path))
    expected = {
        "statistics_format_version": STATISTICS_FORMAT_VERSION,
        "calibration_architecture": "two_stage_A_common_B_clip_profiles",
        "calibration_phase": "A",
        "calibration_grouping_policy": CALIBRATION_GROUPING_POLICY,
        "softmax_site5_gif_policy": SOFTMAX_SITE5_GIF_POLICY,
        "softmax_site5_clip_policy": SOFTMAX_SITE5_CLIP_POLICY,
    }
    mismatched = {
        key: (value, manifest.get(key))
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    group_size = manifest.get("calibration_group_size")
    if mismatched or not isinstance(group_size, int) or group_size == 0 or group_size < -1:
        raise ValueError(f"Calibration manifest has invalid grouping provenance: {mismatched}")
    return manifest


def validate_clip_profile(
    site_root: str | Path,
    clip_root: str | Path,
    *,
    phase_T: int,
    mtn_T: int,
    group_size: int,
    num_samples: int,
) -> dict[str, Any]:
    stage_a_root, root = Path(site_root), Path(clip_root)
    path = root / "clip_profile_manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    profile = json.loads(path.read_text(encoding="utf-8"))
    stage_a_path = stage_a_root / "calibration_state_manifest.json"
    expected = {
        "format_version": CALIBRATION_MANIFEST_FORMAT_VERSION,
        "calibration_phase": "B",
        "phase_T": int(phase_T),
        "mtn_T": int(mtn_T),
        "phase_base": 2.0,
        "calibration_group_size": int(group_size),
        "calibration_num_samples": int(num_samples),
        "stage_a_root": str(stage_a_root.resolve()),
        "stage_a_calibration_manifest_path": str(stage_a_path.resolve()),
        "stage_a_calibration_manifest_sha256": sha256_file(stage_a_path),
        "clip_policy_version": "mask_aware_role_specific_v1",
        "mask_aware_policy": "all_low_all_high_mixed_per_group",
        "role_specific_policy": "site1_qkv_site7_gate_up",
    }
    mismatched = {
        key: (value, profile.get(key))
        for key, value in expected.items()
        if profile.get(key) != value
    }
    if root.name != f"phase_T_{int(phase_T)}_mtn_T_{int(mtn_T)}":
        mismatched["profile_dirname"] = (
            f"phase_T_{int(phase_T)}_mtn_T_{int(mtn_T)}", root.name
        )
    if mismatched:
        raise ValueError(f"Stage B Clip profile provenance mismatch: {mismatched}")
    validate_temporal_policy(profile, context=str(path))
    stage_a = load_calibration_manifest(stage_a_root)
    stage_a_sites = stage_a.get("sites", {})
    profile_sites = profile.get("sites", {})
    if set(profile_sites) != set(stage_a_sites):
        raise ValueError("Stage B Clip profile topology differs from Stage A")
    actual_site_dirs = {
        directory.relative_to(root).as_posix()
        for directory in root.glob("layer_*/site_*")
        if directory.is_dir()
    }
    if actual_site_dirs != set(stage_a_sites):
        raise ValueError("Stage B Clip profile directories differ from Stage A topology")
    for key, stage_a_summary in stage_a_sites.items():
        site_index = int(stage_a_summary["site_index"])
        directory = root / key
        summary_path = directory / "calibration_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if profile_sites[key] != summary:
            raise ValueError(f"Stage B manifest/site summary mismatch: {summary_path}")
        expected_a_hashes = {
            "state_a_phase_sha256": stage_a_summary["state_sha256"]["phase"],
            "state_a_gif_sha256": stage_a_summary["state_sha256"]["gif"],
            "state_a_mtn_sha256": stage_a_summary["state_sha256"]["mtn"],
        }
        if any(summary.get(name) != value for name, value in expected_a_hashes.items()):
            raise ValueError(f"Stage B site provenance differs from Stage A: {summary_path}")
        clip_path = directory / "clip_state.pt"
        if not site_supports_clip(site_index):
            if clip_path.exists() or summary.get("clip_state_present") is not False:
                raise ValueError(f"Site 5 permanently forbids Clip state: {clip_path}")
            continue
        if not clip_path.exists() or summary.get("clip_state_sha256") != sha256_file(clip_path):
            raise ValueError(f"Stage B Clip state hash mismatch: {clip_path}")
        state = torch.load(clip_path, map_location="cpu", weights_only=False)
        if state.get("phase_T") != int(phase_T) or state.get("mtn_T") != int(mtn_T):
            raise ValueError(f"Stage B Clip runtime bounds differ from profile: {clip_path}")
        Clipper(state)
        expected_roles = GIF_MULTI_MASK_ROLES.get(site_index)
        if expected_roles is not None and (
            state.get("clip_role_policy") != "role_specific"
            or tuple(state.get("clip_roles", ())) != expected_roles
        ):
            raise ValueError(f"Invalid role-specific Clip schema at {clip_path}")
        if expected_roles is None and state.get("clip_role_policy") != "single":
            raise ValueError(f"Invalid single-role Clip schema at {clip_path}")
    return profile

def validate_site_state_bundle(
    site_root: str | Path,
    manifest: dict[str, Any] | None = None,
    *,
    clip_policy: ClipBundlePolicy,
    expected_num_hidden_layers: int | None = None,
) -> dict[str, Any]:
    if clip_policy not in CLIP_BUNDLE_POLICIES:
        raise ValueError(f"Unknown Clip bundle policy {clip_policy!r}")
    root = Path(site_root)
    manifest = load_calibration_manifest(root) if manifest is None else manifest
    manifest_layers = manifest.get("expected_num_hidden_layers")
    if not isinstance(manifest_layers, int) or isinstance(manifest_layers, bool) or manifest_layers <= 0:
        raise ValueError("Calibration manifest expected_num_hidden_layers must be a positive integer")
    expected_names = [f"layer_{index:03d}" for index in range(manifest_layers)]
    if manifest.get("expected_layer_names") != expected_names:
        raise ValueError("Calibration manifest expected_layer_names does not match expected_num_hidden_layers")
    if expected_num_hidden_layers is not None and expected_num_hidden_layers != manifest_layers:
        raise ValueError(
            "ANN config num_hidden_layers does not match calibration manifest "
            f"expected_num_hidden_layers: {expected_num_hidden_layers} != {manifest_layers}"
        )
    site_sets = validate_site_topology(root, expected_num_hidden_layers=manifest_layers)
    gif_steps: set[int] = set()
    site_count = 0
    for layer_name in sorted(site_sets):
        for directory in sorted((root / layer_name).glob("site_*")):
            site_count += 1
            key = directory.relative_to(root).as_posix()
            entry = manifest.get("sites", {}).get(key)
            if not isinstance(entry, dict):
                raise ValueError(f"Calibration manifest is missing site entry: {key}")
            site_index = int(entry.get("site_index", -1))
            hashes = entry.get("state_sha256")
            if set(hashes or {}) != {"phase", "gif", "mtn"}:
                raise ValueError(f"Stage A manifest must hash exactly phase/gif/mtn: {key}")
            clip_path = directory / "clip_state.pt"
            summary_path = directory / "calibration_summary.json"
            if clip_path.exists() or summary_path.exists():
                raise ValueError(f"Stage A calibration bundle must be clip-free: {directory}")
            states = {}
            for kind in ("phase", "gif", "mtn"):
                state_path = directory / f"{kind}_state.pt"
                if not state_path.exists():
                    raise FileNotFoundError(state_path)
                if sha256_file(state_path) != hashes[kind]:
                    raise ValueError(f"Calibration state hash mismatch: {state_path}")
                states[kind] = torch.load(state_path, map_location="cpu", weights_only=False)
            try:
                PhaseSurrogate(states["phase"], T=1)
                MultiThresholdNeuron(states["mtn"], T=1, K=1, threshold_factor=0.75)
                gif = gif_module_from_state(states["gif"])
            except Exception as exc:
                raise ValueError(f"Invalid Stage A state at {directory}: {exc}") from exc
            gif_steps.add(int(gif.temporal_steps))
            gif_state = states["gif"]
            expected_roles = GIF_MULTI_MASK_ROLES.get(site_index)
            if expected_roles is not None:
                if (
                    gif_state.get("mask_policy") != "multi_role"
                    or tuple(gif_state.get("mask_roles", ())) != expected_roles
                    or set(gif_state.get("mask_low_by_role", {})) != set(expected_roles)
                ):
                    raise ValueError(f"GIF saliency provenance mismatch at {directory}")
            if site_index in {3, 4, 6, 10} and gif_state.get("mask_policy") != "single":
                raise ValueError(f"GIF saliency provenance mismatch at {directory}")
            expected_manifest = {
                "saliency_enabled": bool(gif_state.get("saliency_enabled", False)),
                "saliency_roles": list(gif_state.get("saliency_rule_by_role", {}).keys()),
                "saliency_rule_by_role": dict(gif_state.get("saliency_rule_by_role", {})),
                "saliency_accumulator_dtype_by_role": dict(gif_state.get("saliency_accumulator_dtype_by_role", {})),
                "gif_mask_policy": gif_state.get("mask_policy"),
                "gif_mask_roles": list(gif_state.get("mask_roles", [])),
            }
            if any(entry.get(name) != value for name, value in expected_manifest.items()):
                raise ValueError(f"GIF saliency provenance mismatch at {directory}")
            if site_index == 6:
                for kind, state in states.items():
                    if state.get("parameter_layout") != "last_dim_grouped" or state.get("num_heads") is not None:
                        raise ValueError(f"Site 6 must use merged last-dim state: {directory}/{kind}_state.pt")
    global_entry = manifest.get("global_states", {}).get("final_rmsnorm")
    if not isinstance(global_entry, dict):
        raise ValueError("Calibration manifest is missing global final RMSNorm Phase state")
    relative_path = global_entry.get("phase_state_path")
    global_hash = global_entry.get("phase_state_sha256")
    if not isinstance(relative_path, str) or not isinstance(global_hash, str):
        raise ValueError("Final RMSNorm Phase state metadata is incomplete")
    global_path = root / relative_path
    if not global_path.exists() or sha256_file(global_path) != global_hash:
        raise ValueError(f"Final RMSNorm Phase state hash mismatch: {global_path}")
    try:
        PhaseSurrogate(torch.load(global_path, map_location="cpu", weights_only=False), T=1)
    except Exception as exc:
        raise ValueError(f"Invalid final RMSNorm Phase state at {global_path}: {exc}") from exc
    if len(gif_steps) != 1:
        raise ValueError(f"Inconsistent temporal steps across GIF site states: {sorted(gif_steps)}")
    return {
        "expected_num_hidden_layers": manifest_layers,
        "layers": len(site_sets),
        "sites": site_count,
        "temporal_steps": {"gif": next(iter(gif_steps))},
        "manifest": manifest,
        "final_norm_phase_state": str(global_path),
    }
