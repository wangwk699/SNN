from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .artifacts import sha256_file
from .neurons import Clipper, MultiThresholdNeuron, PhaseSurrogate, StaticGIF
from .sites import SITE_IDS, validate_site_topology
from .temporal_ops import (
    CALIBRATION_MANIFEST_FORMAT_VERSION,
    GIF_HIGH_QMAX,
    GIF_LOCAL_STEPS,
    GIF_STEP_QMAX,
    validate_temporal_policy,
)


_FACTORIES = {
    "phase": PhaseSurrogate,
    "gif": StaticGIF,
    "mtn": MultiThresholdNeuron,
    "clip": Clipper,
}


def load_calibration_manifest(site_root: str | Path) -> dict[str, Any]:
    path = Path(site_root) / "calibration_state_manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != CALIBRATION_MANIFEST_FORMAT_VERSION:
        raise ValueError(
            "Incompatible legacy calibration manifest schema; calibration manifest "
            "v4 with temporal implementation v3 is required. "
            "Re-materialize calibration states and conversion descriptors before "
            "SNN evaluation"
        )
    validate_temporal_policy(manifest, context=str(path))
    return manifest


def validate_site_state_bundle(
    site_root: str | Path,
    manifest: dict[str, Any] | None = None,
    *,
    require_clip: bool,
    expected_num_hidden_layers: int | None = None,
) -> dict[str, Any]:
    root = Path(site_root)
    manifest = load_calibration_manifest(root) if manifest is None else manifest
    if manifest.get("format_version") != CALIBRATION_MANIFEST_FORMAT_VERSION:
        raise ValueError(
            "Incompatible legacy calibration manifest schema; re-materialize "
            "calibration states for temporal implementation v3"
        )
    manifest_layers = manifest.get("expected_num_hidden_layers")
    if (
        not isinstance(manifest_layers, int)
        or isinstance(manifest_layers, bool)
        or manifest_layers <= 0
    ):
        raise ValueError(
            "Calibration manifest expected_num_hidden_layers must be a positive integer"
        )
    expected_layer_names = [
        f"layer_{index:03d}" for index in range(manifest_layers)
    ]
    if manifest.get("expected_layer_names") != expected_layer_names:
        raise ValueError(
            "Calibration manifest expected_layer_names does not match "
            "expected_num_hidden_layers"
        )
    if (
        expected_num_hidden_layers is not None
        and expected_num_hidden_layers != manifest_layers
    ):
        raise ValueError(
            "ANN config num_hidden_layers does not match calibration manifest "
            f"expected_num_hidden_layers: {expected_num_hidden_layers} != {manifest_layers}"
        )
    site_sets = validate_site_topology(
        root, expected_num_hidden_layers=manifest_layers
    )
    validate_temporal_policy(manifest, context=str(root / "calibration_state_manifest.json"))

    steps_by_neuron: dict[str, set[int]] = {"phase": set(), "gif": set(), "mtn": set()}
    site_count = 0
    required = ("phase", "gif", "mtn", "clip") if require_clip else (
        "phase",
        "gif",
        "mtn",
    )
    for layer_name in sorted(site_sets):
        if len(site_sets[layer_name]) != len(SITE_IDS):
            raise RuntimeError(f"{layer_name} does not contain exactly {len(SITE_IDS)} sites")
        for directory in sorted((root / layer_name).glob("site_*")):
            site_count += 1
            site_key = directory.relative_to(root).as_posix()
            site_manifest = manifest.get("sites", {}).get(site_key)
            if not isinstance(site_manifest, dict):
                raise ValueError(f"Calibration manifest is missing site entry: {site_key}")
            saved_hashes = site_manifest.get("state_sha256")
            if not isinstance(saved_hashes, dict):
                raise ValueError(f"Calibration manifest is missing state hashes: {site_key}")
            for kind, expected_hash in saved_hashes.items():
                state_path = directory / f"{kind}_state.pt"
                if not state_path.exists() or sha256_file(state_path) != expected_hash:
                    raise ValueError(f"Calibration state hash mismatch: {state_path}")
            for kind in required:
                state_path = directory / f"{kind}_state.pt"
                if not state_path.exists():
                    raise FileNotFoundError(state_path)
                state = torch.load(state_path, map_location="cpu", weights_only=False)
                try:
                    module = _FACTORIES[kind](state)
                except Exception as exc:
                    raise ValueError(f"Invalid {kind} state at {state_path}: {exc}") from exc
                if kind in {"phase", "mtn"}:
                    steps_by_neuron[kind].add(int(module.T))
                elif kind == "gif":
                    steps_by_neuron[kind].add(int(module.temporal_steps))
                    if (
                        state["high_qmax"] != GIF_HIGH_QMAX
                        or state["temporal_steps"] != GIF_LOCAL_STEPS
                        or state["per_step_qmax"] != GIF_STEP_QMAX
                    ):
                        raise ValueError(f"Invalid GIF qmax/chunk policy at {state_path}")

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
        final_norm_phase = PhaseSurrogate(
            torch.load(global_path, map_location="cpu", weights_only=False)
        )
    except Exception as exc:
        raise ValueError(f"Invalid final RMSNorm Phase state at {global_path}: {exc}") from exc
    if int(final_norm_phase.T) not in steps_by_neuron["phase"]:
        raise ValueError("Final RMSNorm Phase T differs from per-site Phase T")

    inconsistent = {
        neuron: sorted(values)
        for neuron, values in steps_by_neuron.items()
        if len(values) != 1
    }
    if inconsistent:
        raise ValueError(f"Inconsistent temporal steps across site states: {inconsistent}")
    return {
        "expected_num_hidden_layers": manifest_layers,
        "layers": len(site_sets),
        "sites": site_count,
        "temporal_steps": {
            neuron: next(iter(values)) for neuron, values in steps_by_neuron.items()
        },
        "manifest": manifest,
        "final_norm_phase_state": str(global_path),
    }
