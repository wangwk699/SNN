from __future__ import annotations

import json

import pytest
import torch

from snn2.artifacts import sha256_file
from snn2.calibration import materialize_calibration_states
from snn2.conversion import validate_conversion_metadata
from snn2.sites import SITE_IDS, SITE_NAMES
from snn2.temporal_ops import (
    CONVERSION_METADATA_FORMAT_VERSION,
    GIF_LOCAL_STEPS,
    temporal_policy_metadata,
)


class _Layout:
    def __init__(self, root):
        self.root = root
        self.post_finetuning_site_dir = root / "sites"
        self.post_finetuning_prefix_dir = root / "prefix"
        self.pre_finetuning_prefix_dir = root / "pre_prefix"
        self.ann_training_site_dir = root / "ann_sites"
        self.ann_dir = root / "ann"
        self.conversion_prefix_dir = self.post_finetuning_prefix_dir
        self.conversion_site_dir = self.post_finetuning_site_dir
        self.ann_checkpoint_dir = root / "ann" / "final"
        self.rotation_dir = root / "rotation"

    def snn_conversion_dir(self, neuron):
        return self.root / "snn" / neuron / "conversion"


def _statistics():
    return {
        "channels": 4,
        "value_min": torch.full((4,), -1.0),
        "value_max": torch.full((4,), 1.0),
        "saliency_row_count": torch.ones(4, dtype=torch.long),
        "saliency_sum": torch.arange(4, dtype=torch.float64),
        "phase_ema_abs_max": torch.ones(4),
        "phase_ema_updates": torch.ones(4, dtype=torch.long),
        "phase_tau_statistic": "spikingllm_ema_channel_abs_max",
        "phase_tau_ema_factor": 0.99,
    }


def _cfg(rotation_enabled=False):
    return {
        "experiment": {"name": "test_conversion", "ann_mode": "vanilla"},
        "ann_finetuning": {"mode": "vanilla"},
        "rotation": {"enabled": rotation_enabled},
        "post_finetuning": {"prefix_enabled": False},
        "calibration": {"group_size": -1, "expected_sites_per_layer": 10},
        "phase": {"T": 4, "base": 2.0, "surrogate_slope": 4.0, "max_spikes": 4},
        "gif": {"base_bits": 4, "add_bits": 1, "low_ratio": 0.5},
        "mtn": {"T": 4, "K": 6, "threshold_factor": 0.75},
    }


def _prepare(tmp_path, *, rotation_enabled=False):
    layout = _Layout(tmp_path)
    for index in SITE_IDS:
        site = layout.post_finetuning_site_dir / "layer_000" / f"site_{index:02d}_{SITE_NAMES[index]}"
        site.mkdir(parents=True)
        torch.save(_statistics(), site / "statistics.pt")
    global_directory = layout.post_finetuning_site_dir / "_global" / "final_rmsnorm"
    global_directory.mkdir(parents=True)
    torch.save(_statistics(), global_directory / "statistics.pt")
    materialize_calibration_states(
        layout.post_finetuning_site_dir,
        _cfg(rotation_enabled),
        {
            "purpose": "post_finetuning_conversion_calibration",
            "eligible_for_conversion": True,
            "post_finetuning_recalibration": True,
            "state_profile": "snn_conversion_without_clip",
            "common_clip_required": False,
            "prefix_enabled": False,
        },
        include_clip=False,
        expected_num_hidden_layers=1,
    )
    layout.ann_checkpoint_dir.mkdir(parents=True)
    ann_config = layout.ann_checkpoint_dir / "config.json"
    ann_config.write_text('{"num_hidden_layers": 1}' + '\n', encoding="utf-8")
    output = layout.snn_conversion_dir("gif")
    output.mkdir(parents=True)
    rotation_path = layout.rotation_dir / "rotation_state.pt"
    if rotation_enabled:
        rotation_path.parent.mkdir(parents=True)
        rotation_path.write_bytes(b"rotation-v1")
    manifest = layout.post_finetuning_site_dir / "calibration_state_manifest.json"
    metadata = {
        "format_version": CONVERSION_METADATA_FORMAT_VERSION,
        "deployment_neuron": "gif",
        "full_temporal_steps": GIF_LOCAL_STEPS,
        "source_ann_checkpoint": str(layout.ann_checkpoint_dir.resolve()),
        "source_ann_config_sha256": sha256_file(ann_config),
        "post_finetuning_recalibration": True,
        "rotation_enabled": rotation_enabled,
        "rotation_state_sha256": (
            sha256_file(rotation_path) if rotation_enabled else None
        ),
        "expected_num_hidden_layers": 1,
        "prefix_enabled": False,
        "prefix_token_ids": [],
        "prefix_state_sha256": None,
        "prefix_kv_sha256": None,
        "calibration_root": str(layout.post_finetuning_site_dir.resolve()),
        "calibration_state_manifest_sha256": sha256_file(manifest),
        "calibration_source_stage": "post_finetuning",
        "prefix_source_stage": "post_finetuning",
        "reused_ann_training_artifacts": False,
        "snn_clip_applied": False,
        "gif_local_decomposition_steps": GIF_LOCAL_STEPS,
        **temporal_policy_metadata(),
    }
    path = output / "conversion_metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return layout, path


def test_conversion_metadata_v5_is_accepted(tmp_path):
    layout, _ = _prepare(tmp_path)
    metadata = validate_conversion_metadata(_cfg(), layout, "gif")
    assert metadata["gif_high_qmax"] == 30


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("format_version", 3),
        ("gif_high_qmax", 31),
        ("full_temporal_steps", 3),
        ("snn_clip_applied", True),
    ],
)
def test_conversion_metadata_rejects_legacy_or_mismatched_policy(tmp_path, key, value):
    layout, path = _prepare(tmp_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata[key] = value
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_conversion_metadata(_cfg(), layout, "gif")


def test_conversion_rotation_hash_matches_current_file(tmp_path):
    layout, _ = _prepare(tmp_path, rotation_enabled=True)
    metadata = validate_conversion_metadata(_cfg(True), layout, "gif")
    assert metadata["rotation_state_sha256"] == sha256_file(
        layout.rotation_dir / "rotation_state.pt"
    )


def test_conversion_rejects_modified_rotation_file(tmp_path):
    layout, _ = _prepare(tmp_path, rotation_enabled=True)
    (layout.rotation_dir / "rotation_state.pt").write_bytes(b"rotation-v2")
    with pytest.raises(ValueError, match="rotation_state_sha256"):
        validate_conversion_metadata(_cfg(True), layout, "gif")


def test_conversion_rejects_tampered_rotation_hash(tmp_path):
    layout, path = _prepare(tmp_path, rotation_enabled=True)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["rotation_state_sha256"] = "0" * 64
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="rotation_state_sha256"):
        validate_conversion_metadata(_cfg(True), layout, "gif")


def test_conversion_disabled_rotation_requires_null_hash(tmp_path):
    layout, path = _prepare(tmp_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["rotation_state_sha256"] = "0" * 64
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="rotation_state_sha256"):
        validate_conversion_metadata(_cfg(), layout, "gif")


def test_conversion_rejects_ann_manifest_layer_mismatch(tmp_path):
    layout, _ = _prepare(tmp_path)
    (layout.ann_checkpoint_dir / "config.json").write_text(
        json.dumps({"num_hidden_layers": 2}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="ANN config num_hidden_layers"):
        validate_conversion_metadata(_cfg(), layout, "gif")


def test_conversion_rejects_stale_post_finetuning_clip_state(tmp_path):
    layout, _ = _prepare(tmp_path)
    stale = next(layout.post_finetuning_site_dir.glob("layer_*/site_*"))
    torch.save({}, stale / "clip_state.pt")
    with pytest.raises(ValueError, match="clip-free"):
        validate_conversion_metadata(_cfg(), layout, "gif")
