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
    }


def _cfg():
    return {
        "rotation": {"enabled": False},
        "post_finetuning": {"prefix_enabled": False},
        "calibration": {"group_size": -1, "expected_sites_per_layer": 10},
        "phase": {"T": 4, "base": 2.0, "surrogate_slope": 4.0, "max_spikes": 4},
        "gif": {"base_bits": 4, "add_bits": 1, "low_ratio": 0.5},
        "mtn": {"T": 4, "K": 6, "threshold_factor": 0.75},
    }


def _prepare(tmp_path):
    layout = _Layout(tmp_path)
    for index in SITE_IDS:
        site = layout.post_finetuning_site_dir / "layer_000" / f"site_{index:02d}_{SITE_NAMES[index]}"
        site.mkdir(parents=True)
        torch.save(_statistics(), site / "statistics.pt")
    materialize_calibration_states(layout.post_finetuning_site_dir, _cfg(), include_clip=True)
    layout.ann_checkpoint_dir.mkdir(parents=True)
    ann_config = layout.ann_checkpoint_dir / "config.json"
    ann_config.write_text("{}\n", encoding="utf-8")
    output = layout.snn_conversion_dir("gif")
    output.mkdir(parents=True)
    manifest = layout.post_finetuning_site_dir / "calibration_state_manifest.json"
    metadata = {
        "format_version": CONVERSION_METADATA_FORMAT_VERSION,
        "deployment_neuron": "gif",
        "full_temporal_steps": GIF_LOCAL_STEPS,
        "source_ann_checkpoint": str(layout.ann_checkpoint_dir.resolve()),
        "source_ann_config_sha256": sha256_file(ann_config),
        "post_finetuning_recalibration": True,
        "rotation_enabled": False,
        "prefix_enabled": False,
        "prefix_token_ids": [],
        "prefix_state_sha256": None,
        "prefix_kv_sha256": None,
        "calibration_root": str(layout.post_finetuning_site_dir.resolve()),
        "calibration_state_manifest_sha256": sha256_file(manifest),
        "common_clip_applied": True,
        "gif_local_decomposition_steps": GIF_LOCAL_STEPS,
        **temporal_policy_metadata(),
    }
    path = output / "conversion_metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return layout, path


def test_conversion_metadata_v2_is_accepted(tmp_path):
    layout, _ = _prepare(tmp_path)
    metadata = validate_conversion_metadata(_cfg(), layout, "gif")
    assert metadata["gif_high_qmax"] == 30


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("format_version", 1),
        ("gif_high_qmax", 31),
        ("full_temporal_steps", 3),
        ("common_clip_applied", False),
    ],
)
def test_conversion_metadata_rejects_legacy_or_mismatched_policy(tmp_path, key, value):
    layout, path = _prepare(tmp_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata[key] = value
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_conversion_metadata(_cfg(), layout, "gif")
