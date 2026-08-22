import json

import pytest
import torch

from snn2.calibration import materialize_calibration_states
from snn2.conversion import validate_calibration
from snn2.sites import SITE_IDS, SITE_NAMES, topology_metadata


_REQUIRED = ("statistics.pt",)

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
        "calibration": {"group_size": -1, "expected_sites_per_layer": 10},
        "phase": {"T": 4, "base": 2.0, "surrogate_slope": 4.0, "max_spikes": 4},
        "gif": {"base_bits": 4, "add_bits": 1, "low_ratio": 0.5},
        "mtn": {"T": 4, "K": 6, "threshold_factor": 0.75},
    }


def _write_site(root, index, name):
    directory = root / "layer_000" / f"site_{index:02d}_{name}"
    directory.mkdir(parents=True)
    for filename in _REQUIRED:
        torch.save(_statistics(), directory / filename)
    return directory


def test_validate_calibration_rejects_legacy_site_nine(tmp_path):
    _write_site(tmp_path, 9, "post_mlp_product_r4")
    with pytest.raises(RuntimeError, match="topology"):
        validate_calibration(tmp_path)


def test_validate_calibration_requires_exact_current_topology(tmp_path):
    for index in SITE_IDS:
        _write_site(tmp_path, index, SITE_NAMES[index])
    materialize_calibration_states(tmp_path, _cfg(), include_clip=True)
    metadata = validate_calibration(tmp_path)
    assert metadata["site_count"] == 10
    assert metadata["sites"] == 10


def test_validate_calibration_requires_common_clip_state(tmp_path):
    directories = [
        _write_site(tmp_path, index, SITE_NAMES[index]) for index in SITE_IDS
    ]
    materialize_calibration_states(tmp_path, _cfg(), include_clip=True)
    (directories[0] / "clip_state.pt").unlink()

    with pytest.raises(FileNotFoundError, match="clip_state.pt"):
        validate_calibration(tmp_path)

def test_validate_calibration_rejects_legacy_manifest(tmp_path):
    for index in SITE_IDS:
        _write_site(tmp_path, index, SITE_NAMES[index])
    materialize_calibration_states(tmp_path, _cfg(), include_clip=True)
    manifest = tmp_path / "calibration_state_manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["format_version"] = 1
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="legacy calibration manifest"):
        validate_calibration(tmp_path)
