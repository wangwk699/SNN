import json

import pytest
import torch

from snn2.conversion import validate_calibration
from snn2.sites import SITE_IDS, SITE_NAMES, topology_metadata


_REQUIRED = ("statistics.pt", "phase_state.pt", "gif_state.pt", "mtn_state.pt")


def _write_site(root, index, name):
    directory = root / "layer_000" / f"site_{index:02d}_{name}"
    directory.mkdir(parents=True)
    for filename in _REQUIRED:
        torch.save({}, directory / filename)
    return directory


def test_validate_calibration_rejects_legacy_site_nine(tmp_path):
    _write_site(tmp_path, 9, "post_mlp_product_r4")
    with pytest.raises(RuntimeError, match="topology"):
        validate_calibration(tmp_path)


def test_validate_calibration_requires_exact_current_topology(tmp_path):
    for index in SITE_IDS:
        _write_site(tmp_path, index, SITE_NAMES[index])
    (tmp_path / "calibration_state_manifest.json").write_text(
        json.dumps({"format_version": 1, **topology_metadata()}), encoding="utf-8"
    )
    metadata = validate_calibration(tmp_path)
    assert metadata["site_count"] == 10
    assert metadata["sites"] == 10


def test_validate_calibration_rejects_stale_clip_state(tmp_path):
    directories = [
        _write_site(tmp_path, index, SITE_NAMES[index]) for index in SITE_IDS
    ]
    torch.save({}, directories[0] / "clip_state.pt")

    with pytest.raises(ValueError, match="must not contain.*clip_state.pt"):
        validate_calibration(tmp_path)
