import json

import pytest
import torch

from snn2.conversion import validate_calibration
from snn2.sites import SITE_IDS, SITE_NAMES, topology_metadata


_REQUIRED = ("statistics.pt", "phase_state.pt", "gif_state.pt", "mtn_state.pt", "clip_state.pt")


def _write_site(root, index, name):
    directory = root / "layer_000" / f"site_{index:02d}_{name}"
    directory.mkdir(parents=True)
    for filename in _REQUIRED:
        if filename == "clip_state.pt":
            torch.save({"lower": torch.tensor([0.0]), "upper": torch.tensor([1.0])}, directory / filename)
        else:
            torch.save({}, directory / filename)


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
