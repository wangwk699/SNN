import json
from types import SimpleNamespace

import pytest

from snn2.training import (
    capture_training_artifact_provenance,
    format_runtime_hms,
    verify_training_artifact_provenance_unchanged,
)


def test_format_runtime_hms() -> None:
    assert format_runtime_hms(30550.7217) == "08:29:10.7217"


def test_format_runtime_hms_does_not_wrap_after_24_hours() -> None:
    assert format_runtime_hms(90061.5) == "25:01:01.5000"


def test_format_runtime_hms_carries_fractional_second_rounding() -> None:
    assert format_runtime_hms(3599.99996) == "01:00:00.0000"


def _provenance_fixture(tmp_path):
    prefix_dir = tmp_path / "prefix"
    site_dir = tmp_path / "sites"
    prefix_dir.mkdir()
    site_dir.mkdir()
    (prefix_dir / "prefix_state.json").write_text(
        json.dumps({"prefix_token_ids": [7, 8]}), encoding="utf-8"
    )
    (prefix_dir / "prefixed_key_values.pt").write_bytes(b"kv-v1")
    (site_dir / "calibration_state_manifest.json").write_text(
        json.dumps({"format_version": 6}), encoding="utf-8"
    )
    cfg = {
        "experiment": {"ann_mode": "phase_aware"},
        "ann_training": {"prefix_enabled": True},
        "prefix": {"enabled": True},
    }
    layout = SimpleNamespace(
        ann_training_prefix_dir=prefix_dir,
        ann_training_site_dir=site_dir,
    )
    return cfg, layout


def test_training_artifact_provenance_capture_and_verify(monkeypatch, tmp_path):
    monkeypatch.setattr("snn2.training.validate_site_state_bundle", lambda *args, **kwargs: {})
    cfg, layout = _provenance_fixture(tmp_path)
    captured = capture_training_artifact_provenance(
        cfg, layout, prefix_ids=[7, 8]
    )
    verify_training_artifact_provenance_unchanged(captured, cfg, layout)
    assert captured["ann_training_prefix_token_ids"] == [7, 8]
    assert captured["ann_training_prefix_state_sha256"]
    assert captured["ann_training_prefix_kv_sha256"]
    assert captured["ann_training_calibration_manifest_sha256"]


@pytest.mark.parametrize(
    ("relative_path", "replacement"),
    [
        ("prefix/prefix_state.json", b"{\"prefix_token_ids\": [7, 8], \"changed\": true}"),
        ("prefix/prefixed_key_values.pt", b"kv-v2"),
        ("sites/calibration_state_manifest.json", b"{\"format_version\": 6, \"changed\": true}"),
    ],
)
def test_training_artifact_provenance_rejects_mid_training_changes(
    monkeypatch, tmp_path, relative_path, replacement
):
    monkeypatch.setattr("snn2.training.validate_site_state_bundle", lambda *args, **kwargs: {})
    cfg, layout = _provenance_fixture(tmp_path)
    captured = capture_training_artifact_provenance(
        cfg, layout, prefix_ids=[7, 8]
    )
    (tmp_path / relative_path).write_bytes(replacement)
    with pytest.raises(RuntimeError, match="changed during training"):
        verify_training_artifact_provenance_unchanged(captured, cfg, layout)
