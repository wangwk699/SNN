import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from snn2.artifacts import ArtifactLayout
from snn2.config import (
    resolve_config,
    rotated_pre_finetuning_prefix_enabled,
)
from snn2.modeling import (
    model_source_for_stage,
    prefix_ids_for_stage,
    prefix_key_values_for_stage,
)


def _cfg(mode="unaware", learning_rate=5e-6):
    return {
        "experiment": {
            "id": "exp",
            "task": "tldr",
            "model_name": "model/name",
            "seed": 42,
            "output_root": "artifacts",
            "ann_mode": mode,
        },
        "training": {"learning_rate": learning_rate},
        "rotation": {"enabled": mode != "vanilla"},
        "prefix": {"enabled": mode != "vanilla"},
        "phase": {"surrogate_slope": 1.0},
        "post_finetuning": {"prefix_enabled": True},
    }


@pytest.mark.parametrize(
    ("mode", "learning_rate"),
    [("unaware", 5e-6), ("phase_aware", 1e-6), ("gif_aware", 2e-6)],
)
def test_rotated_pre_finetuning_paths_are_model_shared(mode, learning_rate):
    reference = ArtifactLayout(_cfg())
    layout = ArtifactLayout(_cfg(mode, learning_rate))

    assert layout.rotated_pre_finetuning_dir == reference.rotated_pre_finetuning_dir
    assert layout.rotated_pre_finetuning_dir.parts[-5:] == (
        "model_name", "_shared", "seed42", "rotated_prefix",
        "rotated_pre_finetuning",
    )
    assert layout.rotated_pre_finetuning_config_dir.parent == layout.rotated_pre_finetuning_dir
    assert layout.rotated_pre_finetuning_logs_dir.parent == layout.rotated_pre_finetuning_dir
    assert layout.rotated_pre_finetuning_prefix_dir == layout.ann_training_prefix_dir
    assert layout.ann_training_prefix_dir.parts[-2:] == (
        "rotated_prefix", "pre_finetuning_prefix"
    )


def test_rotated_pre_finetuning_stage_uses_shared_pre_finetuning_prefix(tmp_path):
    cfg = _cfg()
    cfg["experiment"]["output_root"] = str(tmp_path)
    layout = ArtifactLayout(cfg)

    assert model_source_for_stage(cfg, layout, stage="rotated_pre_finetuning") == str(
        layout.rotation_dir / "fused_base"
    )
    with pytest.raises(FileNotFoundError):
        prefix_ids_for_stage(cfg, layout, stage="rotated_pre_finetuning")

    layout.ann_training_prefix_dir.mkdir(parents=True)
    (layout.ann_training_prefix_dir / "prefix_state.json").write_text(
        json.dumps({"prefix_token_ids": [99]}), encoding="utf-8"
    )
    layout.post_finetuning_prefix_dir.mkdir(parents=True)
    (layout.post_finetuning_prefix_dir / "prefix_state.json").write_text(
        json.dumps({"prefix_token_ids": [88]}), encoding="utf-8"
    )
    assert prefix_ids_for_stage(cfg, layout, stage="rotated_pre_finetuning") == [99]
    with pytest.raises(FileNotFoundError, match="fixed KV cache"):
        prefix_key_values_for_stage(cfg, layout, stage="rotated_pre_finetuning")

    cache_path = layout.rotated_pre_finetuning_prefix_dir / "prefixed_key_values.pt"
    expected = ((torch.ones((1, 1, 1, 1)), torch.ones((1, 1, 1, 1))),)
    torch.save(expected, cache_path)
    actual = prefix_key_values_for_stage(cfg, layout, stage="rotated_pre_finetuning")
    assert len(actual) == 1
    torch.testing.assert_close(actual[0][0], expected[0][0])


def test_empty_rotated_pre_finetuning_prefix_needs_no_kv(tmp_path):
    cfg = _cfg()
    cfg["experiment"]["output_root"] = str(tmp_path)
    layout = ArtifactLayout(cfg)
    state = layout.rotated_pre_finetuning_prefix_dir / "prefix_state.json"
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({"prefix_token_ids": []}), encoding="utf-8")

    assert prefix_ids_for_stage(cfg, layout, stage="rotated_pre_finetuning") == []
    assert prefix_key_values_for_stage(cfg, layout, stage="rotated_pre_finetuning") is None


def test_disabled_rotated_pre_finetuning_prefix_ignores_existing_artifacts(tmp_path):
    cfg = _cfg()
    cfg["experiment"]["output_root"] = str(tmp_path)
    cfg["rotated_pre_finetuning"] = {"prefix_enabled": False}
    layout = ArtifactLayout(cfg)
    state = layout.rotated_pre_finetuning_prefix_dir / "prefix_state.json"
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({"prefix_token_ids": [7]}), encoding="utf-8")
    torch.save(
        ((torch.ones((1, 1, 1, 1)), torch.ones((1, 1, 1, 1))),),
        layout.rotated_pre_finetuning_prefix_dir / "prefixed_key_values.pt",
    )

    assert prefix_ids_for_stage(cfg, layout, stage="rotated_pre_finetuning") == []
    assert prefix_key_values_for_stage(cfg, layout, stage="rotated_pre_finetuning") is None


def test_rotated_evaluation_dependencies_are_isolated(tmp_path):
    scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from evaluate_tldr import _validate_rotated_pre_finetuning_dependencies

    cfg = _cfg()
    cfg["experiment"]["output_root"] = str(tmp_path)
    layout = ArtifactLayout(cfg)
    with pytest.raises(FileNotFoundError, match="prepare_rotation"):
        _validate_rotated_pre_finetuning_dependencies(cfg, layout)

    fused_config = layout.rotation_dir / "fused_base" / "config.json"
    fused_config.parent.mkdir(parents=True)
    fused_config.write_text("{}", encoding="utf-8")
    (layout.rotation_dir / "rotation_state.pt").write_bytes(b"state")
    prefix_state = layout.rotated_pre_finetuning_prefix_dir / "prefix_state.json"
    cfg["rotated_pre_finetuning"] = {"prefix_enabled": False}
    _validate_rotated_pre_finetuning_dependencies(cfg, layout)
    cfg["rotated_pre_finetuning"]["prefix_enabled"] = True

    prefix_state.parent.mkdir(parents=True)
    prefix_state.write_text(json.dumps({"prefix_token_ids": []}), encoding="utf-8")
    _validate_rotated_pre_finetuning_dependencies(cfg, layout)

    prefix_state.write_text(json.dumps({"prefix_token_ids": [1]}), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="fixed KV cache"):
        _validate_rotated_pre_finetuning_dependencies(cfg, layout)

    cfg["rotation"]["enabled"] = False
    with pytest.raises(ValueError, match="rotation.enabled"):
        _validate_rotated_pre_finetuning_dependencies(cfg, layout)


def test_rotated_pre_finetuning_prefix_flag_is_independent_from_ann_training():
    cfg = _cfg("unaware")
    cfg["replacement"] = {"train_mode": "none"}
    cfg["rotated_pre_finetuning"] = {"prefix_enabled": False}

    resolved = resolve_config(cfg)

    assert resolved["prefix"]["enabled"] is True
    assert not rotated_pre_finetuning_prefix_enabled(resolved)


def test_base_and_rotated_pre_finetuning_flags_are_mutually_exclusive():
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_tldr.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config", "unused.yaml",
            "--neuron", "ann",
            "--base",
            "--rotated-pre-finetuning",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "not allowed with argument" in result.stderr
