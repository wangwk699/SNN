from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from snn2.artifacts import ArtifactLayout, read_json, write_json
from snn2.calibration import calibration_provenance, materialize_calibration_states, materialize_clip_profile
from snn2.config import (load_config, resolve_config, validate_config, requires_pre_finetuning_prefix,
                         requires_ann_training_calibration, calibration_trajectory_config)
from snn2.conversion import create_conversion, validate_conversion_metadata
from snn2.evaluation import build_evaluation_controller, evaluation_requires_integration
from snn2.model_integration import install_model_integration
from snn2.modeling import (model_source_for_stage, rotation_state, prefix_ids_for_stage,
                           prefix_key_values_for_stage)
from snn2.training import (capture_training_artifact_provenance, validate_recorded_training_artifact_provenance,
                          train_full_parameters)
from test_calibration_profiles import _write_statistics
from test_verify_artifacts import _VERIFY

ROOT = Path(__file__).resolve().parents[1]
ABLATION_ID = "snn2_ablation_no_rotation_and_aware_FT_v1"
CONFIGS = sorted((ROOT / "configs/ablations").glob("*.yaml"))
CHANGES = {
    ("experiment", "id"): ABLATION_ID,
    ("rotation", "enabled"): False,
    ("rotation", "fused_weights_are_finetuned"): False,
    ("prefix", "enabled"): False,
    ("ann_training", "prefix_enabled"): False,
    ("rotated_pre_finetuning", "prefix_enabled"): False,
    ("post_finetuning", "prefix_enabled"): False,
    ("post_finetuning", "rediscover_prefix"): False,
    ("evaluation", "prefix_enabled"): False,
}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_complete_ablation_configs_preserve_all_other_source_fields(path):
    assert len(CONFIGS) == 6
    raw = yaml.safe_load(path.read_text())
    source = ROOT / "configs/generated" / path.name.replace("_no_rotation", "")
    expected = yaml.safe_load(source.read_text())
    for (section, key), value in CHANGES.items():
        expected[section][key] = value
    assert raw == expected
    cfg = load_config(path)
    validate_config(resolve_config(cfg))
    assert cfg["replacement"]["train_mode"] == cfg["experiment"]["ann_mode"].removesuffix("_aware")
    assert not requires_pre_finetuning_prefix(cfg)
    assert requires_ann_training_calibration(cfg)
    layout = ArtifactLayout(cfg)
    for name in ("root", "shared_task_root", "shared_model_root", "policy_root", "data_dir",
                 "ann_training_site_dir", "ann_training_clip_profile_dir", "ann_checkpoint_dir",
                 "post_finetuning_site_dir", "rotation_dir", "conversion_site_dir"):
        assert Path(getattr(layout, name)).is_relative_to(Path("artifacts") / ABLATION_ID)
    for neuron in ("phase", "gif", "mtn"):
        assert layout.snn_conversion_dir(neuron).is_relative_to(Path("artifacts") / ABLATION_ID)
    for stage in ("ann_training", "post_finetuning", "final_ann_evaluation", "final_snn_evaluation"):
        assert prefix_ids_for_stage(cfg, layout, stage=stage) == []
        assert prefix_key_values_for_stage(cfg, layout, stage=stage) is None
    assert rotation_state(cfg, layout) is None
    assert model_source_for_stage(cfg, layout, stage="ann_training") == cfg["experiment"]["model_name"]


@pytest.mark.parametrize("path", CONFIGS)
@pytest.mark.parametrize("section,key,value", [
    ("rotation", "enabled", True), ("ann_training", "prefix_enabled", True),
    ("prefix", "enabled", True), ("post_finetuning", "prefix_enabled", True),
    ("evaluation", "prefix_enabled", True), ("rotated_pre_finetuning", "prefix_enabled", True),
    ("post_finetuning", "rediscover_prefix", True), ("rotation", "fused_weights_are_finetuned", True),
    ("rotation", "enabled", "false"), ("ann_training", "prefix_enabled", "false"),
    ("prefix", "enabled", 0),
])
def test_ablation_rejects_mixed_components_and_non_boolean_switches(path, section, key, value):
    cfg = yaml.safe_load(path.read_text())
    cfg[section][key] = value
    with pytest.raises(ValueError):
        validate_config(resolve_config(cfg))


def test_rotated_aware_rejects_training_prefix_disabled():
    cfg = load_config(ROOT / "configs/generated/exp1_qwen3_1_7b_tldr__phase_aware.yaml")
    cfg["ann_training"]["prefix_enabled"] = False
    cfg["prefix"]["enabled"] = False
    with pytest.raises(ValueError, match="jointly"):
        validate_config(resolve_config(cfg))


def test_stage_a_shares_only_matching_trajectory():
    cfg = load_config(CONFIGS[0])
    other = copy.deepcopy(cfg)
    other["experiment"]["ann_mode"] = "phase_aware" if cfg["experiment"]["ann_mode"] == "gif_aware" else "gif_aware"
    assert ArtifactLayout(cfg).ann_training_site_dir == ArtifactLayout(other).ann_training_site_dir
    other["calibration"]["phase_previous_layers_snn"] = not cfg["calibration"]["phase_previous_layers_snn"]
    assert ArtifactLayout(cfg).ann_training_site_dir != ArtifactLayout(other).ann_training_site_dir


@pytest.fixture(params=["phase_aware", "gif_aware"])
def bundle(tmp_path, request):
    cfg = load_config(ROOT / f"configs/ablations/exp1_qwen3_1_7b_tldr__{request.param}_no_rotation.yaml")
    cfg["experiment"]["output_root"] = str(tmp_path / "artifacts")
    cfg["calibration"].update(group_size=-1, phase_previous_layers_snn=False,
                               gif_previous_layers_snn=False, mtn_previous_layers_snn=False)
    cfg["training"]["deepspeed_config"] = "configs/deepspeed_zero3.json"
    layout = ArtifactLayout(cfg)
    layout.ensure()
    write_json(layout.calibration_data_manifest_path, {"fixture": True})
    _write_statistics(layout.ann_training_site_dir)
    manifest = materialize_calibration_states(
        layout.ann_training_site_dir, cfg, calibration_provenance(cfg, layout, stage="ann_training"),
        expected_num_hidden_layers=1,
    )
    materialize_clip_profile(layout.ann_training_site_dir, layout.ann_training_clip_profile_dir, cfg)
    checkpoint = {"num_hidden_layers": 1, "hidden_size": 4, "num_attention_heads": 1,
                  "snn2_phase_T": cfg["phase"]["T"], "snn2_phase_base": cfg["phase"]["base"],
                  "snn2_rotation_enabled": False, "snn2_fused_weights_are_trainable": False}
    write_json(layout.ann_checkpoint_dir / "config.json", checkpoint)
    result = capture_training_artifact_provenance(cfg, layout, prefix_ids=[])
    result.update(prefix_enabled=False, fused_rotation_weights_trained=False)
    write_json(layout.ann_dir / "training_result.json", result)
    return cfg, layout, manifest


def test_stage_a_b_no_prefix_lineage_and_verifier(bundle):
    cfg, layout, manifest = bundle
    assert manifest["source_model_stage"] == "original_pretrained_base"
    assert manifest["rotation_enabled"] is False
    assert manifest["prefix_enabled"] is False
    assert manifest["prefix_token_ids"] == []
    for key in ("rotation_state_path", "rotation_state_sha256", "prefix_state_path", "prefix_state_sha256",
                "prefix_kv_path", "prefix_kv_sha256"):
        assert manifest[key] is None
    assert not list(layout.ann_training_site_dir.glob("layer_*/site_*/clip_state.pt"))
    _VERIFY._verify_disabled_components(cfg, layout)
    _VERIFY._validate_aware_final_ann_training_provenance(cfg, layout)
    for root, name in [(layout.rotation_dir, "rotation_state.pt"),
                       (layout.ann_training_prefix_dir, "prefix_state.json"),
                       (layout.post_finetuning_prefix_dir, "prefixed_key_values.pt")]:
        path = root / name
        path.write_bytes(b"unexpected")
        with pytest.raises(ValueError, match="must be absent"):
            _VERIFY._verify_disabled_components(cfg, layout)
        path.unlink()


@pytest.mark.parametrize("use_post", [False, True])
@pytest.mark.parametrize("neuron", ["phase", "gif", "mtn"])
def test_no_prefix_conversion_both_selectors_end_to_end(bundle, use_post, neuron):
    cfg, layout, _ = bundle
    cfg["conversion"]["use_post_finetuning_artifacts"] = use_post
    layout = ArtifactLayout(cfg)
    if use_post:
        _write_statistics(layout.post_finetuning_site_dir)
        materialize_calibration_states(
            layout.post_finetuning_site_dir, cfg, calibration_provenance(cfg, layout, stage="post_finetuning"),
            expected_num_hidden_layers=1,
        )
    metadata = create_conversion(cfg, layout, neuron)
    assert validate_conversion_metadata(cfg, layout, neuron) == metadata
    assert metadata["rotation_enabled"] is False
    assert metadata["prefix_enabled"] is False
    assert metadata["prefix_token_ids"] == []
    assert metadata["prefix_root"] is None
    assert metadata["rotation_state_sha256"] is None
    assert metadata["prefix_state_sha256"] is None
    assert metadata["prefix_kv_sha256"] is None
    assert metadata["snn_clip_applied"] is False
    assert metadata["calibration_source_stage"] == ("post_finetuning" if use_post else "ann_training")


@pytest.mark.parametrize("mutation", ["missing_A", "missing_B", "changed_A", "changed_B", "old_rotation",
                                      "old_id", "old_data_id", "data_hash", "prefix_ids", "trajectory"])
def test_no_prefix_conversion_preserves_strict_frozen_checks(bundle, mutation):
    cfg, layout, _ = bundle
    stage_a = layout.ann_training_site_dir / "calibration_state_manifest.json"
    result_path = layout.ann_dir / "training_result.json"
    if mutation == "missing_A":
        stage_a.unlink()
    elif mutation == "missing_B":
        (layout.ann_training_clip_profile_dir / "clip_profile_manifest.json").unlink()
    elif mutation == "changed_A":
        payload = read_json(stage_a); payload["tampered"] = True; write_json(stage_a, payload)
    elif mutation == "changed_B":
        (layout.ann_training_clip_profile_dir / "clip_profile_manifest.json").write_text("{}")
    elif mutation == "old_rotation":
        payload = read_json(stage_a); payload["rotation_enabled"] = True; write_json(stage_a, payload)
    elif mutation in {"old_data_id", "data_hash"}:
        payload = read_json(stage_a)
        if mutation == "old_data_id":
            payload["calibration_data_manifest_path"] = payload["calibration_data_manifest_path"].replace(ABLATION_ID, "snn2_main_v1")
        else:
            payload["calibration_data_manifest_sha256"] = "0" * 64
        write_json(stage_a, payload)
    else:
        result = read_json(result_path)
        if mutation == "old_id":
            result["ann_training_calibration_root"] = result["ann_training_calibration_root"].replace(ABLATION_ID, "snn2_main_v1")
        elif mutation == "prefix_ids":
            result["ann_training_prefix_token_ids"] = [123]
        else:
            result["ann_training_calibration_trajectory_signature"] = {}
        write_json(result_path, result)
    with pytest.raises((ValueError, FileNotFoundError, RuntimeError)):
        create_conversion(cfg, layout, "gif")


def test_absent_prefix_fields_are_valid_but_effective_prefix_is_rejected(bundle):
    cfg, layout, _ = bundle
    path = layout.ann_dir / "training_result.json"
    result = read_json(path)
    for key in list(result):
        if key.startswith("ann_training_prefix_"):
            del result[key]
    write_json(path, result)
    validate_recorded_training_artifact_provenance(cfg, layout)
    create_conversion(cfg, layout, "gif")
    result["prefix_enabled"] = True
    write_json(path, result)
    with pytest.raises(ValueError, match="prefix_enabled"):
        create_conversion(cfg, layout, "gif")


def _tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(vocab_size=16, hidden_size=4, intermediate_size=4, num_hidden_layers=1,
                         num_attention_heads=1, num_key_value_heads=1, max_position_embeddings=16,
                         attention_dropout=0.0, pad_token_id=0)
    return LlamaForCausalLM(config)


def test_final_aware_ann_no_rotation_real_forward_backward(bundle):
    cfg, layout, _ = bundle
    model = _tiny_model()
    controller, steps = build_evaluation_controller(cfg, layout, neuron="ann")
    assert steps == 1
    assert controller.mode == cfg["replacement"]["train_mode"]
    assert evaluation_requires_integration(cfg, neuron="ann")
    assert not evaluation_requires_integration(cfg, neuron="ann", base=True)
    assert not evaluation_requires_integration(cfg, neuron="ann", rotated_pre_finetuning=True)
    install_model_integration(model, controller, rotation_state(cfg, layout))
    assert model.model.layers[0].self_attn._snn2_r3 is None
    seen = set()
    apply = controller.apply
    def record(layer, site, value, **kwargs):
        seen.add(site)
        return apply(layer, site, value, **kwargs)
    controller.apply = record
    logits = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False).logits
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    assert seen == set(range(1, 11))
    assert any(p.grad is not None for p in model.parameters())
    for neuron in ("phase", "gif", "mtn"):
        assert evaluation_requires_integration(cfg, neuron=neuron)


def test_training_no_rotation_keeps_aware_integration_and_metadata(bundle, monkeypatch):
    cfg, layout, _ = bundle
    model = _tiny_model()
    tokenizer = SimpleNamespace(chat_template="", save_pretrained=lambda path: None)
    monkeypatch.setattr("snn2.training.load_model", lambda *args, **kwargs: model)
    monkeypatch.setattr("snn2.training.load_tokenizer", lambda *args: tokenizer)
    count = cfg["training"]["tldr_train_samples"]
    monkeypatch.setattr("snn2.training.load_selected_raw", lambda *args: SimpleNamespace(
        train=list(range(count)), validation=[1], manifests={"train": {}}))
    monkeypatch.setattr("snn2.training.tokenize_dataset", lambda raw, *args, **kwargs: raw)
    monkeypatch.setattr("snn2.training.CausalLMCollator", lambda *args: None)
    class Arguments:
        def __init__(self, **kwargs): pass
        def main_process_first(self, **kwargs):
            from contextlib import nullcontext
            return nullcontext()
    class Trainer:
        def __init__(self, **kwargs):
            assert all(p.requires_grad for p in model.parameters())
            assert model.model.layers[0].self_attn._snn2_controller.mode == cfg["replacement"]["train_mode"]
            assert model.model.layers[0].self_attn._snn2_r3 is None
            self.state = SimpleNamespace(best_model_checkpoint=None, best_metric=None, log_history=[])
        def train(self, **kwargs):
            return SimpleNamespace(metrics={"train_runtime": 0.01})
        def save_model(self, path):
            write_json(Path(path) / "config.json", model.config.to_dict())
        def is_world_process_zero(self): return True
    monkeypatch.setattr("transformers.TrainingArguments", Arguments)
    monkeypatch.setattr("transformers.Trainer", Trainer)
    result = train_full_parameters(cfg, layout)
    assert result["fused_rotation_weights_trained"] is False
    assert result["prefix_enabled"] is False
    assert model.config.snn2_ann_mode == cfg["experiment"]["ann_mode"]
    assert model.config.snn2_fused_weights_are_trainable is False
    assert model.config.snn2_rotation_enabled is False
    _VERIFY._verify_disabled_components(cfg, layout)


@pytest.mark.parametrize("stage", ["pre_finetuning", "post_finetuning"])
def test_disabled_prefix_discovery_refuses_before_model_load(bundle, monkeypatch, stage):
    cfg, layout, _ = bundle
    spec = importlib.util.spec_from_file_location("ablation_discover_prefix", ROOT / "scripts/discover_prefix.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module, "setup", lambda *args, **kwargs: (cfg, layout))
    monkeypatch.setattr(sys, "argv", ["discover_prefix.py", "--config", "unused", "--stage", stage])
    with pytest.raises(ValueError, match="disabled"):
        module.main()
    assert not (layout.ann_training_prefix_dir / "prefix_state.json").exists()
    assert not (layout.post_finetuning_prefix_dir / "prefix_state.json").exists()


def test_original_base_calibration_collects_all_sites_without_prefix(bundle, monkeypatch, tmp_path):
    from snn2.calibration import collect_site_statistics
    from snn2.controller import SiteController

    cfg, layout, _ = bundle
    model = _tiny_model()
    root = tmp_path / 'collected_stage_a'
    controller = SiteController(mode='collect', site_root=root)
    install_model_integration(model, controller, rotation_state(cfg, layout))
    rows = [{'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1], 'labels': [1, 2, 3]}]
    monkeypatch.setattr('snn2.calibration.tokenize_dataset', lambda *args, **kwargs: rows)
    result = collect_site_statistics(
        model, controller, SimpleNamespace(pad_token_id=0, eos_token_id=1), object(), cfg,
        prefix_key_values_for_stage(cfg, layout, stage='ann_training'), root,
        purpose='ann_training_calibration', extra_metadata=calibration_provenance(cfg, layout, stage='ann_training'),
    )
    assert controller.mode == 'collect'
    assert len(result['states']['sites']) == 10
    assert result['states']['source_model_stage'] == 'original_pretrained_base'
    assert result['states']['eligible_for_ann_training'] is True
    assert result['states']['actual_prefix_length'] == 0
    assert not hasattr(model, '_snn2_prefix_key_values')


@pytest.mark.parametrize('neuron', ['phase', 'gif', 'mtn'])
def test_no_rotation_snn_runs_real_temporal_forward_without_prefix(bundle, neuron):
    from snn2.model_integration import temporal_forward

    cfg, layout, _ = bundle
    model = _tiny_model().eval()
    create_conversion(cfg, layout, neuron)
    controller, _ = build_evaluation_controller(cfg, layout, neuron=neuron)
    install_model_integration(model, controller, rotation_state(cfg, layout))
    ids = torch.tensor([[1, 2, 3]])
    logits = temporal_forward(model, controller, ids, torch.ones_like(ids))
    assert logits.shape == (1, 3, 16)
    assert torch.isfinite(logits).all()
    assert controller.mode == f'deploy_{neuron}'
    assert controller.common_clip_enabled is False
    assert not hasattr(model, '_snn2_prefix_key_values')


@pytest.mark.parametrize('key,value', [('prefix_root', '/old/prefix'), ('prefix_token_ids', [9]),
                                      ('prefix_kv_sha256', 'invalid'), ('rotation_enabled', True)])
def test_conversion_validator_rejects_effective_disabled_metadata(bundle, key, value):
    cfg, layout, _ = bundle
    metadata = create_conversion(cfg, layout, 'gif')
    metadata[key] = value
    write_json(layout.snn_conversion_dir('gif') / 'conversion_metadata.json', metadata)
    with pytest.raises(ValueError):
        validate_conversion_metadata(cfg, layout, 'gif')
