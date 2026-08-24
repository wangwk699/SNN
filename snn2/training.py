from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from .artifacts import ArtifactLayout, read_json, sha256_file, write_json
from .config import is_aware_ann_mode, training_prefix_enabled
from .controller import SiteController
from .data import CausalLMCollator, load_selected_raw, tokenize_dataset
from .model_integration import install_model_integration
from .modeling import load_model, load_tokenizer, model_source_for_stage, prefix_ids_for_stage, prefix_key_values_for_stage, rotation_state
from .prefix_cache import install_prefix_kv_forward
from .state_validation import validate_site_state_bundle


def format_runtime_hms(runtime_seconds: float) -> str:
    """Format a runtime in seconds as HH:MM:SS.ffff without a 24-hour wrap."""
    units_per_second = 10_000
    total_units = round(float(runtime_seconds) * units_per_second)
    total_seconds, fractional_units = divmod(total_units, units_per_second)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{fractional_units:04d}"


def capture_training_artifact_provenance(
    cfg: dict[str, Any],
    layout: ArtifactLayout,
    *,
    prefix_ids: list[int],
) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    if training_prefix_enabled(cfg):
        prefix_state_path = layout.ann_training_prefix_dir / "prefix_state.json"
        if not prefix_state_path.exists():
            raise FileNotFoundError(prefix_state_path)
        saved_ids = [
            int(value)
            for value in read_json(prefix_state_path).get("prefix_token_ids", [])
        ]
        if saved_ids != [int(value) for value in prefix_ids]:
            raise ValueError("Loaded ANN-training Prefix IDs do not match prefix_state.json")
        prefix_kv_path = layout.ann_training_prefix_dir / "prefixed_key_values.pt"
        if saved_ids and not prefix_kv_path.exists():
            raise FileNotFoundError(prefix_kv_path)
        captured.update(
            {
                "ann_training_prefix_root": str(
                    layout.ann_training_prefix_dir.resolve()
                ),
                "ann_training_prefix_state_sha256": sha256_file(prefix_state_path),
                "ann_training_prefix_kv_sha256": (
                    sha256_file(prefix_kv_path) if saved_ids else None
                ),
                "ann_training_prefix_token_ids": saved_ids,
            }
        )
    if is_aware_ann_mode(cfg):
        validate_site_state_bundle(layout.ann_training_site_dir, require_clip=True)
        calibration_manifest = (
            layout.ann_training_site_dir / "calibration_state_manifest.json"
        )
        captured.update(
            {
                "ann_training_calibration_root": str(
                    layout.ann_training_site_dir.resolve()
                ),
                "ann_training_calibration_manifest_sha256": sha256_file(
                    calibration_manifest
                ),
            }
        )
    return captured


def verify_training_artifact_provenance_unchanged(
    captured: dict[str, Any],
    cfg: dict[str, Any],
    layout: ArtifactLayout,
) -> None:
    try:
        current = capture_training_artifact_provenance(
            cfg,
            layout,
            prefix_ids=[
                int(value)
                for value in captured.get("ann_training_prefix_token_ids", [])
            ],
        )
    except Exception as exc:
        raise RuntimeError(
            "ANN-training Prefix/calibration artifacts changed during training"
        ) from exc
    if current != captured:
        raise RuntimeError(
            "ANN-training Prefix/calibration artifacts changed during training"
        )


def train_full_parameters(cfg: dict[str, Any], layout: ArtifactLayout) -> dict[str, Any]:
    from transformers import Trainer, TrainingArguments

    source = model_source_for_stage(cfg, layout, stage="ann_training")
    tokenizer = load_tokenizer(cfg, source if Path(source).exists() else None)
    model = load_model(cfg, source, training=True)
    mode = cfg["replacement"]["train_mode"]
    controller = SiteController(mode=mode, site_root=layout.ann_training_site_dir)
    if is_aware_ann_mode(cfg):
        validate_site_state_bundle(layout.ann_training_site_dir, require_clip=True)
    if cfg["rotation"]["enabled"] or mode != "none":
        install_model_integration(model, controller, rotation_state(cfg, layout))
    model.config.snn2_ann_mode = cfg["experiment"]["ann_mode"]
    model.config.snn2_fused_weights_are_trainable = bool(cfg["rotation"]["enabled"])
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    training_cfg = cfg["training"]
    output_dir = layout.ann_dir / "trainer"
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(training_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training_cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        learning_rate=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
        adam_beta1=float(training_cfg.get("adam_beta1", 0.9)),
        adam_beta2=float(training_cfg.get("adam_beta2", 0.999)),
        adam_epsilon=float(training_cfg.get("adam_epsilon", 1e-8)),
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        warmup_ratio=float(training_cfg["warmup_ratio"]),
        bf16=bool(training_cfg["bf16"]),
        fp16=bool(training_cfg.get("fp16", False)),
        max_grad_norm=float(training_cfg.get("max_grad_norm", 1.0)),
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", False)),
        eval_strategy=training_cfg.get("eval_strategy", "no"),
        save_strategy=training_cfg.get("save_strategy", "no"),
        load_best_model_at_end=bool(training_cfg.get("load_best_model_at_end", False)),
        logging_strategy="steps",
        logging_steps=int(training_cfg.get("logging_steps", 10)),
        seed=int(cfg["experiment"]["seed"]),
        data_seed=int(cfg["experiment"]["seed"]),
        deepspeed=str(Path(training_cfg["deepspeed_config"]).resolve()),
        remove_unused_columns=False,
        report_to=[],
        ddp_find_unused_parameters=False,
    )
    bundle = load_selected_raw(
        cfg, layout, use_configured_train_subset=True
    )
    configured_train_samples = training_cfg.get("tldr_train_samples")
    if (
        cfg["experiment"]["task"] == "tldr"
        and configured_train_samples is not None
        and len(bundle.train) != int(configured_train_samples)
    ):
        raise RuntimeError(
            "TL;DR ANN training selection count mismatch: "
            f"configured={configured_train_samples}, selected={len(bundle.train)}"
        )
    prefixes = prefix_ids_for_stage(cfg, layout, stage="ann_training")
    captured_provenance = capture_training_artifact_provenance(
        cfg, layout, prefix_ids=prefixes
    )
    install_prefix_kv_forward(
        model,
        prefix_key_values_for_stage(cfg, layout, stage="ann_training"),
        controller=controller,
    )
    with arguments.main_process_first(desc="tokenize train and validation datasets"):
        train_dataset = tokenize_dataset(
            bundle.train,
            tokenizer,
            cfg,
            prefix_ids=None,
            desc=f"Tokenizing SNN2 training dataset ({len(bundle.train)} samples)",
        )
        validation_dataset = tokenize_dataset(
            bundle.validation,
            tokenizer,
            cfg,
            prefix_ids=None,
            desc=f"Tokenizing SNN2 validation dataset ({len(bundle.validation)} samples)",
        )
    if int(os.environ.get("RANK", "0")) == 0:
        template = getattr(tokenizer, "chat_template", None) or ""
        write_json(
            layout.ann_dir / "tokenization_metadata.json",
            {
                "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", source)),
                "model_revision": cfg["experiment"].get("model_revision"),
                "max_seq_length": int(cfg["data"]["max_seq_length"]),
                "truncation": True,
                "packing": False,
                "loss_tokens": "assistant/completion plus EOS only",
                "prefix_mode": "fixed_past_key_values",
                "prefix_enabled": training_prefix_enabled(cfg),
                "prefix_token_ids": prefixes,
                "prefix_loss_masked": "not_applicable_prefix_not_in_labels",
                "chat_template": template,
                "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
                "configured_tldr_train_samples": training_cfg.get("tldr_train_samples"),
                "tldr_train_seed": int(training_cfg.get("tldr_train_seed", 42)),
                "actual_train_samples": len(train_dataset),
                "train_sampling": bundle.manifests["train"].get("sampling"),
            },
        )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=CausalLMCollator(tokenizer),
        processing_class=tokenizer,
    )
    result = trainer.train(resume_from_checkpoint=training_cfg.get("resume_from_checkpoint"))
    verify_training_artifact_provenance_unchanged(captured_provenance, cfg, layout)
    final_dir = layout.ann_checkpoint_dir
    trainer.save_model(str(final_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_dir)
    metrics = dict(result.metrics)
    metrics.update(
        {
            "final_model_checkpoint": str(layout.ann_checkpoint_dir.resolve()),
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "fused_rotation_weights_trained": bool(cfg["rotation"]["enabled"]),
            "prefix_enabled": training_prefix_enabled(cfg),
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "effective_global_batch_size": (
                int(training_cfg["per_device_train_batch_size"])
                * int(training_cfg["gradient_accumulation_steps"])
                * int(os.environ.get("WORLD_SIZE", "1"))
            ),
            **captured_provenance,
        }
    )
    if "train_runtime" in metrics:
        metrics["train_runtime_hms"] = format_runtime_hms(metrics["train_runtime"])
    if trainer.is_world_process_zero():
        write_json(layout.ann_dir / "training_result.json", metrics)
        write_json(layout.ann_dir / "trainer_log_history.json", trainer.state.log_history)
    return metrics
