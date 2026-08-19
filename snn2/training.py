from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from .artifacts import ArtifactLayout, write_json
from .controller import SiteController
from .data import CausalLMCollator, load_selected_raw, tokenize_dataset
from .model_integration import install_model_integration
from .modeling import load_model, load_tokenizer, model_source, prefix_ids, prefix_key_values, rotation_state
from .prefix_cache import install_prefix_kv_forward


def train_full_parameters(cfg: dict[str, Any], layout: ArtifactLayout) -> dict[str, Any]:
    from transformers import Trainer, TrainingArguments

    source = model_source(cfg, layout, ann=False)
    tokenizer = load_tokenizer(cfg, source if Path(source).exists() else None)
    model = load_model(cfg, source, training=True)
    mode = cfg["replacement"]["train_mode"]
    controller = SiteController(mode=mode, site_root=layout.site_dir)
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
    bundle = load_selected_raw(cfg, layout)
    prefixes = prefix_ids(cfg, layout)
    install_prefix_kv_forward(model, prefix_key_values(cfg, layout))
    with arguments.main_process_first(desc="tokenize train and validation datasets"):
        train_dataset = tokenize_dataset(bundle.train, tokenizer, cfg, prefix_ids=None)
        validation_dataset = tokenize_dataset(bundle.validation, tokenizer, cfg, prefix_ids=None)
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
                "prefix_token_ids": prefixes,
                "prefix_loss_masked": "not_applicable_prefix_not_in_labels",
                "chat_template": template,
                "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
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
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "effective_global_batch_size": (
                int(training_cfg["per_device_train_batch_size"])
                * int(training_cfg["gradient_accumulation_steps"])
                * int(os.environ.get("WORLD_SIZE", "1"))
            ),
        }
    )
    if trainer.is_world_process_zero():
        write_json(layout.ann_dir / "training_result.json", metrics)
        write_json(layout.ann_dir / "trainer_log_history.json", trainer.state.log_history)
    return metrics
