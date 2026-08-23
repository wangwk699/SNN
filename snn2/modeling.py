from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .artifacts import ArtifactLayout, read_json
from .model_integration import register_attention_backend
from .rotation import load_rotation_state
from .prefix_cache import load_prefix_key_values
from .config import (
    evaluation_prefix_enabled,
    final_evaluation_prefix_artifact_stage,
    post_finetuning_prefix_enabled,
    rotated_pre_finetuning_prefix_enabled,
    training_prefix_enabled,
)


def model_source(cfg: dict[str, Any], layout: ArtifactLayout, ann: bool = False) -> str:
    return model_source_for_stage(cfg, layout, stage="post_finetuning" if ann else "ann_training")


def model_source_for_stage(cfg: dict[str, Any], layout: ArtifactLayout, *, stage: str) -> str:
    if stage in {"ann_training", "pre_finetuning"}:
        return str(layout.rotation_dir / "fused_base") if cfg["rotation"]["enabled"] else cfg["experiment"]["model_name"]
    if stage == "rotated_pre_finetuning":
        return str(layout.rotation_dir / "fused_base")
    if stage in {"vanilla_analysis", "base_evaluation"}:
        return cfg["experiment"]["model_name"]
    if stage == "post_finetuning":
        return str(layout.ann_checkpoint_dir)
    raise ValueError(f"Unknown model stage: {stage}")

def load_tokenizer(cfg: dict[str, Any], source: str | None = None):
    from transformers import AutoTokenizer

    source = source or cfg["experiment"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        revision=cfg["experiment"].get("model_revision"),
        trust_remote_code=bool(cfg["experiment"].get("trust_remote_code", False)),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    configured_template = cfg["data"].get("chat_template")
    if configured_template and (
        bool(cfg["data"].get("chat_template_override", False))
        or not getattr(tokenizer, "chat_template", None)
    ):
        tokenizer.chat_template = configured_template
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = cfg["data"].get("truncation_side", "right")
    return tokenizer


def load_model(
    cfg: dict[str, Any],
    source: str,
    training: bool,
    device_map: str | None = None,
):
    from transformers import AutoModelForCausalLM

    register_attention_backend()
    dtype_name = cfg["training"].get("dtype", "bfloat16")
    dtype = getattr(torch, dtype_name)
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": bool(cfg["experiment"].get("trust_remote_code", False)),
        "attn_implementation": "eager",
    }
    if source == cfg["experiment"]["model_name"] and cfg["experiment"].get("model_revision"):
        kwargs["revision"] = cfg["experiment"]["model_revision"]
    if not training and device_map:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    model.config.use_cache = False
    return model


def prefix_ids(cfg: dict[str, Any], layout: ArtifactLayout) -> list[int]:
    return prefix_ids_for_stage(cfg, layout, stage="ann_training")


def prefix_ids_for_stage(cfg: dict[str, Any], layout: ArtifactLayout, *, stage: str) -> list[int]:
    if stage == "ann_training":
        if not training_prefix_enabled(cfg):
            return []
        path = layout.ann_training_prefix_dir / "prefix_state.json"
    elif stage == "post_finetuning":
        if not post_finetuning_prefix_enabled(cfg):
            return []
        path = layout.post_finetuning_prefix_dir / "prefix_state.json"
    elif stage == "final_evaluation":
        if not evaluation_prefix_enabled(cfg):
            return []
        artifact_stage = final_evaluation_prefix_artifact_stage(cfg)
        path = (
            layout.ann_training_prefix_dir
            if artifact_stage == "ann_training"
            else layout.post_finetuning_prefix_dir
        ) / "prefix_state.json"
    elif stage == "rotated_pre_finetuning":
        if (
            not bool(cfg["rotation"]["enabled"])
            or not rotated_pre_finetuning_prefix_enabled(cfg)
        ):
            return []
        path = layout.rotated_pre_finetuning_prefix_dir / "prefix_state.json"
    elif stage in {"vanilla_analysis", "base_evaluation"}:
        return []
    else:
        raise ValueError(f"Unknown prefix stage: {stage}")
    state = read_json(path)
    return [int(value) for value in state["prefix_token_ids"]]


def prefix_key_values(cfg: dict[str, Any], layout: ArtifactLayout):
    return prefix_key_values_for_stage(cfg, layout, stage="ann_training")


def prefix_key_values_for_stage(cfg: dict[str, Any], layout: ArtifactLayout, *, stage: str):
    ids = prefix_ids_for_stage(cfg, layout, stage=stage)
    if not ids:
        return None
    if stage == "ann_training":
        directory = layout.ann_training_prefix_dir
    elif stage == "rotated_pre_finetuning":
        directory = layout.rotated_pre_finetuning_prefix_dir
    elif stage == "post_finetuning":
        directory = layout.post_finetuning_prefix_dir
    elif stage == "final_evaluation":
        directory = (
            layout.ann_training_prefix_dir
            if final_evaluation_prefix_artifact_stage(cfg) == "ann_training"
            else layout.post_finetuning_prefix_dir
        )
    else:
        raise ValueError(f"Unknown prefix stage: {stage}")
    path = directory / "prefixed_key_values.pt"
    if not path.exists():
        raise FileNotFoundError(f"Prefix is enabled but fixed KV cache is missing: {path}. Re-run scripts/discover_prefix.py.")
    return load_prefix_key_values(path)

def rotation_state(cfg: dict[str, Any], layout: ArtifactLayout):
    if not bool(cfg["rotation"]["enabled"]):
        return None
    return load_rotation_state(layout.rotation_dir / "rotation_state.pt")
