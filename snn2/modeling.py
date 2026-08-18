from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .artifacts import ArtifactLayout, read_json
from .model_integration import register_attention_backend
from .rotation import load_rotation_state


def model_source(cfg: dict[str, Any], layout: ArtifactLayout, ann: bool = False) -> str:
    if ann:
        return str(layout.ann_dir / "final")
    if bool(cfg["rotation"]["enabled"]):
        return str(layout.rotation_dir / "fused_base")
    return cfg["experiment"]["model_name"]


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
    if not bool(cfg["prefix"]["enabled"]):
        return []
    state = read_json(layout.prefix_dir / "prefix_state.json")
    return [int(value) for value in state["prefix_token_ids"]]


def rotation_state(cfg: dict[str, Any], layout: ArtifactLayout):
    if not bool(cfg["rotation"]["enabled"]):
        return None
    return load_rotation_state(layout.rotation_dir / "rotation_state.pt")
