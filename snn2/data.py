from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from .artifacts import ArtifactLayout, read_json, write_json


@dataclass
class DatasetBundle:
    train: Any
    validation: Any
    calibration: Any
    evaluation: Any | None
    manifests: dict[str, dict[str, Any]]


def _load_raw(cfg: dict[str, Any]):
    from datasets import load_dataset

    data_cfg = cfg["data"]
    kwargs: dict[str, Any] = {}
    if data_cfg.get("dataset_config_name"):
        kwargs["name"] = data_cfg["dataset_config_name"]
    if data_cfg.get("dataset_revision"):
        kwargs["revision"] = data_cfg["dataset_revision"]
    return load_dataset(data_cfg["dataset_name"], **kwargs)


def _record_id(dataset: Any, index: int) -> Any:
    row = dataset[int(index)]
    for key in ("id", "sample_id", "uuid", "dataset_id"):
        if key in row:
            return row[key]
    return int(index)


def _record_ids(dataset: Any, indices: list[int]) -> list[Any]:
    """Read a manifest ID column in one Arrow operation when one exists."""
    id_column = next(
        (name for name in ("id", "sample_id", "uuid", "dataset_id") if name in dataset.column_names),
        None,
    )
    if id_column is None:
        return [int(index) for index in indices]
    return list(dataset.select(indices)[id_column])


def prepare_manifests(cfg: dict[str, Any], layout: ArtifactLayout) -> dict[str, dict[str, Any]]:
    raw = _load_raw(cfg)
    data_cfg = cfg["data"]
    seed = int(cfg["experiment"]["seed"])
    rng = random.Random(seed)
    train_split = data_cfg.get("train_split", "train")
    raw_train = raw[train_split]
    task = cfg["experiment"]["task"]

    if task == "tulu3":
        train_size = int(data_cfg.get("train_size", 100_000))
        validation_size = int(data_cfg.get("validation_size", 1_000))
        if len(raw_train) < train_size + validation_size:
            raise ValueError(
                f"Tulu 3 requires at least {train_size + validation_size} rows, got {len(raw_train)}"
            )
        permutation = list(range(len(raw_train)))
        rng.shuffle(permutation)
        train_indices = permutation[:train_size]
        validation_indices = permutation[train_size : train_size + validation_size]
        validation_split = train_split
    else:
        train_indices = list(range(len(raw_train)))
        validation_split = data_cfg.get("validation_split", "validation")
        raw_validation = raw[validation_split]
        validation_indices = list(range(len(raw_validation)))

    # calibration_rng = random.Random(int(cfg["calibration"]["seed"]))
    # draws = int(cfg["calibration"]["num_samples"])
    # calibration_positions = [calibration_rng.randrange(len(train_indices)) for _ in range(draws)]
    # calibration_indices = [train_indices[position] for position in calibration_positions]

    calibration_rng = random.Random(int(cfg["calibration"]["seed"]))
    draws = int(cfg["calibration"]["num_samples"])
    with_replacement = bool(cfg["calibration"].get("with_replacement", False))

    if with_replacement:
        calibration_positions = [
            calibration_rng.randrange(len(train_indices))
            for _ in range(draws)
        ]
    else:
        if draws > len(train_indices):
            raise ValueError(
                f"Cannot sample {draws} calibration examples without replacement "
                f"from only {len(train_indices)} training examples"
            )
        calibration_positions = calibration_rng.sample(
            range(len(train_indices)),
            k=draws,
        )

    calibration_indices = [
        train_indices[position]
        for position in calibration_positions
    ]

    common = {
        "dataset_name": data_cfg["dataset_name"],
        "dataset_config_name": data_cfg.get("dataset_config_name"),
        "dataset_revision": data_cfg.get("dataset_revision"),
        "seed": seed,
    }
    manifests = {
        "train": {
            **common,
            "split": train_split,
            "sampling": "full_split" if task != "tulu3" else "seeded_without_replacement",
            "indices": train_indices,
            "record_ids": _record_ids(raw_train, train_indices),
        },
        "validation": {
            **common,
            "split": validation_split,
            "sampling": "full_split" if task != "tulu3" else "seeded_without_replacement",
            "indices": validation_indices,
            "record_ids": _record_ids(raw[validation_split], validation_indices),
        },
        "calibration": {
            **common,
            "split": train_split,
            "sampling": (
                "seeded_with_replacement"
                if with_replacement
                else "seeded_without_replacement"
            ),
            "calibration_seed": int(cfg["calibration"]["seed"]),
            "positions_in_selected_train": calibration_positions,
            "indices": calibration_indices,
            "record_ids": _record_ids(raw_train, calibration_indices),
            "duplicates_preserved": with_replacement,
            "retained_in_training": True,
        },
    }
    if task == "tldr":
        evaluation_split = data_cfg.get("evaluation_split", "test")
        raw_evaluation = raw[evaluation_split]
        evaluation_indices = list(range(len(raw_evaluation)))
        manifests["evaluation"] = {
            **common,
            "split": evaluation_split,
            "sampling": "full_split",
            "indices": evaluation_indices,
            "record_ids": _record_ids(raw_evaluation, evaluation_indices),
        }
    layout.data_dir.mkdir(parents=True, exist_ok=True)
    for name, manifest in manifests.items():
        write_json(layout.data_dir / f"{name}_manifest.json", manifest)
    return manifests


def load_manifests(layout: ArtifactLayout) -> dict[str, dict[str, Any]]:
    result = {
        name: read_json(layout.data_dir / f"{name}_manifest.json")
        for name in ("train", "validation", "calibration")
    }
    evaluation = layout.data_dir / "evaluation_manifest.json"
    if evaluation.exists():
        result["evaluation"] = read_json(evaluation)
    return result


def load_selected_raw(cfg: dict[str, Any], layout: ArtifactLayout) -> DatasetBundle:
    manifests = load_manifests(layout)
    raw = _load_raw(cfg)
    selected = {}
    for name, manifest in manifests.items():
        selected[name] = raw[manifest["split"]].select(manifest["indices"])
    return DatasetBundle(
        train=selected["train"],
        validation=selected["validation"],
        calibration=selected["calibration"],
        evaluation=selected.get("evaluation"),
        manifests=manifests,
    )


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_as_text(item) for item in value)
    if isinstance(value, dict):
        return _as_text(value.get("content", value.get("text", "")))
    return str(value)


def _encode_tldr(row: dict[str, Any], tokenizer: Any) -> tuple[list[int], list[int]]:
    prompt = _as_text(
        row.get("prompt", row.get("pompt", row.get("article", row.get("text", ""))))
    )
    completion = _as_text(
        row.get("completion", row.get("summary", row.get("label", row.get("response", ""))))
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        completion_ids.append(int(tokenizer.eos_token_id))
    return prompt_ids + completion_ids, [-100] * len(prompt_ids) + completion_ids.copy()


def _encode_messages(row: dict[str, Any], tokenizer: Any) -> tuple[list[int], list[int]]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        instruction = _as_text(row.get("instruction", row.get("prompt", "")))
        response = _as_text(row.get("response", row.get("output", row.get("completion", ""))))
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        input_ids = list(encoded["input_ids"])
        mask = encoded.get("assistant_masks") or encoded.get("assistant_tokens_mask")
        if mask is not None:
            labels = [token if int(flag) else -100 for token, flag in zip(input_ids, mask)]
            return input_ids, labels
    except (TypeError, ValueError, KeyError):
        pass

    last_assistant = max(
        (index for index, message in enumerate(messages) if message.get("role") == "assistant"),
        default=-1,
    )
    if last_assistant < 0:
        raise ValueError("Tulu example has no assistant message")
    prompt_messages = messages[:last_assistant]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True
    )
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    labels = [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids) :])
    return list(full_ids), labels


def tokenize_row(
    row: dict[str, Any],
    tokenizer: Any,
    cfg: dict[str, Any],
    prefix_ids: list[int] | None = None,
) -> dict[str, Any]:
    if cfg["experiment"]["task"] == "tldr":
        input_ids, labels = _encode_tldr(row, tokenizer)
    else:
        input_ids, labels = _encode_messages(row, tokenizer)
    prefix_ids = list(prefix_ids or [])
    if prefix_ids:
        input_ids = prefix_ids + input_ids
        labels = [-100] * len(prefix_ids) + labels
    max_length = int(cfg["data"]["max_seq_length"])
    truncation_side = cfg["data"].get("truncation_side", "right")
    if len(input_ids) > max_length:
        if truncation_side == "left":
            input_ids, labels = input_ids[-max_length:], labels[-max_length:]
        else:
            input_ids, labels = input_ids[:max_length], labels[:max_length]
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def tokenize_dataset(dataset: Any, tokenizer: Any, cfg: dict[str, Any], prefix_ids=None):
    columns = list(dataset.column_names)
    return dataset.map(
        lambda row: tokenize_row(row, tokenizer, cfg, prefix_ids),
        remove_columns=columns,
        desc="Tokenizing SNN2 dataset",
    )


class CausalLMCollator:
    def __init__(self, tokenizer: Any):
        self.pad_id = int(tokenizer.pad_token_id or tokenizer.eos_token_id)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in features]
        masks = [torch.tensor(item["attention_mask"], dtype=torch.long) for item in features]
        labels = [torch.tensor(item["labels"], dtype=torch.long) for item in features]
        return {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id),
            "attention_mask": pad_sequence(masks, batch_first=True, padding_value=0),
            "labels": pad_sequence(labels, batch_first=True, padding_value=-100),
        }
