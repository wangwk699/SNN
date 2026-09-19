from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from .artifacts import (
    ArtifactLayout,
    data_selection_seed_dirname,
    read_json,
    sha256_file,
    write_json,
)
from .data_constants import CANONICAL_PREPROCESSING_NUM_SAMPLES


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


def _tldr_train_selection(
    raw_train: Any, cfg: dict[str, Any]
) -> tuple[list[int], str]:
    configured = cfg["training"].get("tldr_train_samples")
    if configured is None:
        return list(range(len(raw_train))), "full_split"
    requested = int(configured)
    if requested <= 0:
        raise ValueError("training.tldr_train_samples must be a positive integer or null")
    if requested > len(raw_train):
        raise ValueError(
            f"Requested {requested} TL;DR training samples, but the train split "
            f"contains only {len(raw_train)} rows"
        )
    if requested == len(raw_train):
        return list(range(len(raw_train))), "full_split"
    rng = random.Random(int(cfg["training"].get("tldr_train_seed", 42)))
    indices = rng.sample(range(len(raw_train)), k=requested)
    indices.sort()
    return indices, "seeded_random_without_replacement"


def _tulu3_train_selection(training_pool_indices: list[int], cfg: dict[str, Any]) -> tuple[list[int], str]:
    configured = cfg["training"].get("train_samples")
    if configured is None or int(configured) == len(training_pool_indices):
        return list(training_pool_indices), "full_training_pool"
    requested = int(configured)
    if requested <= 0 or requested > len(training_pool_indices):
        raise ValueError("training.train_samples must be positive and no larger than the Tulu-3 training pool")
    indices = random.Random(int(cfg["training"].get("train_seed", 42))).sample(training_pool_indices, k=requested)
    indices.sort()
    return indices, "seeded_random_without_replacement"


def _calibration_selection(
    train_indices: list[int],
    *,
    seed: int,
    num_samples: int,
    with_replacement: bool,
) -> tuple[list[int], list[int]]:
    if num_samples <= 0:
        raise ValueError("Calibration sample count must be positive")
    rng = random.Random(seed)
    if with_replacement:
        positions = [rng.randrange(len(train_indices)) for _ in range(num_samples)]
    else:
        if num_samples > len(train_indices):
            raise ValueError(
                f"Cannot sample {num_samples} calibration examples without replacement "
                f"from only {len(train_indices)} training examples"
            )
        positions = rng.sample(range(len(train_indices)), k=num_samples)
    return positions, [train_indices[position] for position in positions]


def _validate_stage_a_calibration_selection(
    train_indices: list[int],
    calibration_positions: list[int],
    calibration_indices: list[int],
) -> None:
    """Guard the Stage-A invariant and the meaning of recorded positions."""
    expected_indices = [train_indices[position] for position in calibration_positions]
    if expected_indices != calibration_indices:
        raise RuntimeError(
            "Stage-A calibration positions do not resolve to the selected ANN "
            "training indices"
        )
    if not set(calibration_indices).issubset(train_indices):
        raise RuntimeError(
            "Stage-A calibration indices must be a subset of the selected ANN "
            "training indices"
        )


def _stage_a_calibration_provenance(
    cfg: dict[str, Any],
    train_indices: list[int],
    train_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the ANN selection from which Stage-A calibration was drawn."""
    task = cfg["experiment"]["task"]
    seed_field = "tldr_train_seed" if task == "tldr" else "train_seed"
    parent_seed = int(
        train_manifest[seed_field]
        if train_manifest is not None
        else cfg["training"].get(seed_field, 42)
    )
    provenance = {
        "parent_training_samples": len(train_indices),
        "parent_training_seed": parent_seed,
    }
    if task == "tulu3":
        provenance.update({
            "selection_pool": "selected_ann_training_subset",
            "retained_in_shared_training_pool": True,
            "retained_in_ann_training_subset": True,
        })
    else:
        provenance["retained_in_training"] = True
    return provenance


def validate_train_manifest_for_config(
    cfg: dict[str, Any], manifest: dict[str, Any]
) -> None:
    """Fail closed unless a fixed Step-2 train manifest matches the config."""
    task = cfg["experiment"]["task"]
    indices = manifest.get("indices")
    mismatches: dict[str, tuple[Any, Any]] = {}

    if not isinstance(indices, list):
        mismatches["indices"] = ("list[int]", type(indices).__name__)
    elif any(type(index) is not int or index < 0 for index in indices):
        mismatches["indices"] = ("non-negative integers", "invalid entries")
    elif len(set(indices)) != len(indices):
        mismatches["indices"] = ("unique training indices", "duplicates present")

    expected_common = {
        "seed": int(cfg["experiment"]["seed"]),
        "split": cfg["data"].get("train_split", "train"),
        "data_selection_identity": data_selection_seed_dirname(cfg),
    }
    for field, expected in expected_common.items():
        if manifest.get(field) != expected:
            mismatches[field] = (expected, manifest.get(field))

    if task == "tldr":
        configured_samples = cfg["training"].get("tldr_train_samples")
        expected = {
            "tldr_train_samples": configured_samples,
            "tldr_train_seed": int(cfg["training"].get("tldr_train_seed", 42)),
        }
        full_sampling = "full_split"
    elif task == "tulu3":
        configured_samples = cfg["training"].get("train_samples")
        expected = {
            "train_samples": configured_samples,
            "train_seed": int(cfg["training"].get("train_seed", 42)),
            "validation_size": int(cfg["data"]["validation_size"]),
        }
        full_sampling = "full_training_pool"
    else:
        configured_samples = None
        expected = {}
        full_sampling = None

    missing = object()
    for field, expected_value in expected.items():
        actual = manifest.get(field, missing)
        if actual != expected_value or type(actual) is not type(expected_value):
            mismatches[field] = (
                expected_value,
                "<missing>" if actual is missing else actual,
            )

    if configured_samples is not None and isinstance(indices, list):
        if len(indices) != int(configured_samples):
            mismatches["indices_count"] = (int(configured_samples), len(indices))
    elif configured_samples is None and full_sampling is not None:
        if manifest.get("sampling") != full_sampling:
            mismatches["sampling"] = (full_sampling, manifest.get("sampling"))

    if mismatches:
        details = ", ".join(
            f"{field}: configured={expected!r}, manifest={actual!r}"
            for field, (expected, actual) in mismatches.items()
        )
        raise ValueError(
            "Training manifest does not match the current configuration. "
            "The ANN training subset must be prepared by Step 2. "
            f"Mismatches: {details}. Re-run: python scripts/prepare_data.py "
            "--config <config>"
        )


def _tulu3_shared_split_selection(raw_train: Any, cfg: dict[str, Any]) -> tuple[list[int], list[int]]:
    """Return fixed validation and the validation-excluded shared training pool."""
    validation_size = int(cfg["data"].get("validation_size", 1_000))
    if validation_size <= 0 or len(raw_train) <= validation_size:
        raise ValueError("Tulu 3 validation_size must be positive and smaller than the source split")
    permutation = list(range(len(raw_train)))
    random.Random(int(cfg["experiment"]["seed"])).shuffle(permutation)
    return permutation[:validation_size], permutation[validation_size:]


def _manifest_split_selection(
    cfg: dict[str, Any], raw: Any
) -> tuple[Any, str, list[int], str, str, list[int]]:
    """Return the deterministic split selections used by data manifests."""
    data_cfg = cfg["data"]
    seed = int(cfg["experiment"]["seed"])
    rng = random.Random(seed)
    train_split = data_cfg.get("train_split", "train")
    raw_train = raw[train_split]
    task = cfg["experiment"]["task"]

    if task == "tulu3":
        validation_indices, shared_training_pool_indices = _tulu3_shared_split_selection(raw_train, cfg)
        train_indices, train_sampling = _tulu3_train_selection(shared_training_pool_indices, cfg)
        return (
            raw_train,
            train_split,
            train_indices,
            train_sampling,
            train_split,
            validation_indices,
        )

    validation_split = data_cfg.get("validation_split", "validation")
    if task == "tldr":
        train_indices, train_sampling = _tldr_train_selection(raw_train, cfg)
    else:
        train_indices = list(range(len(raw_train)))
        train_sampling = "full_split"
    return (
        raw_train,
        train_split,
        train_indices,
        train_sampling,
        validation_split,
        list(range(len(raw[validation_split]))),
    )


def prepare_calibration_manifest(
    cfg: dict[str, Any], layout: ArtifactLayout
) -> dict[str, Any]:
    """Derive Stage-A calibration only from an existing Step-2 train manifest."""
    train_manifest_path = layout.data_dir / "train_manifest.json"
    if not train_manifest_path.exists():
        raise FileNotFoundError(
            "Step 2 training manifest is missing. Run: python "
            "scripts/prepare_data.py --config <config> before using "
            "--calibration-only."
        )
    train_manifest = read_json(train_manifest_path)
    validate_train_manifest_for_config(cfg, train_manifest)
    raw = _load_raw(cfg)
    train_split = train_manifest["split"]
    raw_train = raw[train_split]
    train_indices = [int(index) for index in train_manifest["indices"]]
    data_cfg = cfg["data"]
    calibration_num_samples = int(cfg["calibration"]["num_samples"])
    with_replacement = bool(cfg["calibration"].get("with_replacement", False))
    calibration_positions, calibration_indices = _calibration_selection(
        train_indices,
        seed=int(cfg["calibration"]["seed"]),
        num_samples=calibration_num_samples,
        with_replacement=with_replacement,
    )
    _validate_stage_a_calibration_selection(
        train_indices, calibration_positions, calibration_indices
    )
    manifest = {
        "dataset_name": data_cfg["dataset_name"],
        "dataset_config_name": data_cfg.get("dataset_config_name"),
        "dataset_revision": data_cfg.get("dataset_revision"),
        "seed": int(cfg["experiment"]["seed"]),
        "manifest_role": "stage_a_calibration_selection",
        "split": train_split,
        "sampling": "seeded_with_replacement" if with_replacement else "seeded_without_replacement",
        "calibration_seed": int(cfg["calibration"]["seed"]),
        "num_samples": calibration_num_samples,
        "positions_in_selected_train": calibration_positions,
        "indices": calibration_indices,
        "record_ids": _record_ids(raw_train, calibration_indices),
        "duplicates_preserved": with_replacement,
        "data_selection_identity": data_selection_seed_dirname(cfg),
        **_stage_a_calibration_provenance(cfg, train_indices, train_manifest),
    }
    manifest_path = layout.calibration_data_manifest_path
    write_json(manifest_path, manifest)
    return manifest


def prepare_manifests(cfg: dict[str, Any], layout: ArtifactLayout) -> dict[str, dict[str, Any]]:
    raw = _load_raw(cfg)
    data_cfg = cfg["data"]
    seed = int(cfg["experiment"]["seed"])
    task = cfg["experiment"]["task"]
    (
        raw_train,
        train_split,
        train_indices,
        train_sampling,
        validation_split,
        validation_indices,
    ) = _manifest_split_selection(cfg, raw)

    calibration_num_samples = int(cfg["calibration"]["num_samples"])
    with_replacement = bool(cfg["calibration"].get("with_replacement", False))
    calibration_positions, calibration_indices = _calibration_selection(
        train_indices,
        seed=int(cfg["calibration"]["seed"]),
        num_samples=calibration_num_samples,
        with_replacement=with_replacement,
    )
    _validate_stage_a_calibration_selection(
        train_indices, calibration_positions, calibration_indices
    )
    _, canonical_indices = _calibration_selection(
        list(range(len(raw_train))),
        seed=int(cfg["experiment"]["seed"]),
        num_samples=CANONICAL_PREPROCESSING_NUM_SAMPLES,
        with_replacement=False,
    )

    common = {
        "dataset_name": data_cfg["dataset_name"],
        "dataset_config_name": data_cfg.get("dataset_config_name"),
        "dataset_revision": data_cfg.get("dataset_revision"),
        "seed": seed,
    }
    data_selection_identity = data_selection_seed_dirname(cfg)
    manifests = {
        "train": {
            **common,
            "data_selection_identity": data_selection_identity,
            "split": train_split,
            "sampling": train_sampling,
            **(
                {
                    "tldr_train_samples": cfg["training"].get("tldr_train_samples"),
                    "tldr_train_seed": int(cfg["training"].get("tldr_train_seed", 42)),
                }
                if task == "tldr"
                else {}
            ),
            **({"train_samples": cfg["training"].get("train_samples"), "train_seed": int(cfg["training"].get("train_seed", 42)), "validation_size": int(data_cfg["validation_size"]), "selection_scope": "current_ann_training_config"} if task == "tulu3" else {}),
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
            "data_selection_identity": data_selection_identity,
            "manifest_role": "stage_a_calibration_selection",
            "split": train_split,
            "sampling": "seeded_with_replacement" if with_replacement else "seeded_without_replacement",
            "calibration_seed": int(cfg["calibration"]["seed"]),
            "num_samples": calibration_num_samples,
            "positions_in_selected_train": calibration_positions,
            "indices": calibration_indices,
            "record_ids": _record_ids(raw_train, calibration_indices),
            "duplicates_preserved": with_replacement,
            **_stage_a_calibration_provenance(cfg, train_indices),
        },
        "canonical_preprocessing_calibration": {
            **common,
            "manifest_role": "canonical_preprocessing_calibration",
            "selection_scope": "raw_train_split",
            "selection_seed_source": "experiment.seed",
            "selection_seed": int(cfg["experiment"]["seed"]),
            "split": train_split,
            "sampling": "seeded_without_replacement",
            "num_samples": CANONICAL_PREPROCESSING_NUM_SAMPLES,
            "indices": canonical_indices,
            "record_ids": _record_ids(raw_train, canonical_indices),
            "duplicates_preserved": False,
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
    for name in ("train", "validation", "evaluation"):
        if name in manifests:
            write_json(layout.data_dir / f"{name}_manifest.json", manifests[name])
    stage_a_manifest_path = getattr(
        layout, "calibration_data_manifest_path", layout.data_dir / "calibration_manifest.json"
    )
    canonical_manifest_path = getattr(
        layout,
        "canonical_preprocessing_calibration_manifest_path",
        layout.data_dir / "canonical_preprocessing" / "num_samples_128" / "calibration_manifest.json",
    )
    write_json(stage_a_manifest_path, manifests["calibration"])
    canonical_manifest = manifests["canonical_preprocessing_calibration"]
    if canonical_manifest_path.exists():
        existing = read_json(canonical_manifest_path)
        validate_canonical_preprocessing_manifest_for_config(
            cfg, existing, expected_indices=canonical_indices
        )
        if existing.get("record_ids") != canonical_manifest["record_ids"]:
            raise ValueError(
                "Canonical preprocessing calibration manifest record IDs do not "
                "match the deterministic canonical selection"
            )
        manifests["canonical_preprocessing_calibration"] = existing
    else:
        write_json(canonical_manifest_path, canonical_manifest)
    return manifests


def load_manifests(cfg: dict[str, Any], layout: ArtifactLayout) -> dict[str, dict[str, Any]]:
    result = {
        name: read_json(layout.data_dir / f"{name}_manifest.json")
        for name in ("train", "validation")
    }
    calibration_path = getattr(
        layout, "calibration_data_manifest_path", layout.data_dir / "calibration_manifest.json"
    )
    result["calibration"] = read_json(calibration_path)
    evaluation = layout.data_dir / "evaluation_manifest.json"
    if evaluation.exists():
        result["evaluation"] = read_json(evaluation)
    return result

def validate_canonical_preprocessing_manifest_for_config(
    cfg: dict[str, Any],
    manifest: dict[str, Any],
    *,
    expected_indices: list[int] | None = None,
) -> None:
    """Validate the task-shared canonical selection against its full identity."""
    data_cfg = cfg["data"]
    expected = {
        "dataset_name": data_cfg["dataset_name"],
        "dataset_config_name": data_cfg.get("dataset_config_name"),
        "dataset_revision": data_cfg.get("dataset_revision"),
        "seed": int(cfg["experiment"]["seed"]),
        "manifest_role": "canonical_preprocessing_calibration",
        "selection_scope": "raw_train_split",
        "selection_seed_source": "experiment.seed",
        "selection_seed": int(cfg["experiment"]["seed"]),
        "split": data_cfg.get("train_split", "train"),
        "num_samples": CANONICAL_PREPROCESSING_NUM_SAMPLES,
        "sampling": "seeded_without_replacement",
        "duplicates_preserved": False,
    }
    mismatched = {
        key: (value, manifest.get(key))
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    indices = manifest.get("indices")
    indices_valid = (
        isinstance(indices, list)
        and len(indices) == CANONICAL_PREPROCESSING_NUM_SAMPLES
        and all(isinstance(index, int) and index >= 0 for index in indices)
        and len(set(indices)) == len(indices)
    )
    if not indices_valid:
        mismatched["indices"] = ("128 unique nonnegative integers", indices)
    elif expected_indices is not None and indices != expected_indices:
        mismatched["indices"] = (expected_indices, indices)
    if mismatched:
        raise ValueError(
            "Canonical preprocessing calibration manifest is invalid: "
            f"{mismatched}"
        )


def load_canonical_preprocessing_raw(cfg: dict[str, Any], layout: ArtifactLayout) -> Any:
    """Load the fixed experiment-seed canonical 128-sample selection."""
    manifest_path = layout.canonical_preprocessing_calibration_manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    validate_canonical_preprocessing_manifest_for_config(cfg, manifest)
    raw = _load_raw(cfg)
    return raw[manifest["split"]].select(manifest["indices"])


def validate_prefix_discovery_state(
    cfg: dict[str, Any],
    layout: ArtifactLayout,
    prefix_dir: str | Path,
    *,
    stage: str,
) -> dict[str, Any]:
    """Validate Prefix provenance for an explicit discovery stage."""
    if stage not in {"pre_finetuning", "post_finetuning"}:
        raise ValueError(f"Unknown Prefix discovery stage: {stage}")
    root = Path(prefix_dir)
    pre = stage == "pre_finetuning"
    num_samples = (
        CANONICAL_PREPROCESSING_NUM_SAMPLES
        if pre
        else int(cfg["calibration"]["num_samples"])
    )
    expected_dirname = f"num_samples_{num_samples}"
    if root.name != expected_dirname:
        raise ValueError(f"Prefix root must be {expected_dirname}, got {root.name}")
    path = root / "prefix_state.json"
    if not path.exists():
        raise FileNotFoundError(path)
    state = read_json(path)
    manifest_path = (
        layout.canonical_preprocessing_calibration_manifest_path
        if pre
        else layout.calibration_data_manifest_path
    )
    expected = {
        "discovery_num_samples": num_samples,
        "discovery_data_source": (
            "canonical_preprocessing_calibration"
            if pre
            else "stage_a_calibration_selection"
        ),
        "discovery_manifest_path": str(manifest_path.resolve()),
        "discovery_manifest_sha256": sha256_file(manifest_path),
    }
    mismatched = {
        key: (value, state.get(key))
        for key, value in expected.items()
        if state.get(key) != value
    }
    if mismatched:
        raise ValueError(f"Prefix discovery provenance mismatch: {mismatched}")
    token_ids = [int(value) for value in state.get("prefix_token_ids", [])]
    kv_path = root / "prefixed_key_values.pt"
    if token_ids and not kv_path.exists():
        raise FileNotFoundError(f"Non-empty Prefix requires fixed KV cache: {kv_path}")
    return {
        "state": state,
        "state_path": path,
        "kv_path": kv_path if token_ids else None,
        "token_ids": token_ids,
    }


def load_selected_raw(
    cfg: dict[str, Any],
    layout: ArtifactLayout,
) -> DatasetBundle:
    manifests = load_manifests(cfg, layout)
    validate_train_manifest_for_config(cfg, manifests["train"])
    raw = _load_raw(cfg)
    selected = {
        name: raw[manifest["split"]].select(manifest["indices"])
        for name, manifest in manifests.items()
    }
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


def tldr_prompt_and_reference(row: dict[str, Any]) -> tuple[str, str]:
    prompt = _as_text(
        row.get("prompt", row.get("pompt", row.get("article", row.get("text", ""))))
    )
    reference = _as_text(
        row.get("completion", row.get("summary", row.get("label", row.get("response", ""))))
    )
    return prompt, reference


def encode_tldr_generation_prompt(
    row: dict[str, Any], tokenizer: Any, cfg: dict[str, Any]
) -> list[int]:
    prompt, _ = tldr_prompt_and_reference(row)
    return list(
        tokenizer.encode(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=int(cfg["evaluation"].get("tldr_input_length", 512)),
        )
    )


def encode_generation_prompt(
    row: dict[str, Any], tokenizer: Any, cfg: dict[str, Any]
) -> list[int]:
    """Encode only the generation prompt while preserving task-specific evaluation semantics."""
    if cfg["experiment"]["task"] == "tldr":
        return encode_tldr_generation_prompt(row, tokenizer, cfg)
    messages = row.get("messages")
    if not isinstance(messages, list):
        instruction = _as_text(row.get("instruction", row.get("prompt", "")))
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": _as_text(
                row.get("response", row.get("output", row.get("completion", "")))
            )},
        ]
    last_assistant = max(
        (index for index, message in enumerate(messages) if message.get("role") == "assistant"),
        default=-1,
    )
    if last_assistant < 0:
        raise ValueError("Tulu generation prompt has no assistant response to exclude")
    return list(
        tokenizer.apply_chat_template(
            messages[:last_assistant], tokenize=True, add_generation_prompt=True
        )
    )


def _encode_tldr(row: dict[str, Any], tokenizer: Any) -> tuple[list[int], list[int]]:
    prompt, completion = tldr_prompt_and_reference(row)
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
        raise ValueError(
            "Prefix token IDs must not be prepended to input_ids. "
            "Use the fixed Prefix past_key_values cache instead."
        )
    max_length = int(cfg["data"]["max_seq_length"])
    truncation_side = cfg["data"].get("truncation_side", "right")
    if len(input_ids) > max_length:
        if truncation_side == "left":
            input_ids, labels = input_ids[-max_length:], labels[-max_length:]
        else:
            input_ids, labels = input_ids[:max_length], labels[:max_length]
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def tokenize_dataset(
    dataset: Any,
    tokenizer: Any,
    cfg: dict[str, Any],
    prefix_ids=None,
    *,
    desc: str = "Tokenizing SNN2 dataset",
):
    columns = list(dataset.column_names)
    return dataset.map(
        lambda row: tokenize_row(row, tokenizer, cfg, prefix_ids),
        remove_columns=columns,
        desc=desc,
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
