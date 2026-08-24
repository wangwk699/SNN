from __future__ import annotations
import math
from pathlib import Path
from _common import parser, setup

from snn2.artifacts import prefix_enabled_dirname, read_json, sha256_file, write_json
from snn2.config import (
    conversion_prefix_enabled,
    conversion_reuses_ann_training_artifacts,
    evaluation_prefix_enabled,
    requires_ann_training_calibration,
    requires_pre_finetuning_prefix,
    training_common_clip_enabled,
    training_prefix_enabled,
)
from snn2.conversion import (
    validate_calibration,
    validate_conversion_metadata,
    validate_conversion_prefix,
)
from snn2.sites import topology_metadata
from snn2.evaluation import final_ann_replacement_mode, resolve_tldr_evaluation_layout
from snn2.logging_utils import StageRun
from snn2.state_validation import validate_site_state_bundle
from snn2.temporal_ops import validate_temporal_policy


def _evaluation_metadata(path):
    payload = read_json(path)
    return payload.get("snn2_metadata", payload)


def _verify_final_ann_forward_metadata(cfg, layout, path):
    metadata = _evaluation_metadata(path)
    mode = final_ann_replacement_mode(cfg)
    expected = {
        "identity": ("identity_ann", False, None),
        "phase": ("phase_surrogate_ann", True, "PhaseSurrogate.forward"),
        "gif": ("gif_surrogate_ann", True, "StaticGIF.forward"),
    }[mode]
    required = {
        "evaluation_forward_kind": expected[0],
        "controller_mode": mode,
        "temporal_execution": False,
        "static_replacement_enabled": expected[1],
        "static_replacement_impl": expected[2],
        "evaluation_common_clip_applied": (
            training_common_clip_enabled(cfg) if expected[1] else False
        ),
    }
    for key, value in required.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Final ANN evaluation has stale/incompatible {key}: {path}. "
                "Re-run final ANN evaluation."
            )
    expected_root = layout.ann_training_site_dir if expected[1] else None
    actual_root = metadata.get("replacement_state_root")
    if expected_root is None:
        if actual_root is not None:
            raise ValueError(f"Identity ANN evaluation unexpectedly uses replacement states: {path}")
    elif actual_root is None or Path(actual_root).resolve() != Path(expected_root).resolve():
        raise ValueError(f"Final aware ANN evaluation uses the wrong state root: {path}")
    expected_stage = "ann_training" if expected[1] else None
    if metadata.get("calibration_source_stage") != expected_stage:
        raise ValueError(f"Final ANN evaluation has incompatible calibration provenance: {path}")


def _require_manifest_flags(manifest, expected, label):
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"{label} manifest has invalid {key}: {manifest.get(key)!r}")


def _verify_hashes(manifest, label):
    for path_key, hash_key in (
        ("calibration_data_manifest_path", "calibration_data_manifest_sha256"),
        ("prefix_state_path", "prefix_state_sha256"),
        ("prefix_kv_path", "prefix_kv_sha256"),
        ("rotation_state_path", "rotation_state_sha256"),
        ("source_ann_checkpoint", None),
    ):
        path = manifest.get(path_key)
        expected = manifest.get(hash_key) if hash_key else None
        if path is None:
            if expected is not None:
                raise ValueError(f"{label} has {hash_key} without {path_key}")
            continue
        if hash_key and (not expected or sha256_file(path) != expected):
            raise ValueError(f"{label} provenance hash mismatch: {hash_key}")
    checkpoint = manifest.get("source_ann_checkpoint")
    expected_config = manifest.get("source_ann_config_sha256")
    if checkpoint is not None and (not expected_config or sha256_file(Path(checkpoint) / "config.json") != expected_config):
        raise ValueError(f"{label} provenance hash mismatch: source_ann_config_sha256")


def _verify_scalar_distribution(
    distribution,
    label,
    *,
    expected_count=None,
    include_max=True,
):
    if not isinstance(distribution, dict):
        raise ValueError(f"{label} must be a mapping")
    fields = ["mean", "p50", "p90", "p99"]
    if include_max:
        fields.append("max")
    if expected_count is not None:
        try:
            count = int(distribution.get("count", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} count is invalid") from exc
        if count != expected_count:
            raise ValueError(f"{label} count is inconsistent")
        if expected_count == 0:
            if any(distribution.get(field) is not None for field in fields):
                raise ValueError(f"{label} must use null statistics when empty")
            return
    if any(field not in distribution for field in fields):
        raise ValueError(f"{label} is missing scalar statistics")
    values = {field: float(distribution[field]) for field in fields}
    if not all(math.isfinite(value) and value >= 0.0 for value in values.values()):
        raise ValueError(f"{label} must contain finite non-negative statistics")
    if not values["p50"] <= values["p90"] <= values["p99"]:
        raise ValueError(f"{label} quantiles are not monotonic")
    if include_max and values["p99"] > values["max"] + 1e-12:
        raise ValueError(f"{label} P99 exceeds maximum")


def _verify_rotation_regression_metrics(regression):
    required = {
        "num_tokens_compared",
        "relative_l2_error",
        "max_abs_error",
        "mean_abs_error",
        "p99_abs_error",
        "p999_abs_error",
        "top1_agreement",
        "top1_agreement_count",
        "top1_disagreement_count",
        "absolute_error_percentile_estimator",
        "margin_aware_diagnostic",
    }
    missing = sorted(required - regression.keys())
    if missing:
        raise ValueError(f"Rotation regression is missing required metrics: {missing}")

    num_tokens = int(regression["num_tokens_compared"])
    agreement_count = int(regression["top1_agreement_count"])
    disagreement_count = int(regression["top1_disagreement_count"])
    agreement = float(regression["top1_agreement"])
    if num_tokens <= 0:
        raise ValueError("Rotation regression must compare at least one valid token")
    if not 0.0 <= agreement <= 1.0:
        raise ValueError("Rotation regression top1_agreement must be in [0, 1]")
    if agreement_count < 0 or disagreement_count < 0:
        raise ValueError("Rotation regression Top-1 counts must be non-negative")
    if agreement_count + disagreement_count != num_tokens:
        raise ValueError("Rotation regression Top-1 counts do not match compared tokens")
    expected_agreement = agreement_count / num_tokens
    if abs(expected_agreement - agreement) > 1e-12:
        raise ValueError("Rotation regression top1_agreement is inconsistent with its counts")

    nonnegative_metrics = {
        name: float(regression[name])
        for name in (
            "relative_l2_error",
            "mean_abs_error",
            "p99_abs_error",
            "p999_abs_error",
            "max_abs_error",
        )
    }
    if not all(math.isfinite(value) for value in nonnegative_metrics.values()):
        raise ValueError("Rotation regression error metrics must be finite")
    if any(value < 0.0 for value in nonnegative_metrics.values()):
        raise ValueError("Rotation regression error metrics must be non-negative")
    if nonnegative_metrics["p99_abs_error"] > nonnegative_metrics["p999_abs_error"]:
        raise ValueError("Rotation regression percentiles are not monotonic")

    estimator = regression["absolute_error_percentile_estimator"]
    if not isinstance(estimator, dict):
        raise ValueError("Rotation regression percentile estimator metadata must be a mapping")
    if estimator.get("method") != "streaming_linear_histogram":
        raise ValueError("Rotation regression uses an unknown percentile estimator")
    if int(estimator.get("num_bins", 0)) != 8192:
        raise ValueError("Rotation regression percentile estimator must use 8192 bins")
    if estimator.get("reported_value") != "bin_upper_edge":
        raise ValueError("Rotation regression percentiles must report bin upper edges")
    if estimator.get("exact") is not False:
        raise ValueError("Rotation regression histogram percentiles must be marked inexact")
    range_max = float(estimator.get("final_range_max", 0.0))
    if range_max <= 0.0:
        raise ValueError("Rotation regression histogram range must be positive")
    bin_width = range_max / int(estimator["num_bins"])
    if (
        nonnegative_metrics["p999_abs_error"]
        > nonnegative_metrics["max_abs_error"] + bin_width + 1e-12
    ):
        raise ValueError("Rotation regression P99.9 exceeds the observed maximum allowance")

    threshold = regression.get("threshold")
    if not isinstance(threshold, dict):
        raise ValueError("Rotation regression lacks hard thresholds")
    try:
        relative_l2_threshold = float(threshold["relative_l2_error"])
        top1_threshold = float(threshold["top1_agreement"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Rotation regression hard thresholds are incomplete") from exc
    if relative_l2_threshold <= 0.0 or not 0.0 <= top1_threshold < 1.0:
        raise ValueError("Rotation regression hard thresholds are invalid")
    expected_passed = (
        nonnegative_metrics["relative_l2_error"] <= relative_l2_threshold
        and agreement > top1_threshold
    )
    if bool(regression.get("passed")) != expected_passed:
        raise ValueError("Rotation regression passed flag contradicts its hard gates")

    margin = regression["margin_aware_diagnostic"]
    if not isinstance(margin, dict):
        raise ValueError("Rotation regression margin-aware diagnostic must be a mapping")
    if margin.get("definition") != "base_top1_margin_gt_2x_per_token_max_abs_error":
        raise ValueError("Rotation regression has an unknown margin-safe definition")
    required_margin_fields = (
        "margin_safe_token_count",
        "margin_unsafe_token_count",
        "margin_safe_agreement_count",
        "margin_safe_disagreement_count",
        "margin_unsafe_agreement_count",
        "margin_unsafe_disagreement_count",
        "margin_safe_fraction",
        "disagreement_margin_unsafe_fraction",
        "base_top1_margin_all_tokens",
        "per_token_max_abs_error_all_tokens",
        "base_top1_margin_disagreement_tokens",
        "per_token_max_abs_error_disagreement_tokens",
        "stability_ratio_disagreement_tokens",
    )
    if any(key not in margin for key in required_margin_fields):
        raise ValueError("Rotation regression margin-aware diagnostic is incomplete")
    required_margin_counts = required_margin_fields[:6]
    safe, unsafe, safe_agree, safe_disagree, unsafe_agree, unsafe_disagree = (
        int(margin[key]) for key in required_margin_counts
    )
    if any(value < 0 for value in (safe, unsafe, safe_agree, safe_disagree, unsafe_agree, unsafe_disagree)):
        raise ValueError("Rotation regression margin-aware counts must be non-negative")
    if safe + unsafe != num_tokens:
        raise ValueError("Rotation regression margin-safe partition does not match tokens")
    if safe_agree + safe_disagree != safe or unsafe_agree + unsafe_disagree != unsafe:
        raise ValueError("Rotation regression margin agreement partitions are inconsistent")
    if safe_agree + unsafe_agree != agreement_count:
        raise ValueError("Rotation regression margin agreement count mismatches Top-1")
    if safe_disagree + unsafe_disagree != disagreement_count:
        raise ValueError("Rotation regression margin disagreement count mismatches Top-1")
    if safe_disagree != 0:
        raise ValueError("Margin-safe token changed Top-1; regression metrics are misaligned")
    expected_safe_fraction = safe / num_tokens
    if abs(float(margin.get("margin_safe_fraction", float("nan"))) - expected_safe_fraction) > 1e-12:
        raise ValueError("Rotation regression margin_safe_fraction is inconsistent")
    expected_unsafe_disagreement_fraction = (
        1.0 if disagreement_count == 0 else unsafe_disagree / disagreement_count
    )
    if (
        abs(
            float(margin.get("disagreement_margin_unsafe_fraction", float("nan")))
            - expected_unsafe_disagreement_fraction
        )
        > 1e-12
    ):
        raise ValueError("Rotation regression disagreement margin-unsafe fraction is inconsistent")

    _verify_scalar_distribution(
        margin.get("base_top1_margin_all_tokens"),
        "All-token Base Top-1 margin distribution",
    )
    _verify_scalar_distribution(
        margin.get("per_token_max_abs_error_all_tokens"),
        "All-token per-token maximum error distribution",
    )
    delta_max = float(margin["per_token_max_abs_error_all_tokens"]["max"])
    if abs(delta_max - nonnegative_metrics["max_abs_error"]) > 1e-6:
        raise ValueError("Per-token maximum error distribution disagrees with root maximum")
    _verify_scalar_distribution(
        margin.get("base_top1_margin_disagreement_tokens"),
        "Disagreement Base Top-1 margin distribution",
        expected_count=disagreement_count,
    )
    _verify_scalar_distribution(
        margin.get("per_token_max_abs_error_disagreement_tokens"),
        "Disagreement per-token maximum error distribution",
        expected_count=disagreement_count,
    )
    ratio = margin.get("stability_ratio_disagreement_tokens")
    if not isinstance(ratio, dict) or ratio.get("definition") != "2_delta_over_base_top1_margin_plus_1e-12":
        raise ValueError("Rotation regression disagreement stability-ratio definition is invalid")
    _verify_scalar_distribution(
        ratio,
        "Disagreement stability-ratio distribution",
        expected_count=disagreement_count,
        include_max=False,
    )


def _verify_prompt_end_metrics(metrics, *, expected_prompts: int) -> None:
    required = {
        "num_prompts_compared",
        "relative_l2_error",
        "max_abs_error",
        "mean_abs_error",
        "top1_agreement",
        "top1_agreement_count",
        "top1_disagreement_count",
        "reference_top1_margin",
        "per_prompt_max_abs_error",
        "gating",
    }
    missing = sorted(required - metrics.keys())
    if missing:
        raise ValueError(f"Prompt-end diagnostic is missing required metrics: {missing}")
    if int(metrics["num_prompts_compared"]) != expected_prompts:
        raise ValueError("Prompt-end diagnostic must compare all 128 prompts")
    agreement = float(metrics["top1_agreement"])
    agreement_count = int(metrics["top1_agreement_count"])
    disagreement_count = int(metrics["top1_disagreement_count"])
    if not 0.0 <= agreement <= 1.0:
        raise ValueError("Prompt-end top1_agreement must be in [0, 1]")
    if agreement_count + disagreement_count != expected_prompts:
        raise ValueError("Prompt-end Top-1 counts do not match prompts")
    if abs(agreement - agreement_count / expected_prompts) > 1e-12:
        raise ValueError("Prompt-end Top-1 agreement is inconsistent with counts")
    for name in ("relative_l2_error", "max_abs_error", "mean_abs_error"):
        value = float(metrics[name])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"Prompt-end {name} must be finite and non-negative")
    if metrics["gating"] is not False:
        raise ValueError("Prompt-end diagnostic must not gate this protocol")
    _verify_scalar_distribution(
        metrics["reference_top1_margin"], "Prompt-end reference Top-1 margin"
    )
    _verify_scalar_distribution(
        metrics["per_prompt_max_abs_error"], "Prompt-end per-prompt maximum error"
    )


def _verify_rotation_regression_suite(regression) -> None:
    _require_manifest_flags(
        regression,
        {
            "format_version": 4,
            "purpose": "three_way_rotation_regression",
            "status": "passed",
            "num_samples": 128,
            "passed": True,
        },
        "Rotation regression",
    )
    implementation = regression.get("rotation_implementation")
    if not isinstance(implementation, dict):
        raise ValueError("Rotation regression lacks implementation metadata")
    if implementation.get("random_hadamard_orientation") != "DU":
        raise ValueError("Rotation regression must use random Hadamard orientation DU")
    if implementation.get("precision_policy") != "roste_aligned_v1":
        raise ValueError("Rotation regression must use precision policy roste_aligned_v1")
    hard_gate = regression.get("hard_gate")
    if not isinstance(hard_gate, dict) or hard_gate.get("scope") != "all_tokens_only":
        raise ValueError("Rotation regression hard gate must be all-token only")
    if hard_gate.get("prompt_end_is_gating") is not False:
        raise ValueError("Prompt-end diagnostic must not be a hard gate")

    comparisons = regression.get("comparisons")
    if not isinstance(comparisons, dict):
        raise ValueError("Rotation regression lacks comparisons")
    names = ("A_vs_B", "B_vs_C", "A_vs_C")
    if set(comparisons) != set(names):
        raise ValueError("Rotation regression must contain exactly A_vs_B, B_vs_C, A_vs_C")
    for name in names:
        pair = comparisons[name]
        if not isinstance(pair, dict) or "all_tokens" not in pair or "prompt_end" not in pair:
            raise ValueError(f"Rotation comparison {name} is incomplete")
        _verify_rotation_regression_metrics(pair["all_tokens"])
        _verify_prompt_end_metrics(pair["prompt_end"], expected_prompts=128)
        if bool(pair.get("passed")) != bool(pair["all_tokens"].get("passed")):
            raise ValueError(f"Rotation comparison {name} passed flag is inconsistent")
    expected_passed = all(bool(comparisons[name]["passed"]) for name in names)
    if bool(regression["passed"]) != expected_passed:
        raise ValueError("Rotation suite passed flag contradicts its three comparisons")
    diagnosis = regression.get("diagnosis")
    if not isinstance(diagnosis, dict) or not diagnosis.get("code"):
        raise ValueError("Rotation regression lacks structured diagnosis")


def main():
    args = parser(
        "Verify required artifacts for one ANN run"
    ).parse_args()

    cfg, layout = setup(args.config)

    with StageRun(
        "verify_artifacts",
        layout.logs_dir,
        cfg["experiment"],
    ) as run:

        required = [
            layout.data_dir / "train_manifest.json",
            layout.data_dir / "validation_manifest.json",
            layout.data_dir / "calibration_manifest.json",
            layout.ann_checkpoint_dir / "config.json",
            layout.ann_dir / "training_result.json",
        ]

        task = cfg["experiment"]["task"]

        if task == "tldr":
            required.append(
                layout.data_dir / "evaluation_manifest.json"
            )

        if task == "tldr":
            evaluation_manifest = read_json(layout.data_dir / "evaluation_manifest.json")
            tldr_layout = resolve_tldr_evaluation_layout(
                len(evaluation_manifest["indices"]),
                cfg["evaluation"].get("tldr_test_samples"),
            )
            eval_dir_name = "tldr"
            evaluation_files = ("metrics.json", "selection.json")
            evaluation_subdir = str(tldr_layout["dirname"])
        else:
            tldr_layout = None
            eval_dir_name = "lm_harness"
            evaluation_files = ("results.json",)
            evaluation_subdir = None

        def evaluation_paths(root):
            directory = root / "evaluation" / eval_dir_name
            if evaluation_subdir is not None:
                directory = directory / evaluation_subdir
            directory = directory / prefix_enabled_dirname(evaluation_prefix_enabled(cfg))
            return [directory / name for name in evaluation_files]

        required.extend(evaluation_paths(layout.ann_dir))

        # --------------------------------------------------
        # Rotation / Prefix shared artifacts
        # --------------------------------------------------
        prefix_state_path = layout.conversion_prefix_dir / "prefix_state.json"
        required.append(layout.conversion_site_dir / "calibration_state_manifest.json")
        if conversion_prefix_enabled(cfg) or evaluation_prefix_enabled(cfg):
            required.append(prefix_state_path)

        if cfg["rotation"]["enabled"]:

            required.extend(
                [
                    layout.rotation_dir
                    / "rotation_state.pt",

                    layout.rotation_dir
                    / "rotation_regression.json",

                    layout.rotation_dir
                    / "rotation_summary.json",

                    layout.rotation_dir
                    / "fused_base"
                    / "config.json",

                ]
            )
            if requires_pre_finetuning_prefix(cfg) and training_prefix_enabled(cfg):
                required.append(layout.ann_training_prefix_dir / "prefix_state.json")
            if requires_ann_training_calibration(cfg):
                required.append(layout.ann_training_site_dir / "calibration_state_manifest.json")

        # --------------------------------------------------
        # First existence check.
        #
        # prefix_state.json must exist before we can inspect
        # whether a non-empty Prefix KV cache is required.
        # --------------------------------------------------
        missing = [
            str(path)
            for path in required
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Missing required artifacts:\n"
                + "\n".join(missing)
            )

        if requires_pre_finetuning_prefix(cfg) and training_prefix_enabled(cfg):
            training_prefix_state_path = layout.ann_training_prefix_dir / "prefix_state.json"
            training_prefix_ids = [
                int(value) for value in read_json(training_prefix_state_path).get("prefix_token_ids", [])
            ]
            if training_prefix_ids:
                training_prefix_kv_path = layout.ann_training_prefix_dir / "prefixed_key_values.pt"
                if not training_prefix_kv_path.exists():
                    raise FileNotFoundError(
                        "Non-empty Pre-finetuning Prefix requires its fixed KV cache: "
                        f"{training_prefix_kv_path}"
                    )

        if cfg["rotation"]["enabled"]:
            regression_path = layout.rotation_dir / "rotation_regression.json"
            regression = read_json(regression_path)
            _verify_rotation_regression_suite(regression)
            summary = read_json(layout.rotation_dir / "rotation_summary.json")
            if (
                summary.get("random_hadamard_orientation") != "DU"
                or summary.get("precision_policy") != "roste_aligned_v1"
                or int(summary.get("rotation_regression_format_version", -1)) != 4
            ):
                raise ValueError("Rotation summary metadata is incompatible with regression v4")
            calibration_manifest = layout.data_dir / "calibration_manifest.json"
            recorded_manifest = regression.get("calibration_manifest_path")
            if (
                not recorded_manifest
                or Path(recorded_manifest).resolve() != calibration_manifest.resolve()
            ):
                raise ValueError(
                    "Rotation regression does not reference the current task calibration manifest"
                )
            if regression.get("calibration_manifest_sha256") != sha256_file(
                calibration_manifest
            ):
                raise ValueError("Rotation regression calibration manifest hash mismatch")

        # --------------------------------------------------
        # Prefix KV artifact
        #
        # Non-empty prefix_token_ids:
        #     prefixed_key_values.pt is mandatory.
        #
        # Empty prefix_token_ids:
        #     no Prefix KV cache is required.
        #
        # This is important for Qwen3, where Prefix discovery
        # is allowed to produce an empty Prefix.
        # --------------------------------------------------
        prefix_token_ids = []

        if prefix_state_path in required:
            prefix_state = read_json(
                prefix_state_path
            )

            prefix_token_ids = [
                int(value)
                for value in prefix_state.get(
                    "prefix_token_ids",
                    [],
                )
            ]

            if prefix_token_ids:
                prefix_kv_path = layout.conversion_prefix_dir / "prefixed_key_values.pt"

                required.append(
                    prefix_kv_path
                )

                if not prefix_kv_path.exists():
                    raise FileNotFoundError(
                        "Prefix discovery produced "
                        "non-empty prefix_token_ids, "
                        "but the fixed Prefix KV cache "
                        "is missing:\n"
                        f"{prefix_kv_path}\n"
                        "Re-run "
                        "scripts/discover_prefix.py."
                    )

        # --------------------------------------------------
        # Calibration
        # --------------------------------------------------
        reused = conversion_reuses_ann_training_artifacts(cfg)
        calibration = validate_calibration(
            layout.conversion_site_dir,
            allow_clip_bundle=reused,
        )
        source_manifest = read_json(
            layout.conversion_site_dir / "calibration_state_manifest.json"
        )
        expected_flags = (
            {
                "purpose": "ann_training_calibration",
                "eligible_for_ann_training": True,
                "eligible_for_conversion": True,
                "conversion_reuse_policy": "aware_modes_only",
                "post_finetuning_recalibration": False,
                "state_profile": "ann_training_with_common_clip",
                "common_clip_required": True,
                "common_clip_generated": True,
                "common_clip_application_control": "replacement.common_clip_enabled",
            }
            if reused
            else {
                "purpose": "post_finetuning_conversion_calibration",
                "eligible_for_ann_training": False,
                "eligible_for_conversion": True,
                "conversion_reuse_policy": "final_ann_only",
                "post_finetuning_recalibration": True,
                "state_profile": "snn_conversion_without_clip",
                "common_clip_required": False,
                "common_clip_generated": False,
                "common_clip_application_control": "replacement.common_clip_enabled",
            }
        )
        _require_manifest_flags(source_manifest, expected_flags, "Conversion source")
        _verify_hashes(source_manifest, "Conversion source calibration")
        validate_temporal_policy(source_manifest, context="Conversion source manifest")
        validate_conversion_prefix(cfg, layout)

        conversions = {}

        for neuron in (
            "phase",
            "gif",
            "mtn",
        ):
            path = (
                layout.snn_conversion_dir(neuron)
                / "conversion_metadata.json"
            )

            conversions[neuron] = path.exists()

            required.extend(
                [
                    path,

                    *evaluation_paths(layout.snn_dir(neuron)),
                ]
            )

        missing = [
            str(path)
            for path in required
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Missing required artifacts:\n"
                + "\n".join(missing)
            )

        _verify_final_ann_forward_metadata(
            cfg, layout, evaluation_paths(layout.ann_dir)[0]
        )

        for neuron in ("phase", "gif", "mtn"):
            metadata_path = layout.snn_conversion_dir(neuron) / "conversion_metadata.json"
            metadata = validate_conversion_metadata(cfg, layout, neuron)
            if (
                int(metadata.get("full_temporal_steps", -1))
                != int(calibration["temporal_steps"][neuron])
            ):
                raise ValueError(
                    f"Conversion has incompatible calibration timestep metadata: {neuron}"
                )
            metrics_path = evaluation_paths(layout.snn_dir(neuron))[0]
            metrics = read_json(metrics_path)
            policy_source = metrics.get("snn2_metadata", metrics)
            validate_temporal_policy(policy_source, context=str(metrics_path))
            if (
                policy_source.get("neuron") != neuron
                or int(policy_source.get("full_temporal_steps", -1))
                != int(metadata["full_temporal_steps"])
            ):
                raise ValueError(f"SNN metrics disagree with conversion: {metrics_path}")
            if "evaluation_forward_kind" in policy_source:
                expected_forward = {
                    "evaluation_forward_kind": f"temporal_{neuron}_snn",
                    "controller_mode": f"deploy_{neuron}",
                    "temporal_execution": True,
                    "evaluation_common_clip_applied": False,
                }
                if any(policy_source.get(key) != value for key, value in expected_forward.items()):
                    raise ValueError(f"SNN metrics have incompatible temporal forward metadata: {metrics_path}")

        if tldr_layout is not None:
            expected_count = int(tldr_layout["selected_test_samples"])
            expected_full = bool(tldr_layout["is_full_test"])
            expected_sampling = (
                "full_split" if expected_full else "seeded_random_without_replacement"
            )
            for root in [layout.ann_dir, *(layout.snn_dir(neuron) for neuron in ("phase", "gif", "mtn"))]:
                selection_path = evaluation_paths(root)[1]
                selection = read_json(selection_path)
                if len(selection.get("indices", [])) != expected_count:
                    raise ValueError(f"TL;DR selection size mismatch: {selection_path}")
                if selection.get("sampling") != expected_sampling:
                    raise ValueError(f"TL;DR sampling policy mismatch: {selection_path}")
                if not expected_full and selection.get("seed") != int(cfg["evaluation"].get("tldr_test_seed", 42)):
                    raise ValueError(f"TL;DR selection seed mismatch: {selection_path}")

        result = {
            "required_files": len(required),
            "calibration": calibration,
            "conversion_descriptors": conversions,

            # Prefix verification metadata
            "prefix_token_ids": prefix_token_ids,
            "prefix_length": len(prefix_token_ids),
            "prefix_kv_required": bool(
                prefix_token_ids
            ),
            **topology_metadata(),
        }

        write_json(
            layout.root
            / "artifact_verification.json",
            result,
        )

        run.event(
            "verification_complete",
            **result,
        )


if __name__ == "__main__":
    main()
