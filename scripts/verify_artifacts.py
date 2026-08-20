from __future__ import annotations
import math
from pathlib import Path
from _common import parser, setup

from snn2.artifacts import read_json, sha256_file, write_json
from snn2.conversion import validate_calibration
from snn2.sites import topology_metadata
from snn2.evaluation import resolve_tldr_evaluation_layout
from snn2.logging_utils import StageRun


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
    if not isinstance(threshold, dict) or "relative_l2_error" not in threshold:
        raise ValueError("Rotation regression lacks the relative-L2 hard threshold")
    if nonnegative_metrics["relative_l2_error"] > float(threshold["relative_l2_error"]):
        raise ValueError("Rotation regression passed flag contradicts its relative-L2 hard gate")


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
            return [directory / name for name in evaluation_files]

        required.extend(evaluation_paths(layout.ann_dir))

        # --------------------------------------------------
        # Rotation / Prefix shared artifacts
        # --------------------------------------------------
        prefix_state_path = layout.post_finetuning_prefix_dir / "prefix_state.json"
        required.extend([
            prefix_state_path,
            layout.post_finetuning_site_dir / "calibration_state_manifest.json",
            layout.vanilla_analysis_site_dir / "statistics_manifest.json",
        ])

        if cfg["rotation"]["enabled"]:

            required.extend(
                [
                    layout.rotation_dir
                    / "rotation_state.pt",

                    layout.rotation_dir
                    / "rotation_regression.json",

                    layout.rotation_dir
                    / "fused_base"
                    / "config.json",

                    layout.ann_training_prefix_dir / "prefix_state.json",
                    layout.ann_training_site_dir / "calibration_state_manifest.json",

                ]
            )

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

        if cfg["rotation"]["enabled"]:
            regression_path = layout.rotation_dir / "rotation_regression.json"
            regression = read_json(regression_path)
            _require_manifest_flags(
                regression,
                {
                    "format_version": 2,
                    "purpose": "base_vs_rotated_logits_regression",
                    "num_samples": 128,
                    "passed": True,
                },
                "Rotation regression",
            )
            _verify_rotation_regression_metrics(regression)
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

        if prefix_state_path is not None:
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
                prefix_kv_path = (
                    layout.post_finetuning_prefix_dir
                    / "prefixed_key_values.pt"
                )

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
        calibration = validate_calibration(
            layout.post_finetuning_site_dir
        )

        vanilla_manifest = read_json(layout.vanilla_analysis_site_dir / "statistics_manifest.json")
        _require_manifest_flags(vanilla_manifest, {"purpose": "vanilla_analysis_calibration", "analysis_only": True, "eligible_for_ann_training": False, "eligible_for_conversion": False, "post_finetuning_recalibration": False, "rotation_enabled": False, "prefix_protocol_enabled": False}, "Vanilla analysis")
        _verify_hashes(vanilla_manifest, "Vanilla analysis calibration")
        if cfg["rotation"]["enabled"]:
            ann_manifest = read_json(layout.ann_training_site_dir / "calibration_state_manifest.json")
            _require_manifest_flags(ann_manifest, {"purpose": "ann_training_calibration", "analysis_only": False, "eligible_for_ann_training": True, "eligible_for_conversion": False, "post_finetuning_recalibration": False, "rotation_enabled": True, "prefix_protocol_enabled": True}, "ANN-training")
            _verify_hashes(ann_manifest,"ANN-training calibration")
        post_manifest = read_json(layout.post_finetuning_site_dir / "calibration_state_manifest.json")
        _require_manifest_flags(post_manifest, {"purpose": "post_finetuning_conversion_calibration", "analysis_only": False, "eligible_for_ann_training": False, "eligible_for_conversion": True, "post_finetuning_recalibration": True, "prefix_protocol_enabled": True}, "Post-finetuning")
        if not post_manifest.get("source_ann_checkpoint") or not post_manifest.get("source_ann_config_sha256") or not post_manifest.get("calibration_data_manifest_sha256"):
            raise ValueError("Post-finetuning calibration lacks required final-ANN or data provenance")
        expected_rotation = bool(cfg["rotation"]["enabled"])
        if post_manifest.get("rotation_enabled") != expected_rotation:
            raise ValueError("Post-finetuning calibration rotation provenance disagrees with config")
        _verify_hashes(post_manifest, "Post-finetuning calibration")

        conversions = {}

        for neuron in (
            "phase",
            "gif",
            "mtn",
        ):
            path = (
                layout.snn_dir(neuron)
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

        for neuron in ("phase", "gif", "mtn"):
            metadata = read_json(layout.snn_dir(neuron) / "conversion_metadata.json")
            if not metadata.get("post_finetuning_recalibration") or "post_finetuning/conversion_calibration/sites" not in metadata.get("calibration_root", ""):
                raise ValueError(f"Conversion does not use run-specific post-finetuning calibration: {neuron}")

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