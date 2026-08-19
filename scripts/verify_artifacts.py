from __future__ import annotations

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
        if cfg["rotation"]["enabled"]:
            ann_manifest = read_json(layout.ann_training_site_dir / "calibration_state_manifest.json")
            _require_manifest_flags(ann_manifest, {"purpose": "ann_training_calibration", "analysis_only": False, "eligible_for_ann_training": True, "eligible_for_conversion": False, "post_finetuning_recalibration": False, "rotation_enabled": True, "prefix_protocol_enabled": True}, "ANN-training")
            if not ann_manifest.get("calibration_data_manifest_sha256"):
                raise ValueError("ANN-training calibration lacks calibration data provenance")
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