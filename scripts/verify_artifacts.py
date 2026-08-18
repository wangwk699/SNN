from __future__ import annotations

from _common import parser, setup

from snn2.artifacts import read_json, write_json
from snn2.conversion import validate_calibration
from snn2.sites import topology_metadata
from snn2.logging_utils import StageRun


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
            layout.ann_dir / "best" / "config.json",
            layout.ann_dir / "training_result.json",
        ]

        task = cfg["experiment"]["task"]

        if task == "tldr":
            required.append(
                layout.data_dir / "evaluation_manifest.json"
            )

        eval_name = (
            "metrics.json"
            if task == "tldr"
            else "results.json"
        )

        eval_dir_name = (
            "tldr"
            if task == "tldr"
            else "lm_harness"
        )

        required.append(
            layout.ann_dir
            / "evaluation"
            / eval_dir_name
            / eval_name
        )

        # --------------------------------------------------
        # Rotation / Prefix shared artifacts
        # --------------------------------------------------
        prefix_state_path = None

        if cfg["rotation"]["enabled"]:
            prefix_state_path = (
                layout.prefix_dir
                / "prefix_state.json"
            )

            required.extend(
                [
                    layout.rotation_dir
                    / "rotation_state.pt",

                    layout.rotation_dir
                    / "fused_base"
                    / "config.json",

                    prefix_state_path,
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
                    layout.prefix_dir
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
            layout.site_dir
        )

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

                    layout.snn_dir(neuron)
                    / "evaluation"
                    / eval_dir_name
                    / eval_name,
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