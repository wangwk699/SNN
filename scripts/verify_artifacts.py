from __future__ import annotations

from _common import parser, setup

from snn2.artifacts import write_json
from snn2.conversion import validate_calibration
from snn2.logging_utils import StageRun


def main():
    args = parser("Verify required artifacts for one ANN run").parse_args()
    cfg, layout = setup(args.config)
    with StageRun("verify_artifacts", layout.logs_dir, cfg["experiment"]) as run:
        required = [
            layout.data_dir / "train_manifest.json",
            layout.data_dir / "validation_manifest.json",
            layout.data_dir / "calibration_manifest.json",
            layout.ann_dir / "best" / "config.json",
            layout.ann_dir / "training_result.json",
        ]
        task = cfg["experiment"]["task"]
        if task == "tldr":
            required.append(layout.data_dir / "evaluation_manifest.json")
        eval_name = "metrics.json" if task == "tldr" else "results.json"
        eval_dir_name = "tldr" if task == "tldr" else "lm_harness"
        required.append(layout.ann_dir / "evaluation" / eval_dir_name / eval_name)
        if cfg["rotation"]["enabled"]:
            required.extend(
                [
                    layout.rotation_dir / "rotation_state.pt",
                    layout.rotation_dir / "fused_base" / "config.json",
                    layout.prefix_dir / "prefix_state.json",
                ]
            )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))
        calibration = validate_calibration(layout.site_dir)
        conversions = {}
        for neuron in ("phase", "gif", "mtn"):
            path = layout.snn_dir(neuron) / "conversion_metadata.json"
            conversions[neuron] = path.exists()
            required.extend(
                [
                    path,
                    layout.snn_dir(neuron) / "evaluation" / eval_dir_name / eval_name,
                ]
            )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))
        result = {
            "required_files": len(required),
            "calibration": calibration,
            "conversion_descriptors": conversions,
        }
        write_json(layout.root / "artifact_verification.json", result)
        run.event("verification_complete", **result)


if __name__ == "__main__":
    main()
