from _common import parser, setup

from snn2.artifacts import ArtifactLayout
from snn2.config import load_config
from snn2.data import prepare_calibration_manifest, prepare_manifests
from snn2.logging_utils import StageRun


def main():
    arg_parser = parser("Create deterministic train/validation/calibration manifests")
    arg_parser.add_argument(
        "--calibration-only",
        action="store_true",
        help="Write only the current config's Stage-A calibration manifest.",
    )
    args = arg_parser.parse_args()
    if args.calibration_only:
        # Do not call setup(): it writes a resolved config under the shared
        # artifact directory. This mode must write only the requested manifest.
        cfg = load_config(args.config)
        layout = ArtifactLayout(cfg)
        manifest = prepare_calibration_manifest(cfg, layout)
        print(
            "Calibration manifest saved: "
            f"{layout.calibration_data_manifest_path} "
            f"({len(manifest['indices'])} samples)"
        )
        return
    cfg, layout = setup(args.config, config_scope="task_shared")
    with StageRun("prepare_data", layout.shared_task_logs_dir, cfg["experiment"],) as run:
        manifests = prepare_manifests(cfg, layout)
        run.event(
            "manifests_saved",
            train=len(manifests["train"]["indices"]),
            validation=len(manifests["validation"]["indices"]),
            calibration=len(manifests["calibration"]["indices"]),
        )


if __name__ == "__main__":
    main()
