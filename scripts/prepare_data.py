from _common import parser, setup

from snn2.data import prepare_manifests
from snn2.logging_utils import StageRun


def main():
    args = parser("Create deterministic train/validation/calibration manifests").parse_args()
    cfg, layout = setup(args.config)
    with StageRun("prepare_data", layout.logs_dir, cfg["experiment"]) as run:
        manifests = prepare_manifests(cfg, layout)
        run.event(
            "manifests_saved",
            train=len(manifests["train"]["indices"]),
            validation=len(manifests["validation"]["indices"]),
            calibration=len(manifests["calibration"]["indices"]),
        )


if __name__ == "__main__":
    main()

