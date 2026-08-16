from _common import parser, setup

from snn2.logging_utils import StageRun
from snn2.training import train_full_parameters


def main():
    args = parser("Full-parameter conversion-aware ANN fine-tuning").parse_args()
    cfg, layout = setup(args.config)
    with StageRun("train_ann", layout.logs_dir, cfg["experiment"]) as run:
        result = train_full_parameters(cfg, layout)
        run.event("training_complete", **result)


if __name__ == "__main__":
    main()

