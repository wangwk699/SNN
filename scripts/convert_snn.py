from _common import parser, setup

from snn2.conversion import create_conversion
from snn2.logging_utils import StageRun


def main():
    args = parser("Create a frozen full-temporal SNN conversion descriptor", neuron=True).parse_args()
    cfg, layout = setup(args.config)
    stage = f"convert_snn_{args.neuron}"
    with StageRun(stage, layout.logs_dir, cfg["experiment"]) as run:
        metadata = create_conversion(cfg, layout, args.neuron)
        run.event("conversion_created", **metadata)


if __name__ == "__main__":
    main()

