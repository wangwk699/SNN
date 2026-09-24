from __future__ import annotations

from _common import apply_deployment_overrides, parser, setup
from snn2.conversion import validate_conversion_metadata
from snn2.energy_profiler import profile_energy
from snn2.logging_utils import StageRun


def main() -> None:
    args = parser(
        "Profile theoretical MAC/AC energy on fixed held-out 512-token sequences",
        neuron=True,
        allow_ann=True,
    ).parse_args()
    cfg, layout = setup(args.config)
    apply_deployment_overrides(args, cfg)
    if args.neuron != "ann":
        try:
            validate_conversion_metadata(cfg, layout, args.neuron)
        except (FileNotFoundError, ValueError) as exc:
            raise type(exc)(
                f"{exc}\nRun `python scripts/convert_snn.py --config {args.config} --neuron {args.neuron}` first."
            ) from exc
    with StageRun(f"profile_energy_{args.neuron}", layout.energy_dir(args.neuron) / "logs", cfg["experiment"]):
        result = profile_energy(cfg, layout, neuron=args.neuron)
    print(result)
    print(layout.energy_dir(args.neuron))


if __name__ == "__main__":
    main()
