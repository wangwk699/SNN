from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from snn2.config import resolve_config, save_yaml, validate_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the twelve main ANN run configs")
    parser.add_argument(
        "--matrix",
        default=str(CODE_ROOT / "configs" / "experiment_matrix.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(CODE_ROOT / "configs" / "generated"),
    )
    args = parser.parse_args()
    with Path(args.matrix).open("r", encoding="utf-8") as handle:
        matrix = yaml.safe_load(handle)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for experiment in matrix["experiments"]:
        for mode in experiment["ann_modes"]:
            cfg = copy.deepcopy(experiment["config"])
            cfg.setdefault("experiment", {})["ann_mode"] = mode
            cfg = resolve_config(cfg)
            validate_config(cfg)
            path = output / f"{experiment['name']}__{mode}.yaml"
            save_yaml(cfg, path)
            print(path)
            count += 1
    if count != 12:
        raise RuntimeError(f"Expected 12 ANN configs, generated {count}")


if __name__ == "__main__":
    main()
