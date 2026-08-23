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


def materialize_configs(
    matrix_path: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    with Path(matrix_path).open("r", encoding="utf-8") as handle:
        matrix = yaml.safe_load(handle)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for experiment in matrix["experiments"]:
        for mode in experiment["ann_modes"]:
            cfg = copy.deepcopy(experiment["config"])
            cfg.setdefault("experiment", {})["ann_mode"] = mode
            cfg = resolve_config(cfg)
            validate_config(cfg)
            path = output / f"{experiment['name']}__{mode}.yaml"
            save_yaml(cfg, path)
            generated.append(path)
    if len(generated) != 12:
        raise RuntimeError(f"Expected 12 ANN configs, generated {len(generated)}")
    return generated


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
    for path in materialize_configs(args.matrix, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
