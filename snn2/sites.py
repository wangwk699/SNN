from __future__ import annotations

from pathlib import Path
from typing import Any

SITE_TOPOLOGY_VERSION = 2
SITE_NAMES = {
    1: "post_input_rmsnorm", 2: "q_post_rope_r3", 3: "k_post_rope_r3",
    4: "v_projection_r2", 5: "post_spiking_softmax", 6: "post_attention_value_dot_r2",
    7: "post_mlp_rmsnorm", 8: "post_spiking_silu", 9: "post_mlp_up_proj",
    10: "post_mlp_product_r4",
}
SITE_COORDINATES = {
    1: "R1", 2: "R3", 3: "R3", 4: "R2", 5: "I", 6: "R2", 7: "R1",
    8: "I", 9: "I", 10: "R4",
}
SITE_IDS = tuple(sorted(SITE_NAMES))
SITE_COUNT = len(SITE_IDS)


def site_key(layer_index: int, site_index: int) -> str:
    if site_index not in SITE_NAMES:
        raise ValueError(f"Unknown activation replacement site: {site_index}")
    return f"layer_{layer_index:03d}/site_{site_index:02d}_{SITE_NAMES[site_index]}"


def expected_site_dirnames() -> set[str]:
    return {f"site_{index:02d}_{SITE_NAMES[index]}" for index in SITE_IDS}


def topology_metadata() -> dict[str, Any]:
    return {
        "site_topology_version": SITE_TOPOLOGY_VERSION,
        "site_count": SITE_COUNT,
        "site_names": {str(index): name for index, name in SITE_NAMES.items()},
        "site_coordinates": {str(index): value for index, value in SITE_COORDINATES.items()},
    }


def validate_site_topology(root: str | Path) -> dict[str, set[str]]:
    """Require every calibration layer to contain exactly the current site set."""
    root = Path(root)
    layers = sorted(path for path in root.glob("layer_*") if path.is_dir())
    if not layers:
        raise FileNotFoundError(f"No calibration layers under {root}")
    expected = expected_site_dirnames()
    actual_by_layer: dict[str, set[str]] = {}
    invalid: dict[str, dict[str, list[str]]] = {}
    for layer in layers:
        actual = {path.name for path in layer.glob("site_*") if path.is_dir()}
        actual_by_layer[layer.name] = actual
        if actual != expected:
            invalid[layer.name] = {
                "missing": sorted(expected - actual),
                "unexpected": sorted(actual - expected),
            }
    if invalid:
        raise RuntimeError(
            "Calibration site topology does not match the current topology "
            f"(version={SITE_TOPOLOGY_VERSION}, sites={SITE_COUNT}): {invalid}. "
            "Remove or move stale calibration sites before recalibrating."
        )
    return actual_by_layer
