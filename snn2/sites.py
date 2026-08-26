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
ATTENTION_HEAD_GROUPED_SITE_IDS = frozenset({2, 3, 4, 6})
SOFTMAX_SITE_ID = 5
CLIP_ELIGIBLE_SITE_IDS = frozenset({1, 2, 3, 4, 6, 7, 8, 9, 10})


def is_attention_head_grouped_site(site_index: int) -> bool:
    return int(site_index) in ATTENTION_HEAD_GROUPED_SITE_IDS


def is_softmax_site(site_index: int) -> bool:
    return int(site_index) == SOFTMAX_SITE_ID


def site_supports_clip(site_index: int) -> bool:
    return int(site_index) in CLIP_ELIGIBLE_SITE_IDS


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


def validate_site_topology(
    root: str | Path,
    *,
    expected_num_hidden_layers: int | None = None,
) -> dict[str, set[str]]:
    """Require complete layers and exactly the current site set in each layer."""
    root = Path(root)
    layers = sorted(path for path in root.glob("layer_*") if path.is_dir())
    if not layers:
        raise FileNotFoundError(f"No calibration layers under {root}")
    if expected_num_hidden_layers is not None:
        if (
            not isinstance(expected_num_hidden_layers, int)
            or isinstance(expected_num_hidden_layers, bool)
            or expected_num_hidden_layers <= 0
        ):
            raise ValueError("expected_num_hidden_layers must be a positive integer")
        expected_layers = {
            f"layer_{index:03d}" for index in range(expected_num_hidden_layers)
        }
        actual_layers = {path.name for path in layers}
        if actual_layers != expected_layers:
            raise RuntimeError(
                "Calibration layer topology is incomplete or non-contiguous "
                f"(expected_num_hidden_layers={expected_num_hidden_layers}, "
                f"actual_num_hidden_layers={len(actual_layers)}, "
                f"missing_layers={sorted(expected_layers - actual_layers)}, "
                f"unexpected_layers={sorted(actual_layers - expected_layers)})"
            )
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
