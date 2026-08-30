from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from snn2.artifacts import write_json
from snn2.calibration import materialize_calibration_states
from snn2.controller import SiteController
from snn2.evaluation import (
    evaluation_calibration_metadata,
    evaluation_forward_metadata,
)
from snn2.phase_statistics import (
    PHASE_TAU_ACCUMULATOR_DTYPE,
    PHASE_TAU_CALIBRATION,
    PHASE_TAU_CHANNEL_POLICY,
    PHASE_TAU_EMA_FACTOR,
    PHASE_TAU_REDUCTION_POLICY,
)
from snn2.sites import SITE_IDS, SITE_NAMES, site_key
from snn2.temporal_ops import STATISTICS_FORMAT_VERSION


def _load_verify_artifacts_module():
    """
    Load scripts/verify_artifacts.py as a module.

    The script imports `_common` as a top-level module, so its directory must
    temporarily be on sys.path while the module is executed.
    """
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    module_path = scripts_dir / "verify_artifacts.py"
    module_name = "_snn2_test_verify_artifacts_script"

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(scripts_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(scripts_dir))
    return module


_VERIFY = _load_verify_artifacts_module()


NUM_ATTENTION_HEADS = 8
NUM_KEY_VALUE_HEADS = 2
HEAD_DIM = 4
HIDDEN_SIZE = NUM_ATTENTION_HEADS * HEAD_DIM


def _cfg(ann_mode: str = "vanilla") -> dict:
    return {
        "experiment": {
            "ann_mode": ann_mode,
        },
        "replacement": {
            "common_clip_enabled": False,
        },
        "calibration": {
            "group_size": -1,
            "num_samples": 128,
            "expected_sites_per_layer": 10,
        },
        "phase": {
            "T": 2,
            "base": 2.0,
            "surrogate_slope": 1.0,
        },
        "gif": {
            "base_bits": 4,
            "add_bits": 1,
            "low_ratio": 0.5,
        },
        "mtn": {
            "T": 2,
            "K": 2,
            "threshold_factor": 0.75,
        },
    }


def _saliency_roles(site_index: int | None) -> tuple[str, ...]:
    if site_index == 1:
        return ("q", "k", "v")
    if site_index == 7:
        return ("gate", "up")
    if site_index in {3, 4, 6, 10}:
        return ("default",)
    return ()


def _saliency_rule(site_index: int | None) -> str:
    if site_index == 3:
        return "spikellm_qk_k_fp64"
    if site_index == 4:
        return "spikellm_pv_v_fp64"
    return "spikellm_linear_fp32"


def _statistics(site_index: int | None) -> dict:
    if site_index in {2, 3, 4}:
        shape = (NUM_ATTENTION_HEADS, HEAD_DIM)
        layout_kind = "attention_head"
        num_heads = NUM_ATTENTION_HEADS
        channels_per_head = HEAD_DIM
        channels = HIDDEN_SIZE
    elif site_index == 5:
        shape = (NUM_ATTENTION_HEADS,)
        layout_kind = "attention_softmax"
        num_heads = NUM_ATTENTION_HEADS
        channels_per_head = None
        channels = NUM_ATTENTION_HEADS
    else:
        shape = (HIDDEN_SIZE,)
        layout_kind = "last_dim"
        num_heads = None
        channels_per_head = None
        channels = HIDDEN_SIZE

    roles = _saliency_roles(site_index)
    saliency_dtype = (
        torch.float64 if site_index in {3, 4} else torch.float32
    )
    rule = _saliency_rule(site_index)

    return {
        "format_version": STATISTICS_FORMAT_VERSION,
        "site_index": site_index,
        "layout_kind": layout_kind,
        "num_heads": num_heads,
        "channels_per_head": channels_per_head,
        "channels": channels,
        "value_min": torch.full(shape, -1.0, dtype=torch.float64),
        "value_max": torch.full(shape, 1.0, dtype=torch.float64),
        "saliency_row_count_by_role": {
            role: torch.ones(shape, dtype=torch.long)
            for role in roles
        },
        "saliency_sum_by_role": {
            role: torch.zeros(shape, dtype=saliency_dtype)
            for role in roles
        },
        "saliency_rule_by_role": {
            role: rule
            for role in roles
        },
        "saliency_accumulator_dtype_by_role": {
            role: (
                "float64"
                if saliency_dtype == torch.float64
                else "float32"
            )
            for role in roles
        },
        "phase_ema_abs_max": torch.ones(shape, dtype=torch.float32),
        "phase_ema_updates": torch.ones(shape, dtype=torch.long),
        "phase_tau_calibration": PHASE_TAU_CALIBRATION,
        "phase_tau_ema_factor": PHASE_TAU_EMA_FACTOR,
        "phase_tau_accumulator_dtype": PHASE_TAU_ACCUMULATOR_DTYPE,
        "phase_tau_channel_policy": PHASE_TAU_CHANNEL_POLICY,
        "phase_tau_reduction_policy": PHASE_TAU_REDUCTION_POLICY,
    }


def _write_calibration_bundle(root: Path, cfg: dict) -> dict:
    root.mkdir(parents=True, exist_ok=True)

    for site_index in SITE_IDS:
        directory = root / site_key(0, site_index)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            _statistics(site_index),
            directory / "statistics.pt",
        )

    global_dir = root / "_global" / "final_rmsnorm"
    global_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        _statistics(None),
        global_dir / "statistics.pt",
    )

    return materialize_calibration_states(
        root,
        cfg,
        expected_num_hidden_layers=1,
    )


@pytest.fixture
def grouped_calibration_bundle(tmp_path):
    cfg = _cfg("vanilla")
    site_root = tmp_path / "sites"
    manifest = _write_calibration_bundle(site_root, cfg)

    ann_checkpoint_dir = tmp_path / "ann" / "final"
    ann_checkpoint_dir.mkdir(parents=True)
    write_json(
        ann_checkpoint_dir / "config.json",
        {
            "num_attention_heads": NUM_ATTENTION_HEADS,
            "num_key_value_heads": NUM_KEY_VALUE_HEADS,
            "head_dim": HEAD_DIM,
            "hidden_size": HIDDEN_SIZE,
        },
    )

    layout = SimpleNamespace(
        ann_checkpoint_dir=ann_checkpoint_dir,
        conversion_site_dir=site_root,
    )
    calibration = {"layers": 1}

    return SimpleNamespace(
        cfg=cfg,
        site_root=site_root,
        manifest=manifest,
        layout=layout,
        calibration=calibration,
    )


def _statistics_path(bundle, site_index: int) -> Path:
    return (
        bundle.site_root
        / site_key(0, site_index)
        / "statistics.pt"
    )


def _state_path(
    bundle,
    site_index: int,
    state_name: str,
) -> Path:
    return (
        bundle.site_root
        / site_key(0, site_index)
        / f"{state_name}_state.pt"
    )


def _verify_grouped(bundle, manifest: dict | None = None) -> None:
    _VERIFY._verify_grouped_calibration(
        bundle.cfg,
        bundle.layout,
        bundle.manifest if manifest is None else manifest,
        bundle.calibration,
    )


def test_verify_grouped_calibration_accepts_post_repeat_gqa_geometry(
    grouped_calibration_bundle,
):
    """
    Regression for the Site 3/4 topology change.

    In a GQA model with H_attn=8 and H_kv=2, Sites 3/4 must use the
    repeated/query-head coordinate H_attn=8, not the native KV-head count.
    """
    bundle = grouped_calibration_bundle

    for site_index in (2, 3, 4):
        statistics = torch.load(
            _statistics_path(bundle, site_index),
            map_location="cpu",
            weights_only=False,
        )
        assert statistics["layout_kind"] == "attention_head"
        assert statistics["num_heads"] == NUM_ATTENTION_HEADS
        assert statistics["channels_per_head"] == HEAD_DIM
        assert statistics["channels"] == HIDDEN_SIZE

    site6 = torch.load(
        _statistics_path(bundle, 6),
        map_location="cpu",
        weights_only=False,
    )
    assert site6["layout_kind"] == "last_dim"
    assert site6["num_heads"] is None
    assert site6["channels_per_head"] is None
    assert site6["channels"] == HIDDEN_SIZE

    _verify_grouped(bundle)


@pytest.mark.parametrize("site_index", [3, 4])
def test_verify_grouped_calibration_rejects_native_kv_head_statistics(
    grouped_calibration_bundle,
    site_index,
):
    """
    Old pre-repeat Site 3/4 statistics used native KV heads. They must now be
    rejected even when their local H_kv*D shape is internally consistent.
    """
    bundle = grouped_calibration_bundle
    path = _statistics_path(bundle, site_index)
    statistics = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    statistics["num_heads"] = NUM_KEY_VALUE_HEADS
    statistics["channels"] = NUM_KEY_VALUE_HEADS * HEAD_DIM
    statistics["value_min"] = torch.full(
        (NUM_KEY_VALUE_HEADS, HEAD_DIM),
        -1.0,
        dtype=torch.float64,
    )
    statistics["value_max"] = torch.full(
        (NUM_KEY_VALUE_HEADS, HEAD_DIM),
        1.0,
        dtype=torch.float64,
    )
    torch.save(statistics, path)

    with pytest.raises(
        ValueError,
        match="repeated/query attention heads",
    ):
        _verify_grouped(bundle)


def test_verify_grouped_calibration_rejects_old_per_head_site6_statistics(
    grouped_calibration_bundle,
):
    bundle = grouped_calibration_bundle
    path = _statistics_path(bundle, 6)
    statistics = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    statistics.update(
        {
            "layout_kind": "attention_head",
            "num_heads": NUM_ATTENTION_HEADS,
            "channels_per_head": HEAD_DIM,
            "channels": HIDDEN_SIZE,
        }
    )
    torch.save(statistics, path)

    with pytest.raises(
        ValueError,
        match="Site 6 must use merged last_dim statistics",
    ):
        _verify_grouped(bundle)


def test_verify_grouped_calibration_rejects_wrong_site6_merged_width(
    grouped_calibration_bundle,
):
    bundle = grouped_calibration_bundle
    path = _statistics_path(bundle, 6)
    statistics = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    statistics["channels"] = HIDDEN_SIZE - 1
    torch.save(statistics, path)

    with pytest.raises(
        ValueError,
        match="Site 6 merged width must equal hidden_size",
    ):
        _verify_grouped(bundle)


def test_verify_grouped_calibration_rejects_wrong_site3_materialized_head_count(
    grouped_calibration_bundle,
):
    """
    The verifier must validate materialized states as well as statistics.
    """
    bundle = grouped_calibration_bundle
    path = _state_path(bundle, 3, "phase")
    state = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    state["num_heads"] = NUM_KEY_VALUE_HEADS
    torch.save(state, path)

    with pytest.raises(
        ValueError,
        match="Site 3 must use post-repeat attention-head state",
    ):
        _verify_grouped(bundle)


def test_verify_grouped_calibration_rejects_per_head_site6_materialized_state(
    grouped_calibration_bundle,
):
    bundle = grouped_calibration_bundle
    path = _state_path(bundle, 6, "mtn")
    state = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    state["parameter_layout"] = "attention_head_grouped"
    state["num_heads"] = NUM_ATTENTION_HEADS
    state["channels_per_head"] = HEAD_DIM
    state["group_size"] = HEAD_DIM
    state["groups_per_head"] = 1
    torch.save(state, path)

    with pytest.raises(
        ValueError,
        match="Site 6 must use merged last-dim state",
    ):
        _verify_grouped(bundle)


def test_verify_grouped_calibration_rejects_stale_manifest_gif_topology(
    grouped_calibration_bundle,
):
    bundle = grouped_calibration_bundle
    manifest = copy.deepcopy(bundle.manifest)
    manifest["gif_salient_site_ids"] = [1, 3, 4, 7, 10]

    with pytest.raises(
        ValueError,
        match="gif_salient_site_ids",
    ):
        _verify_grouped(bundle, manifest)


def _gif_eval_fixture(tmp_path):
    cfg = {
        "experiment": {
            "ann_mode": "gif_aware",
        },
        "replacement": {
            "common_clip_enabled": False,
        },
        "calibration": {
            "group_size": -1,
        },
    }
    layout = SimpleNamespace(
        ann_training_site_dir=tmp_path / "ann_training_sites",
        conversion_site_dir=tmp_path / "conversion_sites",
    )
    controller = SiteController(
        mode="gif",
        site_root=layout.ann_training_site_dir,
        common_clip_enabled=False,
    )

    metadata = {
        **evaluation_forward_metadata(
            cfg,
            layout,
            neuron="ann",
            controller=controller,
        ),
        **evaluation_calibration_metadata(
            cfg,
            layout,
            neuron="ann",
        ),
    }
    path = tmp_path / "metrics.json"
    write_json(
        path,
        {
            "snn2_metadata": metadata,
        },
    )
    return cfg, layout, path, metadata


def test_verify_final_ann_forward_metadata_accepts_current_gif_provenance(
    tmp_path,
):
    cfg, layout, path, metadata = _gif_eval_fixture(tmp_path)

    assert metadata["gif_salient_site_ids"] == [1, 3, 4, 6, 7, 10]
    assert metadata["gif_all_low_site_ids"] == [2]
    assert metadata["gif_identity_site_ids"] == [5, 8, 9]
    assert metadata["gif_multi_mask_roles"] == {
        "1": ["q", "k", "v"],
        "7": ["gate", "up"],
    }
    assert (
        metadata["gif_saliency_selection_policy"]
        == "spikellm_global_per_channel_threshold_leq"
    )
    assert (
        metadata["gif_saliency_tie_policy"]
        == "mask_low_equals_score_le_threshold"
    )
    assert metadata["gif_linear_saliency_dtype"] == "float32"
    assert metadata["gif_matmul_saliency_dtype"] == "float64"

    _VERIFY._verify_final_ann_forward_metadata(
        cfg,
        layout,
        path,
    )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        (
            "gif_salient_site_ids",
            [1, 3, 4, 7, 10],
        ),
        (
            "gif_all_low_site_ids",
            [],
        ),
        (
            "gif_identity_site_ids",
            [5, 8],
        ),
        (
            "gif_multi_mask_roles",
            {
                "1": ["q", "k"],
                "7": ["gate", "up"],
            },
        ),
        (
            "gif_saliency_selection_policy",
            "per_head_topk",
        ),
        (
            "gif_saliency_tie_policy",
            "exact_low_quota",
        ),
        (
            "gif_linear_saliency_dtype",
            "float64",
        ),
        (
            "gif_matmul_saliency_dtype",
            "float32",
        ),
    ],
)
def test_verify_final_ann_forward_metadata_rejects_stale_gif_provenance(
    tmp_path,
    field,
    bad_value,
):
    cfg, layout, path, metadata = _gif_eval_fixture(tmp_path)
    metadata[field] = bad_value
    write_json(
        path,
        {
            "snn2_metadata": metadata,
        },
    )

    with pytest.raises(
        ValueError,
        match="stale/incompatible GIF provenance",
    ):
        _VERIFY._verify_final_ann_forward_metadata(
            cfg,
            layout,
            path,
        )


def test_verify_final_ann_forward_metadata_rejects_legacy_gif_impl_string(
    tmp_path,
):
    cfg, layout, path, metadata = _gif_eval_fixture(tmp_path)
    metadata["static_replacement_impl"] = (
        "StaticGIF/SoftmaxIdentityGIF.forward"
    )
    write_json(
        path,
        {
            "snn2_metadata": metadata,
        },
    )

    with pytest.raises(
        ValueError,
        match="static_replacement_impl",
    ):
        _VERIFY._verify_final_ann_forward_metadata(
            cfg,
            layout,
            path,
        )
