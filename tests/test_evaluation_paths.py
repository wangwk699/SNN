import random
from types import SimpleNamespace

from snn2.data import load_selected_raw, prepare_manifests
import pytest

from snn2.artifacts import prefix_enabled_dirname
from snn2.evaluation import (
    activation_neuron_operators_per_temporal_forward,
    build_evaluation_controller,
    evaluation_calibration_metadata,
    evaluation_ann_common_clip_enabled,
    evaluation_forward_metadata,
    final_ann_replacement_mode,
    resolve_tldr_evaluation_layout,
)


@pytest.mark.parametrize(
    ("neuron", "expected"),
    [("phase", 281), ("gif", 280), ("mtn", 280), ("ann", 0)],
)
def test_activation_neuron_operator_count_includes_global_phase(neuron, expected):
    assert activation_neuron_operators_per_temporal_forward(
        num_hidden_layers=28, neuron=neuron
    ) == expected


def test_activation_neuron_operator_count_rejects_unknown_neuron():
    with pytest.raises(ValueError, match="Unknown neuron"):
        activation_neuron_operators_per_temporal_forward(
            num_hidden_layers=28, neuron="unknown"
        )


@pytest.mark.parametrize("mode", ["vanilla", "unaware"])
def test_identity_final_ann_evaluation_has_no_calibration_metadata(mode):
    cfg = {"experiment": {"ann_mode": mode}, "calibration": {"group_size": -1}}
    layout = SimpleNamespace(conversion_site_dir="conversion", ann_training_site_dir="training")
    assert evaluation_calibration_metadata(cfg, layout, neuron="ann") == {
        "calibration_source_stage": None,
        "reused_ann_training_artifacts": False,
        "post_finetuning_recalibration": False,
        "calibration_root": None,
        "calibration_group_size": -1,
        "calibration_grouping_policy": "site234_logical_per_head_site6_merged_last_dim_v2",
    }


@pytest.mark.parametrize("mode", ["phase_aware", "gif_aware"])
def test_aware_final_ann_evaluation_uses_training_calibration_metadata(mode):
    cfg = {"experiment": {"ann_mode": mode}, "calibration": {"group_size": -1}}
    layout = SimpleNamespace(conversion_site_dir="conversion", ann_training_site_dir="training")
    assert evaluation_calibration_metadata(cfg, layout, neuron="ann") == {
        "calibration_source_stage": "ann_training",
        "reused_ann_training_artifacts": True,
        "post_finetuning_recalibration": False,
        "calibration_root": "training",
        "calibration_group_size": -1,
        "calibration_grouping_policy": "site234_logical_per_head_site6_merged_last_dim_v2",
    }


@pytest.mark.parametrize("mode", ["phase_aware", "gif_aware"])
@pytest.mark.parametrize("enabled", [True, False])
def test_evaluation_records_ann_training_common_clip(mode, enabled):
    cfg = {
        "experiment": {"ann_mode": mode},
        "replacement": {"common_clip_enabled": enabled},
    }
    assert evaluation_ann_common_clip_enabled(cfg) is enabled
    assert evaluation_ann_common_clip_enabled(cfg, base=True) is False
    assert evaluation_ann_common_clip_enabled(
        cfg, rotated_pre_finetuning=True
    ) is False


@pytest.mark.parametrize(
    ("configured", "selected", "is_full", "dirname"),
    [
        (None, 6553, True, "test_samples_6553_full"),
        (128, 128, False, "test_samples_128"),
        (9999, 6553, True, "test_samples_6553_full"),
    ],
)
def test_resolve_tldr_evaluation_layout(configured, selected, is_full, dirname):
    layout = resolve_tldr_evaluation_layout(6553, configured)
    assert layout == {
        "selected_test_samples": selected,
        "is_full_test": is_full,
        "dirname": dirname,
    }


@pytest.mark.parametrize("configured", [0, -1])
def test_resolve_tldr_evaluation_layout_rejects_non_positive_counts(configured):
    with pytest.raises(ValueError):
        resolve_tldr_evaluation_layout(6553, configured)

@pytest.mark.parametrize(
    ("configured", "enabled", "path"),
    [
        (128, True, "test_samples_128/prefix_enabled_ture"),
        (128, False, "test_samples_128/prefix_enabled_false"),
        (None, True, "test_samples_6553_full/prefix_enabled_ture"),
        (None, False, "test_samples_6553_full/prefix_enabled_false"),
    ],
)
def test_rotated_pre_finetuning_prefix_result_path(configured, enabled, path):
    layout = resolve_tldr_evaluation_layout(6553, configured)
    assert f"{layout['dirname']}/{prefix_enabled_dirname(enabled)}" == path



class _Dataset(list):
    column_names = ()

    def select(self, indices):
        return _Dataset(self[index] for index in indices)


def test_tulu_full_training_uses_all_rows_except_fixed_validation(monkeypatch, tmp_path):
    raw = {"train": _Dataset({"value": index} for index in range(20))}
    monkeypatch.setattr("snn2.data._load_raw", lambda cfg: raw)
    cfg = {
        "experiment": {"task": "tulu3", "seed": 7},
        "data": {
            "dataset_name": "fake/tulu",
            "train_split": "train",
            "train_size": None,
            "validation_size": 5,
        },
        "training": {},
        "calibration": {"seed": 42, "num_samples": 4, "with_replacement": False},
    }
    manifests = prepare_manifests(cfg, SimpleNamespace(data_dir=tmp_path))
    assert len(manifests["train"]["indices"]) == 15
    assert len(manifests["validation"]["indices"]) == 5
    assert set(manifests["train"]["indices"]).isdisjoint(
        manifests["validation"]["indices"]
    )
    assert set(manifests["train"]["indices"]) | set(
        manifests["validation"]["indices"]
    ) == set(range(20))


def test_tldr_train_subset_is_fixed_random_without_replacement(monkeypatch, tmp_path):
    raw = {
        "train": _Dataset({} for _ in range(20)),
        "validation": _Dataset({} for _ in range(5)),
        "test": _Dataset({} for _ in range(6)),
    }
    monkeypatch.setattr("snn2.data._load_raw", lambda cfg: raw)
    cfg = {
        "experiment": {"task": "tldr", "seed": 7},
        "data": {
            "dataset_name": "fake/tldr",
            "train_split": "train",
            "validation_split": "validation",
            "evaluation_split": "test",
        },
        "training": {"tldr_train_samples": 8, "tldr_train_seed": 42},
        "calibration": {"seed": 42, "num_samples": 4, "with_replacement": False},
    }
    manifests = prepare_manifests(cfg, SimpleNamespace(data_dir=tmp_path))
    expected = random.Random(42).sample(range(20), k=8)
    expected.sort()

    assert manifests["train"]["indices"] == expected
    assert len(set(manifests["train"]["indices"])) == 8
    assert manifests["train"]["sampling"] == "seeded_random_without_replacement"
    assert manifests["train"]["tldr_train_seed"] == 42
    assert set(manifests["calibration"]["indices"]) <= set(expected)


def test_ann_training_subset_uses_current_config_even_when_shared_manifest_is_full(
    monkeypatch, tmp_path
):
    raw = {
        "train": _Dataset({"value": index} for index in range(20)),
        "validation": _Dataset({"value": index} for index in range(5)),
        "test": _Dataset({"value": index} for index in range(6)),
    }
    monkeypatch.setattr("snn2.data._load_raw", lambda cfg: raw)
    cfg = {
        "experiment": {"task": "tldr", "seed": 7},
        "data": {
            "dataset_name": "fake/tldr",
            "train_split": "train",
            "validation_split": "validation",
            "evaluation_split": "test",
        },
        "training": {"tldr_train_samples": None, "tldr_train_seed": 42},
        "calibration": {"seed": 42, "num_samples": 4, "with_replacement": False},
    }
    layout = SimpleNamespace(data_dir=tmp_path)
    prepare_manifests(cfg, layout)
    assert len(load_selected_raw(cfg, layout).train) == 20

    cfg["training"]["tldr_train_samples"] = 8
    bundle = load_selected_raw(cfg, layout, use_configured_train_subset=True)
    expected = random.Random(42).sample(range(20), k=8)
    expected.sort()

    assert len(bundle.train) == 8
    assert [row["value"] for row in bundle.train] == expected
    assert bundle.manifests["train"]["indices"] == expected
    assert bundle.manifests["train"]["selection_scope"] == "current_ann_training_config"
    assert len(bundle.calibration) == 4


def test_ann_training_subset_rejects_more_rows_than_raw_split(monkeypatch, tmp_path):
    raw = {
        "train": _Dataset({"value": index} for index in range(5)),
        "validation": _Dataset({"value": index} for index in range(2)),
        "test": _Dataset({"value": index} for index in range(2)),
    }
    monkeypatch.setattr("snn2.data._load_raw", lambda cfg: raw)
    cfg = {
        "experiment": {"task": "tldr", "seed": 7},
        "data": {
            "dataset_name": "fake/tldr",
            "train_split": "train",
            "validation_split": "validation",
            "evaluation_split": "test",
        },
        "training": {"tldr_train_samples": None, "tldr_train_seed": 42},
        "calibration": {"seed": 42, "num_samples": 2, "with_replacement": False},
    }
    layout = SimpleNamespace(data_dir=tmp_path)
    prepare_manifests(cfg, layout)
    cfg["training"]["tldr_train_samples"] = 6

    with pytest.raises(ValueError, match="contains only 5 rows"):
        load_selected_raw(cfg, layout, use_configured_train_subset=True)


@pytest.mark.parametrize(
    ("ann_mode", "expected"),
    [("vanilla", "identity"), ("unaware", "identity"),
     ("phase_aware", "phase"), ("gif_aware", "gif")],
)
def test_final_ann_replacement_mode_mapping(ann_mode, expected):
    assert final_ann_replacement_mode({"experiment": {"ann_mode": ann_mode}}) == expected


def _evaluation_cfg(ann_mode, clip=False):
    return {
        "experiment": {"ann_mode": ann_mode},
        "phase": {"surrogate_slope": 2.0},
        "replacement": {"common_clip_enabled": clip},
        "calibration": {"group_size": -1},
    }


@pytest.mark.parametrize(
    ("ann_mode", "clip", "expected_mode"),
    [("vanilla", False, "identity"), ("unaware", False, "identity"),
     ("phase_aware", False, "phase"), ("phase_aware", True, "phase"),
     ("gif_aware", False, "gif"), ("gif_aware", True, "gif")],
)
def test_build_final_ann_controller(monkeypatch, ann_mode, clip, expected_mode):
    monkeypatch.setattr("snn2.evaluation.validate_site_state_bundle", lambda *_a, **_k: {"manifest": {"calibration_group_size": -1, "calibration_grouping_policy": "site234_logical_per_head_site6_merged_last_dim_v2"}})
    layout = SimpleNamespace(ann_training_site_dir="training", conversion_site_dir="conversion")
    controller, steps = build_evaluation_controller(
        _evaluation_cfg(ann_mode, clip), layout, neuron="ann"
    )
    assert controller.mode == expected_mode
    assert steps == 1
    assert controller.common_clip_enabled is (clip if ann_mode.endswith("aware") and ann_mode != "unaware" else False)
    assert controller.phase_surrogate_slope == (
        2.0 if ann_mode == "phase_aware" else None
    )
    if ann_mode in {"phase_aware", "gif_aware"}:
        assert str(controller.site_root) == "training"
    else:
        assert controller.site_root is None


@pytest.mark.parametrize("flag", ["base", "rotated_pre_finetuning"])
def test_diagnostic_ann_controller_remains_identity(flag):
    kwargs = {flag: True}
    controller, steps = build_evaluation_controller(
        _evaluation_cfg("phase_aware", True), SimpleNamespace(), neuron="ann", **kwargs
    )
    assert controller.mode == "identity"
    assert controller.common_clip_enabled is False
    assert steps == 1


@pytest.mark.parametrize("ann_mode", ["vanilla", "unaware", "phase_aware", "gif_aware"])
@pytest.mark.parametrize("neuron", ["phase", "gif", "mtn"])
def test_all_ann_modes_use_temporal_snn_controller(monkeypatch, ann_mode, neuron):
    monkeypatch.setattr("snn2.controller.validate_site_state_bundle", lambda *_a, **_k: {"temporal_steps": {"phase": 4, "gif": 2, "mtn": 4}})
    layout = SimpleNamespace(conversion_site_dir="conversion")
    controller, _ = build_evaluation_controller(
        _evaluation_cfg(ann_mode, True), layout, neuron=neuron
    )
    assert controller.mode == f"deploy_{neuron}"
    assert controller.common_clip_enabled is False


@pytest.mark.parametrize(
    ("ann_mode", "neuron", "kind"),
    [("vanilla", "ann", "identity_ann"), ("unaware", "ann", "identity_ann"),
     ("phase_aware", "ann", "phase_surrogate_ann"),
     ("gif_aware", "ann", "gif_surrogate_ann"),
     ("phase_aware", "phase", "temporal_phase_snn"),
     ("phase_aware", "gif", "temporal_gif_snn"),
     ("phase_aware", "mtn", "temporal_mtn_snn")],
)
def test_evaluation_forward_metadata(monkeypatch, ann_mode, neuron, kind):
    monkeypatch.setattr("snn2.evaluation.validate_site_state_bundle", lambda *_a, **_k: {"manifest": {"calibration_group_size": -1, "calibration_grouping_policy": "site234_logical_per_head_site6_merged_last_dim_v2"}})
    monkeypatch.setattr("snn2.controller.validate_site_state_bundle", lambda *_a, **_k: {"temporal_steps": {"phase": 4, "gif": 2, "mtn": 4}})
    cfg = _evaluation_cfg(ann_mode, True)
    layout = SimpleNamespace(ann_training_site_dir="training", conversion_site_dir="conversion")
    controller, _ = build_evaluation_controller(cfg, layout, neuron=neuron)
    metadata = evaluation_forward_metadata(
        cfg, layout, neuron=neuron, controller=controller
    )
    assert metadata["evaluation_forward_kind"] == kind
    assert metadata["temporal_execution"] is (neuron != "ann")
    assert metadata["evaluation_common_clip_applied"] is (
        neuron == "ann" and ann_mode in {"phase_aware", "gif_aware"}
    )
    assert metadata["calibration_group_size"] == -1
    assert metadata["gif_salient_site_ids"] == [1, 3, 4, 6, 7, 10]
    assert metadata["gif_all_low_site_ids"] == [2]
    assert metadata["gif_identity_site_ids"] == [5, 8, 9]
    assert metadata["gif_multi_mask_roles"] == {
        "1": ["q", "k", "v"], "7": ["gate", "up"]
    }
    assert metadata["gif_saliency_selection_policy"] == "spikellm_global_per_channel_threshold_leq"
    assert metadata["gif_saliency_tie_policy"] == "mask_low_equals_score_le_threshold"
    assert metadata["gif_linear_saliency_dtype"] == "float32"
    assert metadata["gif_matmul_saliency_dtype"] == "float64"
    if neuron == "ann" and ann_mode == "gif_aware":
        assert metadata["static_replacement_impl"] == (
            "StaticGIF/AllLowStaticGIF/IdentityGIF/SoftmaxIdentityGIF.forward"
        )
    assert metadata["calibration_grouping_policy"] == "site234_logical_per_head_site6_merged_last_dim_v2"
    assert metadata["softmax_site5_clip_applied"] is False
