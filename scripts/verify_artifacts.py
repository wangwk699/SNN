from __future__ import annotations
import math
from pathlib import Path
import torch
from _common import apply_deployment_overrides, parser, setup

from snn2.artifacts import prefix_enabled_dirname, read_json, sha256_file, write_json
from snn2.data import validate_prefix_discovery_state
from snn2.config import (
    conversion_prefix_enabled,
    conversion_reuses_ann_training_artifacts,
    conversion_calibration_stage,
    conversion_prefix_artifact_stage,
    use_post_finetuning_artifacts,
    is_aware_ann_mode,
    final_ann_evaluation_prefix_enabled,
    final_ann_evaluation_prefix_artifact_stage,
    evaluation_prefix_enabled,
    requires_ann_training_calibration,
    requires_pre_finetuning_prefix,
    training_common_clip_enabled,
    training_prefix_enabled,
)
from snn2.conversion import (
    validate_calibration,
    validate_conversion_metadata,
    validate_conversion_prefix,
)
from snn2.sites import (
    CLIP_ELIGIBLE_SITE_IDS,
    GIF_ALL_LOW_SITE_IDS,
    GIF_IDENTITY_SITE_IDS,
    GIF_MULTI_MASK_ROLES,
    GIF_SALIENT_SITE_IDS,
    SOFTMAX_SITE_ID,
    topology_metadata,
)
from snn2.evaluation import append_evaluation_num_samples_if_needed, final_ann_replacement_mode, resolve_tldr_evaluation_layout
from snn2.logging_utils import StageRun
from snn2.state_validation import validate_clip_profile, validate_site_state_bundle
from snn2.training import validate_recorded_training_artifact_provenance
from snn2.temporal_ops import (
    CALIBRATION_GROUPING_POLICY,
    GIF_LINEAR_SALIENCY_DTYPE,
    GIF_MATMUL_SALIENCY_DTYPE,
    GIF_SALIENCY_SELECTION_POLICY,
    GIF_SALIENCY_TIE_POLICY,
    SOFTMAX_SITE5_CLIP_POLICY,
    SOFTMAX_SITE5_GIF_POLICY,
    STATISTICS_FORMAT_VERSION,
    validate_temporal_policy,
)


def _evaluation_metadata(path):
    payload = read_json(path)
    return payload.get("snn2_metadata", payload)

def _validate_snn_forward_metadata(policy_source, *, neuron, metrics_path):
    expected_forward = {
        "evaluation_forward_kind": f"temporal_{neuron}_snn",
        "controller_mode": f"deploy_{neuron}",
        "temporal_execution": True,
        "evaluation_common_clip_applied": False,
        "global_final_norm_replacement": {
            "phase": "temporal_phase", "mtn": "temporal_mtn", "gif": "identity"
        }[neuron],
        "global_final_norm_clip_applied": False,
    }
    missing = [key for key in expected_forward if key not in policy_source]
    mismatched = {
        key: {"expected": expected, "actual": policy_source.get(key)}
        for key, expected in expected_forward.items()
        if key in policy_source and policy_source[key] != expected
    }
    if missing or mismatched:
        raise ValueError(
            "SNN metrics have incompatible temporal forward metadata: "
            f"{metrics_path}: missing={missing}, mismatched={mismatched}"
        )


def _validate_snn_source_metadata(cfg, descriptor, metrics, *, metrics_path):
    expected = {
        "use_post_finetuning_artifacts": use_post_finetuning_artifacts(cfg),
        "calibration_source_stage": conversion_calibration_stage(cfg),
        "prefix_source_stage": conversion_prefix_artifact_stage(cfg),
        "reused_ann_training_artifacts": conversion_reuses_ann_training_artifacts(cfg),
        "post_finetuning_recalibration": not conversion_reuses_ann_training_artifacts(cfg),
    }
    for key, value in expected.items():
        if descriptor.get(key) != value or metrics.get(key) != value:
            raise ValueError(
                "SNN descriptor and metrics source mismatch for " + key + ": " + str(metrics_path)
            )


def _final_ann_prefix_root(cfg, layout):
    if not final_ann_evaluation_prefix_enabled(cfg):
        return None
    stage = final_ann_evaluation_prefix_artifact_stage(cfg)
    return (
        layout.ann_training_prefix_dir
        if stage == "pre_finetuning"
        else layout.post_finetuning_prefix_dir
    )


def _validate_prefix_artifact(cfg, layout, root, *, label):
    state_path = root / "prefix_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"{label} Prefix is missing: {state_path}")
    return validate_prefix_discovery_state(cfg, layout, root)


def _selected_snn_prefix_summary(prefix_info):
    token_ids = [int(value) for value in prefix_info.get("token_ids", [])]
    return {
        "prefix_length": len(token_ids),
        "prefix_kv_required": bool(token_ids),
    }


def _validate_aware_final_ann_training_provenance(cfg, layout):
    if is_aware_ann_mode(cfg):
        return validate_recorded_training_artifact_provenance(cfg, layout)
    return None


def _verify_final_ann_forward_metadata(cfg, layout, path):
    metadata = _evaluation_metadata(path)
    mode = final_ann_replacement_mode(cfg)
    expected = {
        "identity": ("identity_ann", False, None),
        "phase": ("phase_surrogate_ann", True, "PhaseSurrogate.forward"),
        "gif": ("gif_surrogate_ann", True, "StaticGIF/AllLowStaticGIF/IdentityGIF/SoftmaxIdentityGIF.forward"),
    }[mode]
    required = {
        "evaluation_forward_kind": expected[0],
        "controller_mode": mode,
        "temporal_execution": False,
        "static_replacement_enabled": expected[1],
        "static_replacement_impl": expected[2],
        "evaluation_common_clip_applied": (
            training_common_clip_enabled(cfg) if expected[1] else False
        ),
        "calibration_group_size": int(cfg["calibration"]["group_size"]),
        "calibration_grouping_policy": CALIBRATION_GROUPING_POLICY,
        "softmax_site5_clip_applied": False,
        "global_final_norm_replacement": "phase_surrogate" if mode == "phase" else "identity",
        "global_final_norm_clip_applied": False,
    }
    for key, value in required.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Final ANN evaluation has stale/incompatible {key}: {path}. "
                "Re-run final ANN evaluation."
            )
    gif_provenance = {
        "gif_salient_site_ids": sorted(GIF_SALIENT_SITE_IDS),
        "gif_all_low_site_ids": sorted(GIF_ALL_LOW_SITE_IDS),
        "gif_identity_site_ids": sorted(GIF_IDENTITY_SITE_IDS),
        "gif_multi_mask_roles": {
            str(site): list(roles)
            for site, roles in sorted(GIF_MULTI_MASK_ROLES.items())
        },
        "gif_saliency_selection_policy": GIF_SALIENCY_SELECTION_POLICY,
        "gif_saliency_tie_policy": GIF_SALIENCY_TIE_POLICY,
        "gif_linear_saliency_dtype": GIF_LINEAR_SALIENCY_DTYPE,
        "gif_matmul_saliency_dtype": GIF_MATMUL_SALIENCY_DTYPE,
    }
    for key, value in gif_provenance.items():
        if metadata.get(key) != value:
            raise ValueError(
                "Final ANN evaluation has stale/incompatible GIF provenance "
                f"{key}: {path}. Re-run final ANN evaluation."
            )
    expected_root = layout.ann_training_site_dir if expected[1] else None
    actual_root = metadata.get("replacement_state_root")
    if expected_root is None:
        if actual_root is not None:
            raise ValueError(f"Identity ANN evaluation unexpectedly uses replacement states: {path}")
    elif actual_root is None or Path(actual_root).resolve() != Path(expected_root).resolve():
        raise ValueError(f"Final aware ANN evaluation uses the wrong state root: {path}")
    expected_stage = "ann_training" if expected[1] else None
    if metadata.get("calibration_source_stage") != expected_stage:
        raise ValueError(f"Final ANN evaluation has incompatible calibration provenance: {path}")

    prefix_enabled = final_ann_evaluation_prefix_enabled(cfg)
    if metadata.get("prefix_enabled") != prefix_enabled:
        raise ValueError(f"Final ANN evaluation has incompatible prefix_enabled: {path}")
    if metadata.get("prefix_stage") != "final_ann_evaluation":
        raise ValueError(f"Final ANN evaluation has incompatible prefix_stage: {path}")
    prefix_root = _final_ann_prefix_root(cfg, layout)
    if not prefix_enabled:
        if metadata.get("prefix_root") is not None:
            raise ValueError(f"Final ANN evaluation unexpectedly uses a Prefix root: {path}")
    else:
        expected_prefix_stage = final_ann_evaluation_prefix_artifact_stage(cfg)
        actual_prefix_root = metadata.get("prefix_root")
        if metadata.get("prefix_source_stage") != expected_prefix_stage:
            raise ValueError(f"Final ANN evaluation has incompatible prefix_source_stage: {path}")
        if actual_prefix_root is None or Path(actual_prefix_root).resolve() != prefix_root.resolve():
            raise ValueError(f"Final ANN evaluation has incompatible prefix_root: {path}")


def _require_manifest_flags(manifest, expected, label):
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"{label} manifest has invalid {key}: {manifest.get(key)!r}")


def _verify_grouped_calibration(cfg, layout, manifest, calibration):
    expected = {
        "calibration_group_size": int(cfg["calibration"]["group_size"]),
        "calibration_grouping_policy": CALIBRATION_GROUPING_POLICY,
        "statistics_format_version": STATISTICS_FORMAT_VERSION,
        "softmax_site5_gif_policy": SOFTMAX_SITE5_GIF_POLICY,
        "softmax_site5_clip_policy": SOFTMAX_SITE5_CLIP_POLICY,
        "clip_eligible_site_ids": sorted(CLIP_ELIGIBLE_SITE_IDS),
        "clip_excluded_site_ids": [SOFTMAX_SITE_ID],
        "gif_salient_site_ids": sorted(GIF_SALIENT_SITE_IDS),
        "gif_all_low_site_ids": sorted(GIF_ALL_LOW_SITE_IDS),
        "gif_identity_site_ids": sorted(GIF_IDENTITY_SITE_IDS),
        "gif_multi_mask_roles": {
            str(site): list(roles)
            for site, roles in sorted(GIF_MULTI_MASK_ROLES.items())
        },
    }
    _require_manifest_flags(manifest, expected, "Grouped calibration")
    ann_config = read_json(layout.ann_checkpoint_dir / "config.json")
    query_heads = int(ann_config["num_attention_heads"])
    hidden_size = int(ann_config["hidden_size"])
    head_dim = int(ann_config.get("head_dim", hidden_size // query_heads))
    if query_heads <= 0 or head_dim <= 0 or query_heads * head_dim != hidden_size:
        raise ValueError("ANN config attention-head geometry is inconsistent")
    clip_count = 0
    for statistics_path in sorted(layout.conversion_site_dir.glob("layer_*/site_*/statistics.pt")):
        statistics = torch.load(statistics_path, map_location="cpu", weights_only=False)
        if statistics.get("format_version") != STATISTICS_FORMAT_VERSION:
            raise ValueError(f"Legacy statistics state: {statistics_path}")
        site = int(statistics["site_index"])
        if site in {2, 3, 4}:
            if statistics.get("layout_kind") != "attention_head":
                raise ValueError(
                    f"Site {site} must use attention_head statistics: {statistics_path}"
                )
            if type(statistics.get("num_heads")) is not int:
                raise ValueError(
                    f"Site {site} must record integer num_heads: {statistics_path}"
                )
            if statistics["num_heads"] != query_heads:
                raise ValueError(
                    f"Site {site} must use repeated/query attention heads "
                    f"({query_heads}): {statistics_path}"
                )
            channels_per_head = statistics.get("channels_per_head")
            if type(channels_per_head) is not int or channels_per_head != head_dim:
                raise ValueError(
                    f"Site {site} channels_per_head must equal head_dim={head_dim}: "
                    f"{statistics_path}"
                )
            if statistics.get("channels") != query_heads * head_dim:
                raise ValueError(
                    f"Site {site} channel count must equal num_attention_heads * "
                    f"head_dim: {statistics_path}"
                )
        elif site == 6:
            if statistics.get("layout_kind") != "last_dim":
                raise ValueError(
                    f"Site 6 must use merged last_dim statistics: {statistics_path}"
                )
            if statistics.get("num_heads") is not None:
                raise ValueError(
                    f"Site 6 must not preserve a per-head layout: {statistics_path}"
                )
            if statistics.get("channels_per_head") is not None:
                raise ValueError(
                    f"Site 6 must not save channels_per_head: {statistics_path}"
                )
            if statistics.get("channels") != hidden_size:
                raise ValueError(
                    f"Site 6 merged width must equal hidden_size={hidden_size}: "
                    f"{statistics_path}"
                )
        directory = statistics_path.parent
        clip_present = (directory / "clip_state.pt").exists()
        clip_count += int(clip_present)
        if site in {2, 3, 4, 6}:
            state_names = ["phase_state.pt", "gif_state.pt", "mtn_state.pt"]
            if clip_present:
                state_names.append("clip_state.pt")
            for state_name in state_names:
                state_path = directory / state_name
                state = torch.load(state_path, map_location="cpu", weights_only=False)
                if site in {2, 3, 4}:
                    if (
                        state.get("parameter_layout") != "attention_head_grouped"
                        or state.get("num_heads") != query_heads
                        or state.get("channels_per_head") != head_dim
                    ):
                        raise ValueError(
                            f"Site {site} must use post-repeat attention-head state: "
                            f"{state_path}"
                        )
                    logical_width = head_dim
                else:
                    if (
                        state.get("parameter_layout") != "last_dim_grouped"
                        or state.get("num_heads") is not None
                        or state.get("channels_per_head") != hidden_size
                    ):
                        raise ValueError(
                            f"Site 6 must use merged last-dim state: {state_path}"
                        )
                    logical_width = hidden_size
                group_size = state.get("group_size")
                groups = state.get("groups_per_head")
                if (
                    type(group_size) is not int
                    or group_size <= 0
                    or logical_width % group_size != 0
                    or groups != logical_width // group_size
                ):
                    raise ValueError(
                        f"Site {site} has invalid grouped state width: {state_path}"
                    )
        if site == SOFTMAX_SITE_ID:
            gif = torch.load(directory / "gif_state.pt", map_location="cpu", weights_only=False)
            if gif.get("gif_policy") != SOFTMAX_SITE5_GIF_POLICY or clip_present:
                raise ValueError(
                    f"Site 5 must use SpikeLLM identity GIF and no Clip: {directory}"
                )
    expected_clips = 0
    if clip_count != expected_clips:
        raise ValueError(f"Calibration Clip count mismatch: {clip_count} != {expected_clips}")


def _verify_hashes(manifest, label):
    for path_key, hash_key in (
        ("calibration_data_manifest_path", "calibration_data_manifest_sha256"),
        ("prefix_state_path", "prefix_state_sha256"),
        ("prefix_kv_path", "prefix_kv_sha256"),
        ("rotation_state_path", "rotation_state_sha256"),
        ("source_ann_checkpoint", None),
    ):
        path = manifest.get(path_key)
        expected = manifest.get(hash_key) if hash_key else None
        if path is None:
            if expected is not None:
                raise ValueError(f"{label} has {hash_key} without {path_key}")
            continue
        if hash_key and (not expected or sha256_file(path) != expected):
            raise ValueError(f"{label} provenance hash mismatch: {hash_key}")
    checkpoint = manifest.get("source_ann_checkpoint")
    expected_config = manifest.get("source_ann_config_sha256")
    if checkpoint is not None and (not expected_config or sha256_file(Path(checkpoint) / "config.json") != expected_config):
        raise ValueError(f"{label} provenance hash mismatch: source_ann_config_sha256")


def _verify_scalar_distribution(
    distribution,
    label,
    *,
    expected_count=None,
    include_max=True,
):
    if not isinstance(distribution, dict):
        raise ValueError(f"{label} must be a mapping")
    fields = ["mean", "p50", "p90", "p99"]
    if include_max:
        fields.append("max")
    if expected_count is not None:
        try:
            count = int(distribution.get("count", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} count is invalid") from exc
        if count != expected_count:
            raise ValueError(f"{label} count is inconsistent")
        if expected_count == 0:
            if any(distribution.get(field) is not None for field in fields):
                raise ValueError(f"{label} must use null statistics when empty")
            return
    if any(field not in distribution for field in fields):
        raise ValueError(f"{label} is missing scalar statistics")
    values = {field: float(distribution[field]) for field in fields}
    if not all(math.isfinite(value) and value >= 0.0 for value in values.values()):
        raise ValueError(f"{label} must contain finite non-negative statistics")
    if not values["p50"] <= values["p90"] <= values["p99"]:
        raise ValueError(f"{label} quantiles are not monotonic")
    if include_max and values["p99"] > values["max"] + 1e-12:
        raise ValueError(f"{label} P99 exceeds maximum")


def _verify_rotation_regression_metrics(regression):
    required = {
        "num_tokens_compared",
        "relative_l2_error",
        "max_abs_error",
        "mean_abs_error",
        "p99_abs_error",
        "p999_abs_error",
        "top1_agreement",
        "top1_agreement_count",
        "top1_disagreement_count",
        "absolute_error_percentile_estimator",
        "margin_aware_diagnostic",
    }
    missing = sorted(required - regression.keys())
    if missing:
        raise ValueError(f"Rotation regression is missing required metrics: {missing}")

    num_tokens = int(regression["num_tokens_compared"])
    agreement_count = int(regression["top1_agreement_count"])
    disagreement_count = int(regression["top1_disagreement_count"])
    agreement = float(regression["top1_agreement"])
    if num_tokens <= 0:
        raise ValueError("Rotation regression must compare at least one valid token")
    if not 0.0 <= agreement <= 1.0:
        raise ValueError("Rotation regression top1_agreement must be in [0, 1]")
    if agreement_count < 0 or disagreement_count < 0:
        raise ValueError("Rotation regression Top-1 counts must be non-negative")
    if agreement_count + disagreement_count != num_tokens:
        raise ValueError("Rotation regression Top-1 counts do not match compared tokens")
    expected_agreement = agreement_count / num_tokens
    if abs(expected_agreement - agreement) > 1e-12:
        raise ValueError("Rotation regression top1_agreement is inconsistent with its counts")

    nonnegative_metrics = {
        name: float(regression[name])
        for name in (
            "relative_l2_error",
            "mean_abs_error",
            "p99_abs_error",
            "p999_abs_error",
            "max_abs_error",
        )
    }
    if not all(math.isfinite(value) for value in nonnegative_metrics.values()):
        raise ValueError("Rotation regression error metrics must be finite")
    if any(value < 0.0 for value in nonnegative_metrics.values()):
        raise ValueError("Rotation regression error metrics must be non-negative")
    if nonnegative_metrics["p99_abs_error"] > nonnegative_metrics["p999_abs_error"]:
        raise ValueError("Rotation regression percentiles are not monotonic")

    estimator = regression["absolute_error_percentile_estimator"]
    if not isinstance(estimator, dict):
        raise ValueError("Rotation regression percentile estimator metadata must be a mapping")
    if estimator.get("method") != "streaming_linear_histogram":
        raise ValueError("Rotation regression uses an unknown percentile estimator")
    if int(estimator.get("num_bins", 0)) != 8192:
        raise ValueError("Rotation regression percentile estimator must use 8192 bins")
    if estimator.get("reported_value") != "bin_upper_edge":
        raise ValueError("Rotation regression percentiles must report bin upper edges")
    if estimator.get("exact") is not False:
        raise ValueError("Rotation regression histogram percentiles must be marked inexact")
    range_max = float(estimator.get("final_range_max", 0.0))
    if range_max <= 0.0:
        raise ValueError("Rotation regression histogram range must be positive")
    bin_width = range_max / int(estimator["num_bins"])
    if (
        nonnegative_metrics["p999_abs_error"]
        > nonnegative_metrics["max_abs_error"] + bin_width + 1e-12
    ):
        raise ValueError("Rotation regression P99.9 exceeds the observed maximum allowance")

    threshold = regression.get("threshold")
    if not isinstance(threshold, dict):
        raise ValueError("Rotation regression lacks hard thresholds")
    try:
        relative_l2_threshold = float(threshold["relative_l2_error"])
        top1_threshold = float(threshold["top1_agreement"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Rotation regression hard thresholds are incomplete") from exc
    if relative_l2_threshold <= 0.0 or not 0.0 <= top1_threshold < 1.0:
        raise ValueError("Rotation regression hard thresholds are invalid")
    expected_passed = (
        nonnegative_metrics["relative_l2_error"] <= relative_l2_threshold
        and agreement > top1_threshold
    )
    if bool(regression.get("passed")) != expected_passed:
        raise ValueError("Rotation regression passed flag contradicts its hard gates")

    margin = regression["margin_aware_diagnostic"]
    if not isinstance(margin, dict):
        raise ValueError("Rotation regression margin-aware diagnostic must be a mapping")
    if margin.get("definition") != "base_top1_margin_gt_2x_per_token_max_abs_error":
        raise ValueError("Rotation regression has an unknown margin-safe definition")
    required_margin_fields = (
        "margin_safe_token_count",
        "margin_unsafe_token_count",
        "margin_safe_agreement_count",
        "margin_safe_disagreement_count",
        "margin_unsafe_agreement_count",
        "margin_unsafe_disagreement_count",
        "margin_safe_fraction",
        "disagreement_margin_unsafe_fraction",
        "base_top1_margin_all_tokens",
        "per_token_max_abs_error_all_tokens",
        "base_top1_margin_disagreement_tokens",
        "per_token_max_abs_error_disagreement_tokens",
        "stability_ratio_disagreement_tokens",
    )
    if any(key not in margin for key in required_margin_fields):
        raise ValueError("Rotation regression margin-aware diagnostic is incomplete")
    required_margin_counts = required_margin_fields[:6]
    safe, unsafe, safe_agree, safe_disagree, unsafe_agree, unsafe_disagree = (
        int(margin[key]) for key in required_margin_counts
    )
    if any(value < 0 for value in (safe, unsafe, safe_agree, safe_disagree, unsafe_agree, unsafe_disagree)):
        raise ValueError("Rotation regression margin-aware counts must be non-negative")
    if safe + unsafe != num_tokens:
        raise ValueError("Rotation regression margin-safe partition does not match tokens")
    if safe_agree + safe_disagree != safe or unsafe_agree + unsafe_disagree != unsafe:
        raise ValueError("Rotation regression margin agreement partitions are inconsistent")
    if safe_agree + unsafe_agree != agreement_count:
        raise ValueError("Rotation regression margin agreement count mismatches Top-1")
    if safe_disagree + unsafe_disagree != disagreement_count:
        raise ValueError("Rotation regression margin disagreement count mismatches Top-1")
    if safe_disagree != 0:
        raise ValueError("Margin-safe token changed Top-1; regression metrics are misaligned")
    expected_safe_fraction = safe / num_tokens
    if abs(float(margin.get("margin_safe_fraction", float("nan"))) - expected_safe_fraction) > 1e-12:
        raise ValueError("Rotation regression margin_safe_fraction is inconsistent")
    expected_unsafe_disagreement_fraction = (
        1.0 if disagreement_count == 0 else unsafe_disagree / disagreement_count
    )
    if (
        abs(
            float(margin.get("disagreement_margin_unsafe_fraction", float("nan")))
            - expected_unsafe_disagreement_fraction
        )
        > 1e-12
    ):
        raise ValueError("Rotation regression disagreement margin-unsafe fraction is inconsistent")

    _verify_scalar_distribution(
        margin.get("base_top1_margin_all_tokens"),
        "All-token Base Top-1 margin distribution",
    )
    _verify_scalar_distribution(
        margin.get("per_token_max_abs_error_all_tokens"),
        "All-token per-token maximum error distribution",
    )
    delta_max = float(margin["per_token_max_abs_error_all_tokens"]["max"])
    if abs(delta_max - nonnegative_metrics["max_abs_error"]) > 1e-6:
        raise ValueError("Per-token maximum error distribution disagrees with root maximum")
    _verify_scalar_distribution(
        margin.get("base_top1_margin_disagreement_tokens"),
        "Disagreement Base Top-1 margin distribution",
        expected_count=disagreement_count,
    )
    _verify_scalar_distribution(
        margin.get("per_token_max_abs_error_disagreement_tokens"),
        "Disagreement per-token maximum error distribution",
        expected_count=disagreement_count,
    )
    ratio = margin.get("stability_ratio_disagreement_tokens")
    if not isinstance(ratio, dict) or ratio.get("definition") != "2_delta_over_base_top1_margin_plus_1e-12":
        raise ValueError("Rotation regression disagreement stability-ratio definition is invalid")
    _verify_scalar_distribution(
        ratio,
        "Disagreement stability-ratio distribution",
        expected_count=disagreement_count,
        include_max=False,
    )


def _verify_prompt_end_metrics(metrics, *, expected_prompts: int) -> None:
    required = {
        "num_prompts_compared",
        "relative_l2_error",
        "max_abs_error",
        "mean_abs_error",
        "top1_agreement",
        "top1_agreement_count",
        "top1_disagreement_count",
        "reference_top1_margin",
        "per_prompt_max_abs_error",
        "gating",
    }
    missing = sorted(required - metrics.keys())
    if missing:
        raise ValueError(f"Prompt-end diagnostic is missing required metrics: {missing}")
    if int(metrics["num_prompts_compared"]) != expected_prompts:
        raise ValueError("Prompt-end diagnostic must compare all 128 prompts")
    agreement = float(metrics["top1_agreement"])
    agreement_count = int(metrics["top1_agreement_count"])
    disagreement_count = int(metrics["top1_disagreement_count"])
    if not 0.0 <= agreement <= 1.0:
        raise ValueError("Prompt-end top1_agreement must be in [0, 1]")
    if agreement_count + disagreement_count != expected_prompts:
        raise ValueError("Prompt-end Top-1 counts do not match prompts")
    if abs(agreement - agreement_count / expected_prompts) > 1e-12:
        raise ValueError("Prompt-end Top-1 agreement is inconsistent with counts")
    for name in ("relative_l2_error", "max_abs_error", "mean_abs_error"):
        value = float(metrics[name])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"Prompt-end {name} must be finite and non-negative")
    if metrics["gating"] is not False:
        raise ValueError("Prompt-end diagnostic must not gate this protocol")
    _verify_scalar_distribution(
        metrics["reference_top1_margin"], "Prompt-end reference Top-1 margin"
    )
    _verify_scalar_distribution(
        metrics["per_prompt_max_abs_error"], "Prompt-end per-prompt maximum error"
    )


def _verify_rotation_regression_suite(regression) -> None:
    _require_manifest_flags(
        regression,
        {
            "format_version": 4,
            "purpose": "three_way_rotation_regression",
            "status": "passed",
            "num_samples": 128,
            "passed": True,
        },
        "Rotation regression",
    )
    implementation = regression.get("rotation_implementation")
    if not isinstance(implementation, dict):
        raise ValueError("Rotation regression lacks implementation metadata")
    if implementation.get("random_hadamard_orientation") != "DU":
        raise ValueError("Rotation regression must use random Hadamard orientation DU")
    if implementation.get("precision_policy") != "roste_aligned_v1":
        raise ValueError("Rotation regression must use precision policy roste_aligned_v1")
    hard_gate = regression.get("hard_gate")
    if not isinstance(hard_gate, dict) or hard_gate.get("scope") != "all_tokens_only":
        raise ValueError("Rotation regression hard gate must be all-token only")
    if hard_gate.get("prompt_end_is_gating") is not False:
        raise ValueError("Prompt-end diagnostic must not be a hard gate")

    comparisons = regression.get("comparisons")
    if not isinstance(comparisons, dict):
        raise ValueError("Rotation regression lacks comparisons")
    names = ("A_vs_B", "B_vs_C", "A_vs_C")
    if set(comparisons) != set(names):
        raise ValueError("Rotation regression must contain exactly A_vs_B, B_vs_C, A_vs_C")
    for name in names:
        pair = comparisons[name]
        if not isinstance(pair, dict) or "all_tokens" not in pair or "prompt_end" not in pair:
            raise ValueError(f"Rotation comparison {name} is incomplete")
        _verify_rotation_regression_metrics(pair["all_tokens"])
        _verify_prompt_end_metrics(pair["prompt_end"], expected_prompts=128)
        if bool(pair.get("passed")) != bool(pair["all_tokens"].get("passed")):
            raise ValueError(f"Rotation comparison {name} passed flag is inconsistent")
    expected_passed = all(bool(comparisons[name]["passed"]) for name in names)
    if bool(regression["passed"]) != expected_passed:
        raise ValueError("Rotation suite passed flag contradicts its three comparisons")
    diagnosis = regression.get("diagnosis")
    if not isinstance(diagnosis, dict) or not diagnosis.get("code"):
        raise ValueError("Rotation regression lacks structured diagnosis")


def main():
    args = parser(
        "Verify required artifacts for one ANN run", deployment_overrides=True
    ).parse_args()

    cfg, layout = setup(args.config)
    apply_deployment_overrides(args, cfg)

    with StageRun(
        "verify_artifacts",
        layout.logs_dir,
        cfg["experiment"],
    ) as run:

        required = [
            layout.data_dir / "train_manifest.json",
            layout.data_dir / "validation_manifest.json",
            layout.calibration_data_manifest_path,
            layout.ann_checkpoint_dir / "config.json",
            layout.ann_dir / "training_result.json",
        ]

        task = cfg["experiment"]["task"]

        if task == "tldr":
            required.append(
                layout.data_dir / "evaluation_manifest.json"
            )

        if task == "tldr":
            evaluation_manifest = read_json(layout.data_dir / "evaluation_manifest.json")
            tldr_layout = resolve_tldr_evaluation_layout(
                len(evaluation_manifest["indices"]),
                cfg["evaluation"].get("tldr_test_samples"),
            )
            eval_dir_name = "tldr"
            evaluation_files = ("metrics.json", "selection.json")
            evaluation_subdir = str(tldr_layout["dirname"])
        else:
            tldr_layout = None
            eval_dir_name = "lm_harness"
            evaluation_files = ("results.json",)
            evaluation_subdir = None

        def evaluation_paths(root, *, neuron="ann"):
            directory = root / "evaluation" / eval_dir_name
            if evaluation_subdir is not None:
                directory = directory / evaluation_subdir
            enabled = final_ann_evaluation_prefix_enabled(cfg) if neuron == "ann" else evaluation_prefix_enabled(cfg)
            directory = directory / prefix_enabled_dirname(enabled)
            directory = append_evaluation_num_samples_if_needed(directory, cfg, neuron=neuron)
            return [directory / name for name in evaluation_files]

        required.extend(evaluation_paths(layout.ann_dir))

        # --------------------------------------------------
        # Rotation / Prefix shared artifacts
        # --------------------------------------------------
        required.append(layout.conversion_site_dir / "calibration_state_manifest.json")
        if requires_ann_training_calibration(cfg):
            required.append(
                layout.ann_training_site_dir / "calibration_state_manifest.json"
            )

        if cfg["rotation"]["enabled"]:

            required.extend(
                [
                    layout.rotation_dir
                    / "rotation_state.pt",

                    layout.rotation_dir
                    / "rotation_regression.json",

                    layout.rotation_dir
                    / "rotation_summary.json",

                    layout.rotation_dir
                    / "fused_base"
                    / "config.json",

                ]
            )
            if requires_ann_training_calibration(cfg):
                required.append(layout.ann_training_site_dir / "calibration_state_manifest.json")

        # --------------------------------------------------
        # First existence check.
        #
        # prefix_state.json must exist before we can inspect
        # whether a non-empty Prefix KV cache is required.
        # --------------------------------------------------
        missing = [
            str(path)
            for path in required
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Missing required artifacts:\n"
                + "\n".join(missing)
            )
        if requires_pre_finetuning_prefix(cfg) and training_prefix_enabled(cfg):
            _validate_prefix_artifact(
                cfg, layout, layout.ann_training_prefix_dir,
                label="ANN-training Pre-finetuning",
            )

        final_ann_prefix_root = _final_ann_prefix_root(cfg, layout)
        if final_ann_prefix_root is not None:
            _validate_prefix_artifact(
                cfg, layout, final_ann_prefix_root,
                label=(
                    "Post-finetuning Final ANN"
                    if final_ann_evaluation_prefix_artifact_stage(cfg) == "post_finetuning"
                    else "Pre-finetuning Final ANN"
                ),
            )

        selected_snn_prefix_info = {"token_ids": []}
        if conversion_prefix_enabled(cfg) or evaluation_prefix_enabled(cfg):
            selected_snn_prefix_info = _validate_prefix_artifact(
                cfg, layout, layout.conversion_prefix_dir,
                label="Selected SNN",
            )

        _validate_aware_final_ann_training_provenance(cfg, layout)

        if cfg["rotation"]["enabled"]:
            regression_path = layout.rotation_dir / "rotation_regression.json"
            regression = read_json(regression_path)
            _verify_rotation_regression_suite(regression)
            summary = read_json(layout.rotation_dir / "rotation_summary.json")
            if (
                summary.get("random_hadamard_orientation") != "DU"
                or summary.get("precision_policy") != "roste_aligned_v1"
                or int(summary.get("rotation_regression_format_version", -1)) != 4
            ):
                raise ValueError("Rotation summary metadata is incompatible with regression v4")
            calibration_manifest = layout.canonical_preprocessing_calibration_manifest_path
            recorded_manifest = regression.get("calibration_manifest_path")
            if (
                not recorded_manifest
                or Path(recorded_manifest).resolve() != calibration_manifest.resolve()
            ):
                raise ValueError(
                    "Rotation regression does not reference the current task calibration manifest"
                )
            if regression.get("calibration_manifest_sha256") != sha256_file(
                calibration_manifest
            ):
                raise ValueError("Rotation regression calibration manifest hash mismatch")

        # --------------------------------------------------
        # Calibration
        # --------------------------------------------------
        reused = conversion_reuses_ann_training_artifacts(cfg)
        calibration = validate_calibration(
            layout.conversion_site_dir,
            clip_policy="forbid_all",
        )
        validate_site_state_bundle(
            layout.conversion_site_dir, clip_policy="forbid_all"
        )
        source_manifest = read_json(
            layout.conversion_site_dir / "calibration_state_manifest.json"
        )
        expected_flags = {
            "purpose": ("ann_training_calibration" if reused else "post_finetuning_conversion_calibration"),
            "eligible_for_ann_training": reused,
            "eligible_for_conversion": True,
            "conversion_reuse_policy": ("non_vanilla_when_selected" if reused else "final_ann_only"),
            "post_finetuning_recalibration": not reused,
            "state_profile": "stage_a_common_states",
            "common_clip_required": False,
            "common_clip_generated": False,
            "common_clip_application_control": "replacement.common_clip_enabled",
        }
        if reused and is_aware_ann_mode(cfg):
            training_result = read_json(layout.ann_dir / "training_result.json")
            training_profile_root = Path(training_result["ann_training_clip_profile_root"])
            required.append(training_profile_root / "clip_profile_manifest.json")
            validate_clip_profile(
                layout.ann_training_site_dir,
                training_profile_root,
                phase_T=int(training_result["ann_training_phase_T"]),
                mtn_T=int(training_result["ann_training_mtn_T"]),
                group_size=int(training_result["ann_training_calibration_group_size"]),
                num_samples=int(training_result["ann_training_calibration_num_samples"]),
            )
        _require_manifest_flags(source_manifest, expected_flags, "Conversion source")
        _verify_hashes(source_manifest, "Conversion source calibration")
        validate_temporal_policy(source_manifest, context="Conversion source manifest")
        _verify_grouped_calibration(cfg, layout, source_manifest, calibration)
        validate_conversion_prefix(cfg, layout)

        conversions = {}

        for neuron in (
            "phase",
            "gif",
            "mtn",
        ):
            path = (
                layout.snn_conversion_dir(neuron)
                / "conversion_metadata.json"
            )

            conversions[neuron] = path.exists()
            required.extend(evaluation_paths(layout.snn_dir(neuron), neuron=neuron))

        missing = [
            str(path)
            for path in required
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Missing required artifacts:\n"
                + "\n".join(missing)
            )

        _verify_final_ann_forward_metadata(
            cfg, layout, evaluation_paths(layout.ann_dir)[0]
        )

        for neuron in ("phase", "gif", "mtn"):
            metadata = validate_conversion_metadata(cfg, layout, neuron)
            expected_steps = {
                "phase": int(cfg["phase"]["T"]),
                "gif": 2,
                "mtn": int(cfg["mtn"]["T"]),
            }[neuron]
            if int(metadata.get("full_temporal_steps", -1)) != expected_steps:
                raise ValueError(f"{neuron} conversion metadata has incompatible temporal steps")
            metrics_path = evaluation_paths(layout.snn_dir(neuron), neuron=neuron)[0]
            metrics = read_json(metrics_path)
            policy_source = metrics.get("snn2_metadata", metrics)
            validate_temporal_policy(policy_source, context=str(metrics_path))
            if (
                policy_source.get("neuron") != neuron
                or int(policy_source.get("full_temporal_steps", -1)) != int(metadata["full_temporal_steps"])
            ):
                raise ValueError(f"SNN metrics disagree with conversion: {metrics_path}")
            _validate_snn_forward_metadata(
                policy_source, neuron=neuron, metrics_path=metrics_path
            )
            _validate_snn_source_metadata(
                cfg, metadata, policy_source, metrics_path=metrics_path
            )

        if tldr_layout is not None:
            expected_count = int(tldr_layout["selected_test_samples"])
            expected_full = bool(tldr_layout["is_full_test"])
            expected_sampling = "full_split" if expected_full else "seeded_random_without_replacement"
            evaluation_roots = [
                ("ann", layout.ann_dir),
                *((neuron, layout.snn_dir(neuron)) for neuron in ("phase", "gif", "mtn")),
            ]
            for neuron, root in evaluation_roots:
                selection_path = evaluation_paths(root, neuron=neuron)[1]
                selection = read_json(selection_path)
                if len(selection.get("indices", [])) != expected_count:
                    raise ValueError(f"TL;DR selection size mismatch: {selection_path}")
                if selection.get("sampling") != expected_sampling:
                    raise ValueError(f"TL;DR sampling policy mismatch: {selection_path}")
                if not expected_full and selection.get("seed") != int(cfg["evaluation"].get("tldr_test_seed", 42)):
                    raise ValueError(f"TL;DR selection seed mismatch: {selection_path}")

        result = {
            "required_files": len(required),
            "calibration": calibration,
            "conversion_descriptors": conversions,
            **_selected_snn_prefix_summary(selected_snn_prefix_info),
            **topology_metadata(),
        }

        write_json(
            layout.root
            / "artifact_verification.json",
            result,
        )

        run.event(
            "verification_complete",
            **result,
        )


if __name__ == "__main__":
    main()
