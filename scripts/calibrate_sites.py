from _common import parser, setup

from snn2.calibration import (
    calibration_provenance,
    collect_site_statistics,
    materialize_clip_profile,
)
from snn2.config import requires_ann_training_calibration, requires_post_finetuning_artifacts
from snn2.controller import SiteController
from snn2.data import load_selected_raw
from snn2.logging_utils import StageRun
from snn2.model_integration import install_model_integration
from snn2.modeling import (
    load_model,
    load_tokenizer,
    model_source_for_stage,
    prefix_key_values_for_stage,
    rotation_state,
)


def main():
    arg_parser = parser("Run calibration Stage A statistics/state or Stage B Clip materialization")
    arg_parser.add_argument(
        "--stage", required=True,
        choices=("ann_training", "vanilla_analysis", "post_finetuning"),
    )
    arg_parser.add_argument("--calibration-phase", required=True, choices=("A", "B"))
    args = arg_parser.parse_args()
    scope = {
        ("ann_training", "A"): "ann_training_calibration",
        ("ann_training", "B"): "ann_training_clip_profile",
        ("vanilla_analysis", "A"): "vanilla_analysis_calibration",
        ("post_finetuning", "A"): "post_finetuning_calibration",
        ("post_finetuning", "B"): "post_finetuning_clip_profile",
    }.get((args.stage, args.calibration_phase))
    if scope is None:
        raise ValueError("vanilla_analysis does not materialize Stage B Clip profiles")
    cfg, layout = setup(args.config, config_scope=scope)
    if args.stage == "ann_training" and not requires_ann_training_calibration(cfg):
        raise ValueError("ANN-training calibration is only used by phase_aware/gif_aware modes")
    if args.stage == "post_finetuning" and not requires_post_finetuning_artifacts(cfg):
        raise ValueError("Aware ANN modes reuse ANN-training calibration; do not run post-finetuning calibration")
    if args.stage == "vanilla_analysis" and (
        cfg["experiment"]["ann_mode"] != "vanilla" or cfg["rotation"]["enabled"]
    ):
        raise ValueError("vanilla_analysis calibration requires a vanilla config")
    site_root = {
        "ann_training": layout.ann_training_site_dir,
        "vanilla_analysis": layout.vanilla_analysis_site_dir,
        "post_finetuning": layout.post_finetuning_site_dir,
    }[args.stage]
    profile_root = {
        "ann_training": layout.ann_training_clip_profile_dir,
        "post_finetuning": layout.post_finetuning_clip_profile_dir,
    }.get(args.stage)
    logs_dir = {
        ("ann_training", "A"): layout.ann_training_calibration_logs_dir,
        ("ann_training", "B"): layout.ann_training_clip_profile_logs_dir,
        ("vanilla_analysis", "A"): layout.vanilla_analysis_calibration_logs_dir,
        ("post_finetuning", "A"): layout.post_finetuning_conversion_calibration_logs_dir,
        ("post_finetuning", "B"): layout.post_finetuning_clip_profile_logs_dir,
    }[(args.stage, args.calibration_phase)]
    with StageRun(
        f"calibrate_sites_{args.stage}_phase_{args.calibration_phase}",
        logs_dir,
        cfg["experiment"],
    ) as run:
        if args.calibration_phase == "B":
            if profile_root is None:
                raise ValueError("vanilla_analysis does not materialize Stage B Clip profiles")
            result = materialize_clip_profile(site_root, profile_root, cfg)
            run.event("clip_profile_saved", stage=args.stage, profile_root=str(profile_root), sites=len(result["sites"]))
            return
        if args.stage == "post_finetuning" and not layout.ann_checkpoint_dir.exists():
            raise FileNotFoundError(layout.ann_checkpoint_dir)
        purpose = {
            "ann_training": "ann_training_calibration",
            "vanilla_analysis": "vanilla_analysis_calibration",
            "post_finetuning": "post_finetuning_conversion_calibration",
        }[args.stage]
        source = model_source_for_stage(cfg, layout, stage=args.stage)
        model = load_model(cfg, source, training=False, device_map=cfg["calibration"].get("device_map"))
        tokenizer = load_tokenizer(cfg, source)
        controller = SiteController(mode="collect")
        install_model_integration(
            model, controller,
            None if args.stage == "vanilla_analysis" else rotation_state(cfg, layout),
        )
        result = collect_site_statistics(
            model, controller, tokenizer, load_selected_raw(cfg, layout).calibration,
            cfg,
            None if args.stage == "vanilla_analysis" else prefix_key_values_for_stage(
                cfg, layout, stage="ann_training" if args.stage == "ann_training" else "post_finetuning"
            ),
            site_root, purpose=purpose,
            materialize_states=args.stage != "vanilla_analysis",
            extra_metadata=calibration_provenance(cfg, layout, stage=args.stage),
        )
        run.event("stage_a_saved", stage=args.stage, sites=len(result["statistics"].get("sites", {})), site_root=str(site_root))


if __name__ == "__main__":
    main()
