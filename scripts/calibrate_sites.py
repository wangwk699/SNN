from _common import parser, setup

from snn2.calibration import calibration_provenance, collect_site_statistics
from snn2.controller import SiteController
from snn2.data import load_selected_raw
from snn2.logging_utils import StageRun
from snn2.model_integration import install_model_integration
from snn2.modeling import load_model, load_tokenizer, model_source_for_stage, prefix_key_values_for_stage, rotation_state
from snn2.config import training_prefix_enabled


def main():
    arg_parser = parser("Collect activation statistics for every replacement site")
    arg_parser.add_argument("--stage", required=True, choices=("ann_training", "vanilla_analysis", "post_finetuning"))
    args = arg_parser.parse_args()
    scope = "policy_shared" if args.stage in {"ann_training", "vanilla_analysis"} else "run"
    cfg, layout = setup(args.config, config_scope=scope)
    if args.stage == "ann_training":
        if (not cfg["rotation"]["enabled"] or not training_prefix_enabled(cfg)):
            raise ValueError("ann_training calibration requires a rotated non-vanilla config")
    if args.stage == "vanilla_analysis":
        if (cfg["experiment"]["ann_mode"] != "vanilla" or cfg["rotation"]["enabled"] or cfg["prefix"]["enabled"]):
            raise ValueError("vanilla_analysis calibration requires a vanilla config with rotation/prefix disabled")
    site_root = {"ann_training": layout.ann_training_site_dir, "vanilla_analysis": layout.vanilla_analysis_site_dir, "post_finetuning": layout.post_finetuning_site_dir}[args.stage]
    purpose = {"ann_training": "ann_training_calibration", "vanilla_analysis": "vanilla_analysis_calibration", "post_finetuning": "post_finetuning_conversion_calibration"}[args.stage]
    source = model_source_for_stage(cfg, layout, stage=args.stage)
    with StageRun(f"calibrate_sites_{args.stage}", layout.policy_logs_dir if args.stage in {"ann_training", "vanilla_analysis"} else layout.logs_dir, cfg["experiment"]) as run:
        if args.stage == "post_finetuning" and not layout.ann_checkpoint_dir.exists():
            raise FileNotFoundError(layout.ann_checkpoint_dir)
        model = load_model(cfg, source, training=False, device_map=cfg["calibration"].get("device_map"))
        tokenizer = load_tokenizer(cfg, source)
        controller = SiteController(mode="collect")
        install_model_integration(model, controller, None if args.stage == "vanilla_analysis" else rotation_state(cfg, layout))
        result = collect_site_statistics(model, controller, tokenizer, load_selected_raw(cfg, layout).calibration, cfg, None if args.stage == "vanilla_analysis" else prefix_key_values_for_stage(cfg, layout, stage="ann_training" if args.stage == "ann_training" else "post_finetuning"), site_root, purpose=purpose, materialize_states=args.stage != "vanilla_analysis", extra_metadata=calibration_provenance(cfg, layout, stage=args.stage))
        run.event("site_statistics_saved", stage=args.stage, sites=len(result["statistics"].get("sites", {})), site_root=str(site_root))


if __name__ == "__main__":
    main()
