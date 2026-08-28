from _common import parser, setup

from snn2.artifacts import read_json
from snn2.calibration import calibration_provenance, materialize_calibration_states
from snn2.config import requires_ann_training_calibration, requires_post_finetuning_artifacts
from snn2.logging_utils import StageRun


def main():
    arg_parser = parser("Materialize calibration states from Stage-A statistics")
    arg_parser.add_argument("--stage", required=True, choices=("ann_training", "post_finetuning", "vanilla_analysis"))
    args = arg_parser.parse_args()
    if args.stage == "vanilla_analysis":
        raise ValueError("vanilla_analysis is statistics-only and has no Stage B")
    scope = "ann_training_states" if args.stage == "ann_training" else "post_finetuning_states"
    cfg, layout = setup(args.config, config_scope=scope)
    if args.stage == "ann_training" and not requires_ann_training_calibration(cfg):
        raise ValueError("ANN-training calibration is only used by phase_aware/gif_aware modes")
    if args.stage == "post_finetuning" and not requires_post_finetuning_artifacts(cfg):
        raise ValueError("Aware modes reuse ANN-training calibration")
    statistics_root = layout.ann_training_statistics_dir if args.stage == "ann_training" else layout.post_finetuning_statistics_dir
    state_root = layout.ann_training_state_dir if args.stage == "ann_training" else layout.post_finetuning_state_dir
    stats_manifest = read_json(statistics_root / "statistics_manifest.json")
    expected_layers = int(stats_manifest["expected_num_hidden_layers"])
    logs_dir = layout.ann_training_state_logs_dir if args.stage == "ann_training" else layout.post_finetuning_state_logs_dir
    with StageRun(f"materialize_calibration_states_{args.stage}", logs_dir, cfg["experiment"]) as run:
        result = materialize_calibration_states(
            statistics_root, state_root, cfg, calibration_provenance(cfg, layout, stage=args.stage),
            include_clip=args.stage == "ann_training", expected_num_hidden_layers=expected_layers,
        )
        run.event("calibration_states_saved", stage=args.stage, sites=len(result.get("sites", {})), state_root=str(state_root))


if __name__ == "__main__":
    main()
