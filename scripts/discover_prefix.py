from _common import parser, setup

from snn2.artifacts import write_json
from snn2.config import post_finetuning_prefix_enabled, training_prefix_enabled
from snn2.controller import SiteController
from snn2.data import load_selected_raw
from snn2.logging_utils import StageRun
from snn2.model_integration import install_model_integration
from snn2.modeling import load_model, load_tokenizer, model_source_for_stage, rotation_state
from snn2.prefix import discover_prefix_tokens
from snn2.prefix_cache import build_prefix_key_values, save_prefix_key_values


def main():
    arg_parser = parser("Discover fixed PrefixQuant tokens")
    arg_parser.add_argument(
        "--stage",
        required=True,
        choices=("ann_training", "rotated_pre_finetuning", "post_finetuning"),
    )
    args = arg_parser.parse_args()
    config_scope = {
        "ann_training": "policy_shared",
        "rotated_pre_finetuning": "rotated_pre_finetuning",
        "post_finetuning": "run",
    }[args.stage]
    cfg, layout = setup(args.config, config_scope=config_scope)
    logs_dir = {
        "ann_training": layout.policy_logs_dir,
        "rotated_pre_finetuning": layout.rotated_pre_finetuning_logs_dir,
        "post_finetuning": layout.logs_dir,
    }[args.stage]
    with StageRun(f"discover_prefix_{args.stage}", logs_dir, cfg["experiment"]) as run:
        if args.stage == "ann_training":
            if not cfg["rotation"]["enabled"] or not training_prefix_enabled(cfg):
                raise ValueError("ann_training prefix requires a rotated non-vanilla config")
            output_dir = layout.ann_training_prefix_dir
        elif args.stage == "rotated_pre_finetuning":
            if not bool(cfg["rotation"]["enabled"]):
                raise ValueError("rotated_pre_finetuning prefix requires rotation.enabled=true")
            fused_config = layout.rotation_dir / "fused_base" / "config.json"
            rotation_state_path = layout.rotation_dir / "rotation_state.pt"
            missing = [path for path in (fused_config, rotation_state_path) if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    "Rotated pre-finetuning prefix requires rotation artifacts. "
                    "Run `python scripts/prepare_rotation.py --config ...` first. "
                    f"Missing: {', '.join(str(path) for path in missing)}"
                )
            output_dir = layout.rotated_pre_finetuning_prefix_dir
        else:
            if not post_finetuning_prefix_enabled(cfg):
                raise ValueError("post-finetuning prefix is required by the main protocol")
            if not layout.ann_checkpoint_dir.exists():
                raise FileNotFoundError(layout.ann_checkpoint_dir)
            output_dir = layout.post_finetuning_prefix_dir
        source = model_source_for_stage(cfg, layout, stage=args.stage)
        model = load_model(cfg, source, training=False, device_map=cfg["calibration"].get("device_map"))
        tokenizer = load_tokenizer(cfg, source)
        install_model_integration(model, SiteController(mode="identity"), rotation_state(cfg, layout))
        state = discover_prefix_tokens(model, tokenizer, load_selected_raw(cfg, layout).calibration, cfg, output_dir / "prefix_state.json")
        values = build_prefix_key_values(model, state["prefix_token_ids"])
        cache_path = output_dir / "prefixed_key_values.pt"
        if values is not None:
            save_prefix_key_values(cache_path, values)
        elif cache_path.exists():
            cache_path.unlink()
        run.event("prefix_saved", stage=args.stage, count=len(state["prefix_token_ids"]), prefix_root=str(output_dir), kv_cache_saved=values is not None)


if __name__ == "__main__":
    main()
