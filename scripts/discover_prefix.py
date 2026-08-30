from _common import parser, setup

from snn2.artifacts import sha256_file, write_json
from snn2.controller import SiteController
from snn2.data import load_selected_raw
from snn2.logging_utils import StageRun
from snn2.model_integration import install_model_integration
from snn2.modeling import load_model, load_tokenizer, model_source_for_stage, rotation_state
from snn2.prefix import discover_prefix_tokens
from snn2.prefix_cache import build_prefix_key_values, save_prefix_key_values
from snn2.config import requires_post_finetuning_artifacts, requires_pre_finetuning_prefix


def main():
    arg_parser = parser("Discover fixed PrefixQuant tokens")
    arg_parser.add_argument(
        "--stage",
        required=True,
        choices=("pre_finetuning", "ann_training", "rotated_pre_finetuning", "post_finetuning"),
    )
    args = arg_parser.parse_args()
    canonical_stage = (
        "pre_finetuning"
        if args.stage in {"pre_finetuning", "ann_training", "rotated_pre_finetuning"}
        else "post_finetuning"
    )
    config_scope = "policy_shared" if canonical_stage == "pre_finetuning" else "run"
    cfg, layout = setup(args.config, config_scope=config_scope)
    if canonical_stage == "pre_finetuning" and not requires_pre_finetuning_prefix(cfg):
        raise ValueError("vanilla does not use or require a Pre-finetuning Prefix")
    if canonical_stage == "post_finetuning" and not requires_post_finetuning_artifacts(cfg):
        raise ValueError(
            "This ANN mode reuses the pre-finetuning Prefix for SNN conversion; "
            "do not rediscover a post-finetuning Prefix."
        )
    logs_dir = layout.policy_logs_dir if canonical_stage == "pre_finetuning" else layout.logs_dir
    with StageRun(f"discover_prefix_{args.stage}", logs_dir, cfg["experiment"]) as run:
        if canonical_stage == "pre_finetuning":
            missing = []
            if bool(cfg["rotation"]["enabled"]):
                fused_config = layout.rotation_dir / "fused_base" / "config.json"
                rotation_state_path = layout.rotation_dir / "rotation_state.pt"
                missing = [
                    path
                    for path in (fused_config, rotation_state_path)
                    if not path.exists()
                ]
            if missing:
                raise FileNotFoundError(
                    "Pre-finetuning prefix requires rotation artifacts. "
                    "Run `python scripts/prepare_rotation.py --config ...` first. "
                    f"Missing: {', '.join(str(path) for path in missing)}"
                )
            output_dir = layout.ann_training_prefix_dir
        else:
            if not layout.ann_checkpoint_dir.exists():
                raise FileNotFoundError(layout.ann_checkpoint_dir)
            output_dir = layout.post_finetuning_prefix_dir
        source = model_source_for_stage(cfg, layout, stage=canonical_stage)
        model = load_model(cfg, source, training=False, device_map=cfg["calibration"].get("device_map"))
        tokenizer = load_tokenizer(cfg, source)
        install_model_integration(model, SiteController(mode="identity"), rotation_state(cfg, layout))
        state = discover_prefix_tokens(model, tokenizer, load_selected_raw(cfg, layout).calibration, cfg, output_dir / "prefix_state.json")
        manifest_path = layout.calibration_data_manifest_path
        state.update({
            "discovery_num_samples": int(cfg["calibration"]["num_samples"]),
            "discovery_data_source": "stage_a_calibration_selection",
            "discovery_manifest_path": str(manifest_path.resolve()),
            "discovery_manifest_sha256": sha256_file(manifest_path),
        })
        write_json(output_dir / "prefix_state.json", state)
        values = build_prefix_key_values(model, state["prefix_token_ids"])
        cache_path = output_dir / "prefixed_key_values.pt"
        if values is not None:
            save_prefix_key_values(cache_path, values)
        elif cache_path.exists():
            cache_path.unlink()
        run.event("prefix_saved", stage=canonical_stage, count=len(state["prefix_token_ids"]), prefix_root=str(output_dir), kv_cache_saved=values is not None)


if __name__ == "__main__":
    main()
