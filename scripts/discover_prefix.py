from _common import parser, setup

from snn2.artifacts import write_json
from snn2.controller import SiteController
from snn2.data import load_selected_raw
from snn2.logging_utils import StageRun
from snn2.model_integration import install_model_integration
from snn2.modeling import load_model, load_tokenizer, model_source, rotation_state
from snn2.prefix import discover_prefix_tokens


def main():
    args = parser("Discover PrefixQuant prefixed outlier tokens").parse_args()
    cfg, layout = setup(args.config)
    with StageRun("discover_prefix", layout.logs_dir, cfg["experiment"]) as run:
        output = layout.prefix_dir / "prefix_state.json"
        if not cfg["prefix"]["enabled"]:
            write_json(
                layout.root / "policy" / "prefix_disabled.json",
                {"enabled": False, "prefix_token_ids": [], "reason": "vanilla baseline"},
            )
            run.event("prefix_disabled")
            return
        source = model_source(cfg, layout)
        model = load_model(
            cfg, source, training=False, device_map=cfg["calibration"].get("device_map")
        )
        tokenizer = load_tokenizer(cfg, source)
        controller = SiteController(mode="identity")
        install_model_integration(model, controller, rotation_state(cfg, layout))
        bundle = load_selected_raw(cfg, layout)
        state = discover_prefix_tokens(model, tokenizer, bundle.calibration, cfg, output)
        run.event("prefix_saved", count=len(state["prefix_token_ids"]), ids=state["prefix_token_ids"])


if __name__ == "__main__":
    main()
