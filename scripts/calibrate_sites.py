from _common import parser, setup

from snn2.calibration import collect_site_statistics
from snn2.controller import SiteController
from snn2.data import load_selected_raw
from snn2.logging_utils import StageRun
from snn2.model_integration import install_model_integration
from snn2.modeling import (
    load_model,
    load_tokenizer,
    model_source,
    prefix_key_values,
    rotation_state,
)


def main():
    args = parser("Collect and save statistics for every activation replacement site").parse_args()
    cfg, layout = setup(args.config, config_scope="policy_shared")
    with StageRun("calibrate_sites", layout.policy_logs_dir, cfg["experiment"]) as run:
        source = model_source(cfg, layout)
        model = load_model(
            cfg, source, training=False, device_map=cfg["calibration"].get("device_map")
        )
        tokenizer = load_tokenizer(cfg, source if cfg["rotation"]["enabled"] else None)
        controller = SiteController(mode="collect")
        install_model_integration(model, controller, rotation_state(cfg, layout))
        bundle = load_selected_raw(cfg, layout)
        result = collect_site_statistics(
            model,
            controller,
            tokenizer,
            bundle.calibration,
            cfg,
            prefix_key_values(cfg, layout),
            layout.site_dir,
        )
        run.event("site_states_saved", sites=len(result["states"]["sites"]))


if __name__ == "__main__":
    main()

