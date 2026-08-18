from _common import parser, setup

import torch

from snn2.artifacts import write_json
from snn2.logging_utils import StageRun
from snn2.modeling import load_model, load_tokenizer
from snn2.rotation import fuse_rotations, save_rotation_state


def main():
    args = parser("Fuse fixed random Hadamard rotations into the Base checkpoint").parse_args()
    cfg, layout = setup(args.config, config_scope="policy_shared")
    with StageRun("prepare_rotation", layout.policy_logs_dir, cfg["experiment"]) as run:
        if not cfg["rotation"]["enabled"]:
            disabled = {"enabled": False, "reason": "vanilla baseline"}
            write_json(layout.root / "policy" / "rotation_disabled.json", disabled)
            run.event("rotation_disabled")
            return
        model = load_model(cfg, cfg["experiment"]["model_name"], training=False)
        tokenizer = load_tokenizer(cfg)
        state = fuse_rotations(
            model,
            seed=int(cfg["rotation"]["seed"]),
            device=cfg["rotation"].get("fusion_device", "cuda"),
        )
        save_rotation_state(state, layout.rotation_dir / "rotation_state.pt")
        destination = layout.rotation_dir / "fused_base"
        model.save_pretrained(destination, safe_serialization=True)
        tokenizer.save_pretrained(destination)
        write_json(
            layout.rotation_dir / "rotation_summary.json",
            {key: value for key, value in state.items() if key != "specs"},
        )
        run.event("fused_checkpoint_saved", path=str(destination.resolve()))


if __name__ == "__main__":
    main()
