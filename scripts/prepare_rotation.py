from _common import parser, setup

import gc

import torch

from snn2.artifacts import sha256_file, write_json
from snn2.controller import SiteController
from snn2.data import load_selected_raw
from snn2.logging_utils import StageRun
from snn2.model_integration import install_model_integration
from snn2.modeling import load_model, load_tokenizer
from snn2.rotation import (
    RotationRegressionError,
    fuse_rotations,
    save_rotation_state,
    validate_rotation_logits,
)


def _fast_hadamard_module_path() -> str:
    import fast_hadamard_transform

    return str(fast_hadamard_transform.__file__)


def main():
    args = parser("Fuse fixed random Hadamard rotations into the Base checkpoint").parse_args()
    cfg, layout = setup(args.config, config_scope="policy_shared")
    with StageRun("prepare_rotation", layout.policy_logs_dir, cfg["experiment"]) as run:
        if not cfg["rotation"]["enabled"]:
            disabled = {"enabled": False, "reason": "vanilla baseline"}
            write_json(layout.root / "policy" / "rotation_disabled.json", disabled)
            run.event("rotation_disabled")
            return

        regression_path = layout.rotation_dir / "rotation_regression.json"
        write_json(
            regression_path,
            {
                "format_version": 1,
                "purpose": "base_vs_rotated_logits_regression",
                "model_name": cfg["experiment"]["model_name"],
                "rotation_seed": int(cfg["rotation"]["seed"]),
                "num_samples": 0,
                "passed": False,
                "status": "in_progress",
            },
        )
        tokenizer = load_tokenizer(cfg)
        device_map = cfg["calibration"].get("device_map", "auto")
        base_model = load_model(
            cfg,
            cfg["experiment"]["model_name"],
            training=False,
            device_map=device_map,
        )
        model = load_model(
            cfg,
            cfg["experiment"]["model_name"],
            training=False,
            device_map=device_map,
        )

        state = fuse_rotations(
            model,
            seed=int(cfg["rotation"]["seed"]),
            device=cfg["rotation"].get("fusion_device", "cuda"),
        )

        # Base 和 Rotated 两边都使用完全相同的 SNN2 forward integration，
        # 唯一区别是 Base 不加载 rotation_state，而 Rotated 加载完整 R3/R4 rotation。
        base_controller = SiteController(mode="identity")
        install_model_integration(
            base_model,
            base_controller,
            None,
        )

        rotated_controller = SiteController(mode="identity")
        install_model_integration(
            model,
            rotated_controller,
            state,
        )

        calibration = load_selected_raw(cfg, layout).calibration
        manifest_path = layout.data_dir / "calibration_manifest.json"

        try:
            regression = validate_rotation_logits(
                base_model,
                model,
                tokenizer,
                calibration,
                cfg,
                rotated_controller,   # 注意：这里不能再写 controller
                calibration_manifest_path=manifest_path,
                calibration_manifest_sha256=sha256_file(manifest_path),
            )
        except RotationRegressionError as exc:
            write_json(regression_path, exc.result)
            run.event(
                "rotation_regression_failed",
                path=str(regression_path.resolve()),
            )
            raise
        write_json(regression_path, regression)
        run.event("rotation_regression_passed", path=str(regression_path.resolve()))

        del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Integration changes Python forwards but not checkpoint weights. Restore
        # load-time attention metadata before saving the reusable fused Base.
        model.config._attn_implementation = "eager"
        model.config._attn_implementation_internal = "eager"
        if hasattr(model.config, "snn2_site_integration"):
            delattr(model.config, "snn2_site_integration")

        save_rotation_state(state, layout.rotation_dir / "rotation_state.pt")
        destination = layout.rotation_dir / "fused_base"
        model.save_pretrained(destination, safe_serialization=True)
        tokenizer.save_pretrained(destination)
        write_json(
            layout.rotation_dir / "rotation_summary.json",
            {
                **{key: value for key, value in state.items() if key != "specs"},
                "hadamard_backend": "fast_hadamard_transform",
                "fast_hadamard_transform_module_path": _fast_hadamard_module_path(),
                "rotation_regression_path": str(regression_path.resolve()),
                "rotation_regression_passed": True,
            },
        )
        run.event("fused_checkpoint_saved", path=str(destination.resolve()))


if __name__ == "__main__":
    main()
