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
    RotationRegressionSuiteError,
    fuse_rotations,
    save_rotation_state,
    validate_rotation_regression_suite,
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
                "format_version": 4,
                "purpose": "three_way_rotation_regression",
                "model_name": cfg["experiment"]["model_name"],
                "rotation_seed": int(cfg["rotation"]["seed"]),
                "num_samples": 0,
                "passed": False,
                "status": "in_progress",
            },
        )
        tokenizer = load_tokenizer(cfg)
        device_map = cfg["calibration"].get("device_map", "auto")

        # A: untouched Hugging Face Base; no SNN2 integration and no Rotation.
        model_a = load_model(
            cfg, cfg["experiment"]["model_name"], training=False, device_map=device_map
        )
        # B: the same Base plus identity SNN2 integration, without Rotation.
        model_b = load_model(
            cfg, cfg["experiment"]["model_name"], training=False, device_map=device_map
        )
        # C: the same Base with fused/offline Rotation and online R3/R4.
        model_c = load_model(
            cfg, cfg["experiment"]["model_name"], training=False, device_map=device_map
        )

        state = fuse_rotations(
            model_c,
            seed=int(cfg["rotation"]["seed"]),
            device=cfg["rotation"].get("fusion_device", "cuda"),
        )
        controller_b = SiteController(mode="identity")
        install_model_integration(model_b, controller_b, None)
        model_b.config.snn2_regression_variant = "B_identity_no_rotation"
        controller_c = SiteController(mode="identity")
        install_model_integration(model_c, controller_c, state)

        calibration = load_selected_raw(cfg, layout).calibration
        manifest_path = layout.calibration_data_manifest_path
        try:
            regression = validate_rotation_regression_suite(
                model_a,
                model_b,
                model_c,
                tokenizer,
                calibration,
                cfg,
                controller_b,
                controller_c,
                calibration_manifest_path=manifest_path,
                calibration_manifest_sha256=sha256_file(manifest_path),
            )
        except RotationRegressionSuiteError as exc:
            write_json(regression_path, exc.result)
            run.event(
                "rotation_regression_failed",
                path=str(regression_path.resolve()),
                diagnosis=exc.result["diagnosis"]["code"],
            )
            raise
        write_json(regression_path, regression)
        run.event("rotation_regression_passed", path=str(regression_path.resolve()))

        del model_a, model_b
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Integration changes Python forwards but not checkpoint weights. Restore
        # load-time attention metadata before saving the reusable fused Base.
        model_c.config._attn_implementation = "eager"
        model_c.config._attn_implementation_internal = "eager"
        if hasattr(model_c.config, "snn2_site_integration"):
            delattr(model_c.config, "snn2_site_integration")

        save_rotation_state(state, layout.rotation_dir / "rotation_state.pt")
        destination = layout.rotation_dir / "fused_base"
        model_c.save_pretrained(destination, safe_serialization=True)
        tokenizer.save_pretrained(destination)
        write_json(
            layout.rotation_dir / "rotation_summary.json",
            {
                **{key: value for key, value in state.items() if key != "specs"},
                "hadamard_backend": "fast_hadamard_transform",
                "fast_hadamard_transform_module_path": _fast_hadamard_module_path(),
                "random_hadamard_orientation": "DU",
                "precision_policy": "roste_aligned_v1",
                "precision_details": {
                    "rmsnorm_fusion": "float64",
                    "R1_offline": "float64_explicit_matmul",
                    "R2_value_output_side": "float64_matmul",
                    "R2_o_proj_input_side": "float32_fht",
                    "R3_online": "preserve_input_dtype",
                    "R4_offline": "float32_fht",
                    "R4_online": "float32_fht_then_cast_back",
                },
                "rotation_regression_format_version": 4,
                "rotation_regression_path": str(regression_path.resolve()),
                "rotation_regression_passed": True,
            },
        )
        run.event("fused_checkpoint_saved", path=str(destination.resolve()))


if __name__ == "__main__":
    main()
