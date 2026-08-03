#!/usr/bin/env python3
import json
import sys
from pathlib import Path

XLM_REQUIRED = {
    "preproc": True,
    "postproc": True,
    "sim_env": "so100",
    "exec_size": 6,
    "fs": 0,
}

MODEL_PATH_KEYS = (
    "siglip_hbm_path",
    "paligemma_hbm_path",
    "action_hbm_path",
)

CORE_KEYS = (
    "siglip_bpu_core",
    "paligemma_bpu_core",
    "action_bpu_core",
)


def main() -> int:
    config_path = Path(sys.argv[1])
    config = json.loads(config_path.read_text())
    standalone = config.get("runtime") == "standalone_dnn"
    if not standalone:
        for key, expected in XLM_REQUIRED.items():
            actual = config.get(key)
            if actual != expected:
                raise RuntimeError(f"{key} must be {expected!r}, got {actual!r}")
    real_camera_num = config.get("real_camera_num", 1)
    if not isinstance(real_camera_num, int) or not 1 <= real_camera_num <= 3:
        raise RuntimeError(
            f"real_camera_num must be an integer from 1 to 3, got {real_camera_num!r}"
        )
    denoise_num = config.get("denoise_num")
    if not isinstance(denoise_num, int) or not 1 <= denoise_num <= 10:
        raise RuntimeError(
            f"denoise_num must be an integer from 1 to 10, got {denoise_num!r}"
        )
    for key in MODEL_PATH_KEYS:
        model_path = Path(config[key])
        if not model_path.is_file():
            raise RuntimeError(f"{key} does not exist: {model_path}")
    if standalone:
        prompt_embedding = Path(config["prompt_embedding_path"])
        if not prompt_embedding.is_file():
            raise RuntimeError(
                f"prompt_embedding_path does not exist: {prompt_embedding}"
            )
        if prompt_embedding.stat().st_size != 48 * 2048 * 2:
            raise RuntimeError(
                "prompt_embedding_path must contain [1,48,2048] FP16 data"
            )
    else:
        tokenizer_dir = Path(config["tokenizer_dir"])
        if not tokenizer_dir.is_dir():
            raise RuntimeError(f"tokenizer_dir does not exist: {tokenizer_dir}")
    for key in CORE_KEYS:
        cores = config.get(key)
        if not isinstance(cores, list) or not cores:
            raise RuntimeError(f"{key} must be a non-empty list")
        if len(set(cores)) != len(cores) or any(core not in range(4) for core in cores):
            raise RuntimeError(f"{key} must contain unique core IDs from 0 to 3")
    if len(config["siglip_bpu_core"]) != 3:
        raise RuntimeError("siglip_bpu_core must assign one core to each camera slot")
    stats_path = Path(config["norm_stats_path"])
    stats = json.loads(stats_path.read_text())["norm_stats"]
    for name in ("state", "actions"):
        for field in ("mean", "std"):
            values = stats[name][field]
            if len(values) != 6:
                raise RuntimeError(f"{name}.{field} must contain 6 values")
    if "action_size" in config:
        raise RuntimeError("Do not set action_size; HBM width is fixed at 32")
    print("PI0_SO100_STANDALONE_CONFIG_OK" if standalone else "PI0_SO100_CONFIG_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
