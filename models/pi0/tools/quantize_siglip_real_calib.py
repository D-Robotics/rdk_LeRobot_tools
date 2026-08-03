#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as safetensors_load_file

os.environ.setdefault("DEV_B30_TRITON_VPU", "1")
os.environ.setdefault("DEV_B30_ENABLE_VPU_EXTRA_OP", "1")
os.environ.setdefault("DEV_B30_ENABLE_VPU_TRIAL_OP", "1")

import leap_llm.models.pi0.model_siglip as model_siglip_module  # noqa: E402
from leap_llm.models.pi0.model_siglip import Siglip  # noqa: E402

from pi0_sdk_precision_patch import (  # noqa: E402
    configure_siglip_export_precision,
    install_pi0_attention_precision_patch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize the S600 Pi0 SigLIP encoder with real SO100 images."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vision-tokens-num", type=int, default=256)
    parser.add_argument("--jobs", type=int, default=20)
    parser.add_argument("--march", default="nash-p")
    parser.add_argument("--max-l2m-size", type=int, default=6 * 1024 * 1024)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--calibration-dtype",
        choices=("float16", "float32"),
        default="float32",
    )
    parser.add_argument("--valid-camera-slots", type=int, default=None)
    parser.add_argument(
        "--patch-embedding-mode", choices=("quant8", "fp16"), default="quant8"
    )
    parser.add_argument(
        "--position-embedding-mode", choices=("quant8", "fp16"), default="quant8"
    )
    parser.add_argument(
        "--attention-linear-mode",
        choices=("dynamic", "fakequant16", "w8a16", "fp16"),
        default="dynamic",
    )
    parser.add_argument(
        "--mlp-linear-mode",
        choices=("dynamic", "fakequant16", "w8a16", "fp16"),
        default="dynamic",
    )
    parser.add_argument(
        "--attention-matmul-mode",
        choices=("dynamic", "fixed8x16", "fixed16", "fp16"),
        default="dynamic",
    )
    parser.add_argument("--attention-matmul-last-layers", type=int, default=None)
    parser.add_argument("--fp16-last-layers", type=int, default=0)
    parser.add_argument(
        "--projector-linear-mode",
        choices=("dynamic", "fakequant16", "w8a16", "fp16"),
        default="dynamic",
    )
    parser.add_argument(
        "--layernorm-mode",
        choices=("standard", "split"),
        default="split",
    )
    parser.add_argument("--calibrate-only", action="store_true")
    return parser.parse_args()


def load_pi0_checkpoint(path: str):
    state_dict = safetensors_load_file(path)
    if state_dict and all(key.startswith("model.") for key in state_dict):
        return {key.removeprefix("model."): value for key, value in state_dict.items()}
    return state_dict


def load_calibration_metadata(calibration_dir: Path) -> dict:
    manifest_path = calibration_dir / "manifest.json"
    if not manifest_path.is_file():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_calibration_layout(
    calibration_dir: Path, valid_camera_slots: int | None
) -> tuple[int, list[str], dict]:
    metadata = load_calibration_metadata(calibration_dir)
    if valid_camera_slots is None:
        valid_camera_slots = int(metadata.get("valid_camera_slots", 3))
    if not 1 <= valid_camera_slots <= 3:
        raise ValueError(
            f"valid_camera_slots must be in [1, 3], got {valid_camera_slots}"
        )
    camera_keys = list(metadata.get("camera_keys", []))[:valid_camera_slots]
    camera_keys.extend(
        f"camera_{index}"
        for index in range(len(camera_keys), valid_camera_slots)
    )
    return valid_camera_slots, camera_keys, metadata


def load_calibration(
    calibration_dir: Path,
    device: str,
    dtype: torch.dtype,
    valid_camera_slots: int,
):
    image_root = calibration_dir / "images"
    folders = sorted(
        (path for path in image_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    samples = []
    for folder in folders:
        images = []
        for image_index in range(valid_camera_slots):
            image_path = folder / f"image_{image_index}.jpg"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            with Image.open(image_path) as image_file:
                image_rgb = np.asarray(image_file.convert("RGB"), dtype=np.float32)
            image = (image_rgb - 127.5) / 127.5
            image = np.transpose(image, (2, 0, 1))[None]
            images.append(torch.from_numpy(image).to(device=device, dtype=dtype))
        samples.append((folder.name, images))
    return samples


def tensor_stats(tensor: torch.Tensor) -> dict:
    value = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": float(value.mean().item()),
        "std": float(value.std().item()),
        "absmax": float(value.abs().max().item()),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    args = parse_args()
    precision_patch = install_pi0_attention_precision_patch(include_leap_build=True)
    model_dir = Path(args.model_dir).resolve()
    calibration_dir = Path(args.calibration_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_hbm = output_dir / "pi0_siglip_ptq.hbm"
    if output_hbm.exists():
        raise FileExistsError(f"Refusing to overwrite existing model: {output_hbm}")
    model_file = model_dir / "model.safetensors"
    if not model_file.is_file():
        raise FileNotFoundError(model_file)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {args.device}")

    calibration_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.calibration_dtype]
    valid_camera_slots, camera_keys, calibration_metadata = (
        resolve_calibration_layout(calibration_dir, args.valid_camera_slots)
    )
    samples = load_calibration(
        calibration_dir,
        args.device,
        calibration_dtype,
        valid_camera_slots,
    )
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError("No calibration samples loaded")

    model_siglip_module.load_file = load_pi0_checkpoint
    print(f"Loading Pi0 checkpoint: {model_file}", flush=True)
    siglip = Siglip.build(str(model_file), args.vision_tokens_num)
    export_precision = configure_siglip_export_precision(
        siglip,
        patch_embedding_mode=args.patch_embedding_mode,
        position_embedding_mode=args.position_embedding_mode,
        attention_linear_mode=args.attention_linear_mode,
        mlp_linear_mode=args.mlp_linear_mode,
        attention_matmul_mode=args.attention_matmul_mode,
        attention_matmul_last_layers=args.attention_matmul_last_layers,
        fp16_last_layers=args.fp16_last_layers,
        projector_linear_mode=args.projector_linear_mode,
        layernorm_mode=args.layernorm_mode,
    )
    siglip.model.to(device=args.device, dtype=calibration_dtype)
    siglip.model.compile_mode(False)
    position_ids = torch.arange(256, device=args.device).view(1, 256)

    started = time.perf_counter()
    calibration_stats = []
    with torch.inference_mode():
        for sample_name, images in samples:
            outputs = [siglip.model.forward(image, position_ids) for image in images]
            for output in outputs:
                expected_shape = (1, args.vision_tokens_num, 2048)
                if tuple(output.shape) != expected_shape:
                    raise RuntimeError(
                        f"Unexpected SigLIP output shape {tuple(output.shape)} != {expected_shape}"
                    )
                if not torch.isfinite(output).all():
                    raise FloatingPointError(f"Non-finite SigLIP output for sample {sample_name}")
            sample_stats = {"sample": sample_name}
            sample_stats.update(
                {
                    camera_key: tensor_stats(output)
                    for camera_key, output in zip(camera_keys, outputs, strict=True)
                }
            )
            calibration_stats.append(sample_stats)
            print(
                f"calibration {len(calibration_stats)}/{len(samples)}: "
                f"{camera_keys[0]}_absmax="
                f"{calibration_stats[-1][camera_keys[0]]['absmax']:.4f}",
                flush=True,
            )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    calibration_seconds = time.perf_counter() - started
    calibration_result = {
        "model_dir": str(model_dir),
        "calibration_dir": str(calibration_dir),
        "sample_count": len(samples),
        "calibration_forward_count": len(samples) * valid_camera_slots,
        "calibration_dtype": args.calibration_dtype,
        "valid_camera_slots": valid_camera_slots,
        "camera_keys": camera_keys,
        "calibration_manifest_format": calibration_metadata.get("format"),
        "vision_tokens_num": args.vision_tokens_num,
        "export_precision": export_precision,
        "device": args.device,
        "calibration_seconds": calibration_seconds,
        "precision_patch": precision_patch,
        "samples": calibration_stats,
    }
    (output_dir / "calibration_forward.json").write_text(
        json.dumps(calibration_result, indent=2) + chr(10), encoding="utf-8"
    )
    if args.calibrate_only:
        print("Calibration-only run completed.", flush=True)
        return

    siglip.model.compile_mode(True)
    siglip.model.to(device="cpu", dtype=torch.float16)
    compile_kwargs = {
        "march": args.march,
        "jobs": args.jobs,
        "progress_bar": True,
        "max_time_per_fc": 0.0,
        "opt": 2,
        "debug": False,
        "advice": 0.0,
        "balance": 100,
        "input_no_padding": True,
        "output_no_padding": True,
        "enable_hpc": True,
        "core_num": 1,
        "max_l2m_size": args.max_l2m_size,
    }
    print(f"Compiling SigLIP to: {output_hbm}", flush=True)
    compile_started = time.perf_counter()
    siglip.compile(output_model_path=str(output_hbm), **compile_kwargs)
    compile_seconds = time.perf_counter() - compile_started

    artifact_paths = sorted(output_dir.glob("pi0_siglip_ptq.*"))
    result = {
        **calibration_result,
        "compile_seconds": compile_seconds,
        "compile_kwargs": compile_kwargs,
        "model_sha256": sha256_file(model_file),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "leap_llm": package_version("leap-llm"),
            "hbdk4_compiler": package_version("hbdk4-compiler"),
        },
        "artifacts": {
            path.name: {"size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in artifact_paths
        },
    }
    result_path = output_dir / "quantization_manifest.json"
    result_path.write_text(json.dumps(result, indent=2) + chr(10), encoding="utf-8")
    print(f"Quantization manifest: {result_path}", flush=True)


if __name__ == "__main__":
    main()
