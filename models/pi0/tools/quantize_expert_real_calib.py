#!/usr/bin/env python3

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as safetensors_load_file
from transformers import AutoTokenizer

os.environ.setdefault("DEV_B30_TRITON_VPU", "1")
os.environ.setdefault("DEV_B30_ENABLE_VPU_EXTRA_OP", "1")
os.environ.setdefault("DEV_B30_ENABLE_VPU_TRIAL_OP", "1")

import leap_llm.models.pi0.model_gemma as model_gemma_module  # noqa: E402
import leap_llm.models.pi0.model_gemma_expert as model_gemma_expert_module  # noqa: E402
import leap_llm.models.pi0.model_siglip as model_siglip_module  # noqa: E402
from leap_llm.models.pi0.model_gemma import LanguageModel  # noqa: E402
from leap_llm.models.pi0.model_gemma_expert import GemmaExpertModel  # noqa: E402
from leap_llm.models.pi0.model_siglip import Siglip  # noqa: E402

from pi0_position_layout import (  # noqa: E402
    VALID_VISION_TOKEN_COUNT,
    apply_compact_paligemma_position_layout,
    apply_paligemma_position_layout,
    build_compact_expert_attention_mask,
    build_compact_paligemma_attention_mask,
    build_expert_attention_mask,
    build_expert_position_ids,
    build_paligemma_attention_mask,
    compact_prefix_token_count,
    position_layout_manifest,
)
from pi0_sdk_precision_patch import (  # noqa: E402
    configure_expert_export_precision,
    install_pi0_attention_precision_patch,
)


def load_pi0_checkpoint(path: str):
    state_dict = safetensors_load_file(path)
    if state_dict and all(key.startswith("model.") for key in state_dict):
        return {key.removeprefix("model."): value for key, value in state_dict.items()}
    return state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize only the S600 pi0 action expert with real SO100 data."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument(
        "--paligemma-kv-dir",
        default=None,
        help=(
            "Load precomputed FP16 PaliGemma HBM KV tensors from "
            "<dir>/<sample>/expert_kv_00_fp16.bin through expert_kv_35_fp16.bin."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vision-tokens-num", type=int, default=256)
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument("--core-num", type=int)
    parser.add_argument("--max-l2m-size", type=int)
    parser.add_argument("--march", default="nash-p")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--transformer-linear-mode",
        choices=("dynamic", "w8a16", "fp16"),
        default="dynamic",
    )
    parser.add_argument(
        "--projection-linear-mode",
        choices=("dynamic", "w8a16", "fp16"),
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
        "--output-linear-mode",
        choices=("dynamic", "w8a16", "fp16"),
        default="dynamic",
    )
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--compact-prefix", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_calibration(
    calibration_dir: Path,
    device: str,
    dtype: torch.dtype,
    camera_count: int,
    load_images: bool,
):
    image_root = calibration_dir / "images"
    action_root = calibration_dir / "action"
    text_path = calibration_dir / "text" / "calibration.json"
    prompts = json.loads(text_path.read_text(encoding="utf-8"))
    folders = sorted(
        (path for path in image_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    if len(folders) != len(prompts):
        raise RuntimeError(
            f"Image folder count {len(folders)} does not match prompt count {len(prompts)}"
        )

    samples = []
    for folder, prompt in zip(folders, prompts, strict=True):
        images = []
        if load_images:
            for image_index in range(camera_count):
                image_path = folder / f"image_{image_index}.jpg"
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                with Image.open(image_path) as image_file:
                    image_rgb = np.asarray(image_file.convert("RGB"), dtype=np.float32)
                image = (image_rgb - 127.5) / 127.5
                image = np.transpose(image, (2, 0, 1))[None]
                images.append(torch.from_numpy(image).to(device=device, dtype=dtype))

        action_path = action_root / folder.name
        state = np.load(action_path / "state.npy")
        x_t = np.load(action_path / "x_t.npy")
        if state.shape != (32,):
            raise ValueError(f"Unexpected state shape in {action_path}: {state.shape}")
        if x_t.shape != (50, 32):
            raise ValueError(f"Unexpected x_t shape in {action_path}: {x_t.shape}")
        samples.append((folder.name, images, prompt["text"], state, x_t))
    return samples


def load_precomputed_kv(
    kv_root: Path,
    sample_name: str,
    prefix_token_length: int,
    device: str,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    sample_dir = kv_root / sample_name
    if not sample_dir.is_dir():
        raise NotADirectoryError(sample_dir)
    shape = (1, prefix_token_length, 256)
    expected_values = int(np.prod(shape))
    caches = []
    for index in range(36):
        path = sample_dir / f"expert_kv_{index:02d}_fp16.bin"
        values = np.fromfile(path, dtype=np.float16)
        if values.size != expected_values:
            raise ValueError(
                f"Unexpected Expert KV size in {path}: "
                f"{values.size}, expected {expected_values}"
            )
        caches.append(
            torch.from_numpy(values.reshape(shape).copy()).to(
                device=device,
                dtype=dtype,
            )
        )
    return caches


def tokenize(tokenizer, prompt: str, max_length: int = 48) -> tuple[np.ndarray, int]:
    text = prompt if prompt.endswith("\n") else f"{prompt}\n"
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="np",
    )
    tokens = encoded["input_ids"][0].astype(np.int32)
    valid_length = int(encoded["attention_mask"][0].sum())
    return tokens, valid_length


def compile_expert(
    expert: GemmaExpertModel,
    output_hbm: Path,
    token_length: int,
    compile_kwargs: dict,
) -> dict:
    input_types = expert.get_leap_input_types(32, 50, token_length)
    bc_path = output_hbm.with_suffix(".bc")
    bc_module = expert.model.export_module(
        input_types,
        "gemma_expert",
        str(bc_path),
    )
    converted_path = output_hbm.with_suffix(".convert.bc")
    mlir_module = expert.model.convert_mlir(
        bc_module,
        save_path=str(converted_path),
        march=compile_kwargs["march"],
        dynamic_quant=True,
    )
    hbo_path = output_hbm.with_suffix(".hbo")
    hbo_model = expert.model.compile_hbo(
        mlir_module,
        str(hbo_path),
        **compile_kwargs,
    )
    expert.model.link_models([hbo_model], str(output_hbm))
    return compile_kwargs


def main() -> None:
    args = parse_args()
    if args.paligemma_kv_dir is not None and args.compact_prefix:
        raise ValueError("--paligemma-kv-dir does not support --compact-prefix")
    precision_patch = install_pi0_attention_precision_patch(include_leap_build=True)
    model_dir = Path(args.model_dir).resolve()
    calibration_dir = Path(args.calibration_dir).resolve()
    paligemma_kv_dir = (
        Path(args.paligemma_kv_dir).resolve()
        if args.paligemma_kv_dir is not None
        else None
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_hbm = output_dir / "pi0_gemma_expert_ptq.hbm"
    if output_hbm.exists():
        raise FileExistsError(f"Refusing to overwrite existing model: {output_hbm}")

    model_file = model_dir / "model.safetensors"
    if not model_file.is_file():
        raise FileNotFoundError(model_file)
    if paligemma_kv_dir is not None and not paligemma_kv_dir.is_dir():
        raise NotADirectoryError(paligemma_kv_dir)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {args.device}")

    dtype = torch.float16
    samples = load_calibration(
        calibration_dir,
        args.device,
        dtype,
        camera_count=1 if args.compact_prefix else 3,
        load_images=paligemma_kv_dir is None,
    )
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError("No calibration samples loaded")

    model_siglip_module.load_file = load_pi0_checkpoint
    model_gemma_module.load_file = load_pi0_checkpoint
    model_gemma_expert_module.load_file = load_pi0_checkpoint
    print(f"Loading pi0 checkpoint: {model_file}", flush=True)
    siglip = None
    paligemma = None
    if paligemma_kv_dir is None:
        siglip = Siglip.build(str(model_file), args.vision_tokens_num)
        paligemma = LanguageModel.build(str(model_file), args.vision_tokens_num)
    expert = GemmaExpertModel.build(str(model_file), args.vision_tokens_num)
    export_precision = configure_expert_export_precision(
        expert,
        transformer_linear_mode=args.transformer_linear_mode,
        projection_linear_mode=args.projection_linear_mode,
        attention_matmul_mode=args.attention_matmul_mode,
        attention_matmul_last_layers=args.attention_matmul_last_layers,
        fp16_last_layers=args.fp16_last_layers,
        output_linear_mode=args.output_linear_mode,
    )

    if siglip is not None:
        siglip.model.to(device=args.device, dtype=dtype)
    if paligemma is not None:
        paligemma.model.to(device=args.device, dtype=dtype)
    expert.model.to(device=args.device, dtype=dtype)
    if siglip is not None:
        siglip.model.compile_mode(False)
    if paligemma is not None:
        paligemma.model.compile_mode(False)
    expert.model.compile_mode(False)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer.padding_side = "right"
    _, position_valid_token_length = tokenize(tokenizer, samples[0][2])
    if args.compact_prefix:
        position_layout = apply_compact_paligemma_position_layout(
            paligemma.model, position_valid_token_length
        )
        prefix_token_length = compact_prefix_token_count(position_valid_token_length)
    elif paligemma is not None:
        position_layout = apply_paligemma_position_layout(
            paligemma.model, position_valid_token_length
        )
        prefix_token_length = args.vision_tokens_num * 3 + 48
    else:
        position_layout = position_layout_manifest(position_valid_token_length)
        prefix_token_length = args.vision_tokens_num * 3 + 48
    siglip_position_ids = torch.arange(0, 256).view(1, 256).to(args.device)
    calibration_stats = []
    started = time.perf_counter()

    with torch.inference_mode():
        for sample_index, (
            sample_name,
            images,
            prompt,
            state_array,
            x_t_array,
        ) in enumerate(samples):
            token_array, valid_token_length = tokenize(tokenizer, prompt)
            if valid_token_length != position_valid_token_length:
                raise RuntimeError(
                    "All calibration prompts must have the same valid token length "
                    "for the baked PaliGemma RoPE layout"
                )
            if paligemma_kv_dir is not None:
                vision_token_length = args.vision_tokens_num * 3
                kv_cache = load_precomputed_kv(
                    paligemma_kv_dir,
                    sample_name,
                    prefix_token_length,
                    args.device,
                    dtype,
                )
            else:
                if args.compact_prefix:
                    token_array = token_array[:valid_token_length].copy()
                siglip_outputs = [
                    siglip.model.forward(image, siglip_position_ids)
                    for image in images
                ]
                inputs_embeds = torch.cat(siglip_outputs, dim=1)
                vision_token_length = inputs_embeds.shape[1]
                expected_vision_token_length = (
                    VALID_VISION_TOKEN_COUNT
                    if args.compact_prefix
                    else args.vision_tokens_num * 3
                )
                if vision_token_length != expected_vision_token_length:
                    raise RuntimeError(
                        f"Expected {expected_vision_token_length} vision tokens, "
                        f"got {vision_token_length}"
                    )
                tokens = torch.from_numpy(token_array).unsqueeze(0).to(args.device)
                if args.compact_prefix:
                    gemma_mask = build_compact_paligemma_attention_mask(
                        valid_token_length,
                        device=args.device,
                        dtype=dtype,
                    )
                else:
                    gemma_mask = build_paligemma_attention_mask(
                        valid_token_length,
                        device=args.device,
                        dtype=dtype,
                    )
                gemma_outputs = paligemma.model.forward(
                    tokens=tokens,
                    inputs_embeds=inputs_embeds,
                    attention_mask=gemma_mask,
                )
                kv_cache = gemma_outputs[1:]
            if kv_cache[0].shape[1] != prefix_token_length:
                raise RuntimeError(
                    f"Expected KV prefix length {prefix_token_length}, "
                    f"got {kv_cache[0].shape[1]}"
                )
            if args.compact_prefix:
                action_mask = build_compact_expert_attention_mask(
                    valid_token_length,
                    device=args.device,
                    dtype=dtype,
                )
            else:
                action_mask = build_expert_attention_mask(
                    valid_token_length,
                    device=args.device,
                    dtype=dtype,
                )
            position_ids = build_expert_position_ids(
                valid_token_length,
                device=args.device,
            )
            state = torch.from_numpy(state_array).unsqueeze(0).to(args.device, dtype=dtype)
            x_t = torch.from_numpy(x_t_array).unsqueeze(0).to(args.device, dtype=dtype)

            initial_absmax = float(x_t.abs().max().item())
            for denoise_step in range(10):
                denoise_index = torch.tensor([denoise_step], dtype=torch.int32)
                x_t = expert.model.forward(
                    state=state,
                    x_t=x_t,
                    denoise_idx=denoise_index,
                    attention_mask=action_mask,
                    position_ids=position_ids,
                    caches=kv_cache,
                )
            if not torch.isfinite(x_t).all():
                raise FloatingPointError(f"Non-finite expert output at sample {sample_index}")
            calibration_stats.append(
                {
                    "sample": sample_name,
                    "prompt_tokens": valid_token_length,
                    "vision_tokens": int(vision_token_length),
                    "state_absmax": float(state.abs().max().item()),
                    "initial_x_t_absmax": initial_absmax,
                    "final_x_t_absmax": float(x_t.abs().max().item()),
                    "final_x_t_mean": float(x_t.float().mean().item()),
                    "final_x_t_std": float(x_t.float().std().item()),
                }
            )
            print(
                f"calibration {sample_index + 1}/{len(samples)}: "
                f"state_absmax={calibration_stats[-1]['state_absmax']:.4f} "
                f"final_absmax={calibration_stats[-1]['final_x_t_absmax']:.4f}",
                flush=True,
            )

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    calibration_seconds = time.perf_counter() - started
    calibration_result = {
        "model_dir": str(model_dir),
        "calibration_dir": str(calibration_dir),
        "paligemma_kv_dir": (
            str(paligemma_kv_dir) if paligemma_kv_dir is not None else None
        ),
        "paligemma_kv_source": (
            "precomputed_s600_paligemma_hbm"
            if paligemma_kv_dir is not None
            else "float_paligemma"
        ),
        "sample_count": len(samples),
        "vision_tokens_num": args.vision_tokens_num,
        "prefix_token_length": prefix_token_length,
        "compact_prefix": args.compact_prefix,
        "device": args.device,
        "calibration_seconds": calibration_seconds,
        "position_layout": position_layout,
        "export_precision": export_precision,
        "precision_patch": precision_patch,
        "samples": calibration_stats,
    }
    (output_dir / "calibration_forward.json").write_text(
        json.dumps(calibration_result, indent=2) + "\n", encoding="utf-8"
    )

    if args.calibrate_only:
        print("Calibration-only run completed.", flush=True)
        return

    if siglip is not None:
        del siglip
    if paligemma is not None:
        del paligemma
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    expert.model.compile_mode(True)
    expert.model.to(device="cpu", dtype=dtype)
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
    }
    if args.core_num is not None:
        compile_kwargs["core_num"] = args.core_num
    if args.max_l2m_size is not None:
        compile_kwargs["max_l2m_size"] = args.max_l2m_size
    print(f"Compiling expert to: {output_hbm}", flush=True)
    compile_started = time.perf_counter()
    compile_kwargs = compile_expert(
        expert,
        output_hbm,
        prefix_token_length,
        compile_kwargs,
    )
    compile_seconds = time.perf_counter() - compile_started

    artifact_paths = sorted(output_dir.glob("pi0_gemma_expert_ptq.*"))
    result = {
        **calibration_result,
        "compile_seconds": compile_seconds,
        "compile_kwargs": compile_kwargs,
        "model_sha256": sha256_file(model_file),
        "calibration_manifest_sha256": sha256_file(calibration_dir / "manifest.json"),
        "artifacts": {
            path.name: {
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifact_paths
        },
    }
    result_path = output_dir / "quantization_manifest.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Quantization manifest: {result_path}", flush=True)


if __name__ == "__main__":
    main()
