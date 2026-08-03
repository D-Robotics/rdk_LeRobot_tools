#!/usr/bin/env python3

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import time
import types
from pathlib import Path

import numpy as np
import torch
from hbdk4.compiler import convert as hbdk_convert
from hbdk4.compiler import leap, save as hbdk_save
from hbdk4.compiler.extra_apis import llm_convert
from PIL import Image
from safetensors.torch import load_file as safetensors_load_file
from transformers import AutoTokenizer

os.environ.setdefault("DEV_B30_TRITON_VPU", "1")
os.environ.setdefault("DEV_B30_ENABLE_VPU_EXTRA_OP", "1")
os.environ.setdefault("DEV_B30_ENABLE_VPU_TRIAL_OP", "1")

import leap_llm.models.pi0.model_gemma as model_gemma_module  # noqa: E402
import leap_llm.models.pi0.model_siglip as model_siglip_module  # noqa: E402
from leap_llm.models.pi0.model_gemma import LanguageModel  # noqa: E402
from leap_llm.models.pi0.model_siglip import Siglip  # noqa: E402

from pi0_position_layout import (  # noqa: E402
    LANGUAGE_TOKEN_CAPACITY,
    PHYSICAL_VISION_TOKEN_COUNT,
    VALID_VISION_TOKEN_COUNT,
    apply_compact_paligemma_position_layout,
    apply_paligemma_position_layout,
    build_compact_paligemma_attention_mask,
    build_paligemma_attention_mask,
)
from pi0_sdk_precision_patch import (  # noqa: E402
    configure_paligemma_export_precision,
    finalize_static_linear_calibration,
    install_pi0_attention_precision_patch,
)


def load_pi0_checkpoint(path: str):
    state_dict = safetensors_load_file(path)
    if state_dict and all(key.startswith("model.") for key in state_dict):
        return {key.removeprefix("model."): value for key, value in state_dict.items()}
    return state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile only the S600 pi0 PaliGemma model with real three-slot SO100 calibration data."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument(
        "--vision-embeddings-dir",
        default=None,
        help=(
            "Load precomputed FP16 PaliGemma vision embeddings from "
            "<dir>/<sample>/paligemma_inputs_embeds.bin instead of running float SigLIP."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vision-tokens-num", type=int, default=256)
    parser.add_argument("--jobs", type=int, default=20)
    parser.add_argument("--compile-opt", type=int, default=2)
    parser.add_argument("--compile-balance", type=int, default=2)
    parser.add_argument("--compile-max-l2m-size", type=int, default=0)
    parser.add_argument("--max-hbm-bytes", type=int, default=2_140_000_000)
    parser.add_argument("--march", default="nash-p")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-hidden-layers", type=int, default=18)
    parser.add_argument(
        "--attention-linear-mode",
        choices=(
            "dynamic",
            "block32",
            "block64",
            "block128",
            "block256",
            "block512",
            "block1024",
            "block512u",
            "block1024u",
            "block256qt",
            "block512qt",
            "block1024qt",
            "block2048",
            "block4096",
            "block8192",
            "w8a16",
            "static_w8a16",
            "fp16",
        ),
        default="dynamic",
    )
    parser.add_argument(
        "--attention-linear-last-layer-count",
        type=int,
        default=None,
        help=(
            "Apply --attention-linear-mode only to the last N transformer "
            "layers. Earlier layers remain dynamic."
        ),
    )
    parser.add_argument(
        "--mlp-linear-mode",
        choices=(
            "dynamic",
            "block32",
            "block64",
            "block128",
            "block256",
            "block512",
            "block1024",
            "block512u",
            "block1024u",
            "block256qt",
            "block512qt",
            "block1024qt",
            "block2048",
            "block4096",
            "block8192",
            "w8a16",
            "static_w8a16",
            "fp16",
        ),
        default="dynamic",
    )
    parser.add_argument(
        "--mlp-down-linear-mode",
        choices=(
            "dynamic",
            "block32",
            "block64",
            "block128",
            "block256",
            "block512",
            "block1024",
            "block512u",
            "block1024u",
            "block256qt",
            "block512qt",
            "block1024qt",
            "block2048",
            "block4096",
            "block8192",
            "block2048qt",
            "block4096qt",
            "block8192qt",
            "w8a16",
            "static_w8a16",
            "fp16",
        ),
        default=None,
        help="Override only the wide MLP down projection precision mode.",
    )
    parser.add_argument(
        "--mlp-down-linear-layer-count",
        type=int,
        default=None,
        help=(
            "Apply --mlp-down-linear-mode only to the first N transformer "
            "layers. Remaining layers use --mlp-linear-mode."
        ),
    )
    parser.add_argument(
        "--attention-matmul-mode",
        choices=("dynamic", "fixed8x16", "fixed16", "fp16"),
        default="dynamic",
    )
    parser.add_argument("--fp16-last-layers", type=int, default=0)
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument(
        "--fixed-prompt",
        default=None,
        help=(
            "Freeze one task prompt into the PaliGemma graph so the exported HBM "
            "accepts only vision embeddings and the attention mask."
        ),
    )
    parser.add_argument(
        "--prompt-embedding-input",
        action="store_true",
        help=(
            "Expose the fixed prompt embedding as an FP16 HBM input instead of "
            "baking it into the graph. Requires --fixed-prompt."
        ),
    )
    parser.add_argument("--compact-prefix", action="store_true")
    return parser.parse_args()


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


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def update_run_state(path: Path, stage: str, **extra) -> None:
    value = {
        "stage": stage,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **extra,
    }
    write_json(path, value)
    print(f"RUN_STAGE={stage}", flush=True)


def load_calibration(
    calibration_dir: Path,
    device: str,
    dtype: torch.dtype,
    camera_count: int,
):
    image_root = calibration_dir / "images"
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
        for image_index in range(camera_count):
            image_path = folder / f"image_{image_index}.jpg"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            with Image.open(image_path) as image_file:
                image_rgb = np.asarray(image_file.convert("RGB"), dtype=np.float32)
            image = (image_rgb - 127.5) / 127.5
            image = np.transpose(image, (2, 0, 1))[None]
            images.append(torch.from_numpy(image).to(device=device, dtype=dtype))
        samples.append((folder.name, images, prompt["text"]))
    return samples


def load_vision_embeddings(
    vision_embeddings_dir: Path,
    sample_name: str,
    device: str,
    dtype: torch.dtype,
    vision_token_count: int,
    camera_index: int = 0,
) -> torch.Tensor:
    path = vision_embeddings_dir / sample_name / "paligemma_inputs_embeds.bin"
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.fromfile(path, dtype=np.float16)
    source_token_counts = [vision_token_count]
    if vision_token_count == VALID_VISION_TOKEN_COUNT:
        source_token_counts.append(PHYSICAL_VISION_TOKEN_COUNT)
    source_token_count = next(
        (
            token_count
            for token_count in source_token_counts
            if values.size == token_count * 2048
        ),
        None,
    )
    if source_token_count is None:
        raise ValueError(
            f"Unexpected vision embedding size for {sample_name}: "
            f"{values.size}, expected one of "
            f"{[token_count * 2048 for token_count in source_token_counts]}"
        )
    values = values.reshape(1, source_token_count, 2048)
    if source_token_count != vision_token_count:
        camera_start = camera_index * VALID_VISION_TOKEN_COUNT
        camera_end = camera_start + vision_token_count
        if camera_end > source_token_count:
            raise ValueError(
                f"Camera index {camera_index} is unavailable in {path} with "
                f"{source_token_count} tokens"
            )
        values = values[:, camera_start:camera_end]
    elif camera_index != 0:
        raise ValueError(
            f"Camera index {camera_index} is unavailable in compact embedding {path}"
        )
    return torch.from_numpy(values.copy()).to(device=device, dtype=dtype)


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


def fixed_prompt_build(self, inputs_embeds, attention_mask):
    hidden_states = leap.concat([inputs_embeds, self.fixed_lang_emb], dim=1)
    new_keys = []
    new_values = []
    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        layer_outputs, new_k, new_v = decoder_layer(
            hidden_states,
            cos=self.cos,
            sin=self.sin,
            attention_mask=attention_mask,
        )
        hidden_states = layer_outputs
        new_keys.append(new_k)
        new_values.append(new_v)
    hidden_states = self.norm(hidden_states)
    return hidden_states, *new_keys, *new_values


def fixed_prompt_forward(self, inputs_embeds, attention_mask):
    hidden_states = torch.concat([inputs_embeds, self.fixed_lang_emb], dim=1)
    new_keys = []
    new_values = []
    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        layer_outputs, new_k, new_v = decoder_layer(
            hidden_states,
            cos=self.cos.to(device=hidden_states.device),
            sin=self.sin.to(device=hidden_states.device),
            attention_mask=attention_mask,
        )
        hidden_states = layer_outputs
        new_keys.append(new_k)
        new_values.append(new_v)
    hidden_states = self.norm(hidden_states)
    return hidden_states, *new_keys, *new_values


def prompt_embedding_input_build(
    self, prompt_embeddings, inputs_embeds, attention_mask
):
    hidden_states = leap.concat([inputs_embeds, prompt_embeddings], dim=1)
    new_keys = []
    new_values = []
    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        layer_outputs, new_k, new_v = decoder_layer(
            hidden_states,
            cos=self.cos,
            sin=self.sin,
            attention_mask=attention_mask,
        )
        hidden_states = layer_outputs
        new_keys.append(new_k)
        new_values.append(new_v)
    hidden_states = self.norm(hidden_states)
    return hidden_states, *new_keys, *new_values


def prompt_embedding_input_forward(
    self, prompt_embeddings, inputs_embeds, attention_mask
):
    hidden_states = torch.concat([inputs_embeds, prompt_embeddings], dim=1)
    new_keys = []
    new_values = []
    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        layer_outputs, new_k, new_v = decoder_layer(
            hidden_states,
            cos=self.cos.to(device=hidden_states.device),
            sin=self.sin.to(device=hidden_states.device),
            attention_mask=attention_mask,
        )
        hidden_states = layer_outputs
        new_keys.append(new_k)
        new_values.append(new_v)
    hidden_states = self.norm(hidden_states)
    return hidden_states, *new_keys, *new_values


def freeze_prompt_embedding(model, tokens: torch.Tensor) -> torch.Tensor:
    fixed_lang_emb = model.embed_tokens(tokens) * model.lang_sqrt
    model.register_buffer(
        "fixed_lang_emb",
        fixed_lang_emb.detach().clone(),
        persistent=False,
    )
    model.embed_tokens = None
    model.build = types.MethodType(fixed_prompt_build, model)
    model.forward = types.MethodType(fixed_prompt_forward, model)
    model._original_forward = model.forward
    return fixed_lang_emb


def expose_prompt_embedding_input(model, tokens: torch.Tensor) -> torch.Tensor:
    fixed_lang_emb = model.embed_tokens(tokens) * model.lang_sqrt
    model.embed_tokens = None
    model.build = types.MethodType(prompt_embedding_input_build, model)
    model.forward = types.MethodType(prompt_embedding_input_forward, model)
    model._original_forward = model.forward
    return fixed_lang_emb.detach().clone()


def fixed_prompt_input_types(
    vision_token_count: int,
    prompt_token_count: int,
) -> list[leap.TensorType]:
    total_token_count = vision_token_count + prompt_token_count
    return [
        leap.TensorType([1, vision_token_count, 2048], leap.float16),
        leap.TensorType(
            [1, 1, total_token_count, total_token_count],
            leap.float16,
        ),
    ]


def prompt_embedding_input_types(
    vision_token_count: int,
    prompt_token_count: int,
) -> list[leap.TensorType]:
    total_token_count = vision_token_count + prompt_token_count
    return [
        leap.TensorType([1, prompt_token_count, 2048], leap.float16),
        leap.TensorType([1, vision_token_count, 2048], leap.float16),
        leap.TensorType(
            [1, 1, total_token_count, total_token_count],
            leap.float16,
        ),
    ]


def graph_summary(
    module,
    expected_input_count: int = 3,
    expected_output_count: int = 37,
) -> dict:
    functions = list(module.functions)
    if len(functions) != 1:
        raise RuntimeError(f"Expected one graph function, got {len(functions)}")
    function = functions[0]
    inputs = []
    for index, argument in enumerate(function.inputs):
        uses = len(list(argument.value.uses))
        removable = argument.is_removable
        inputs.append(
            {
                "index": index,
                "name": argument.name,
                "type": str(argument.type),
                "uses": uses,
                "is_removable": bool(removable[0]),
                "removable_reason": str(removable[1]),
            }
        )
    summary = {
        "name": function.name,
        "input_count": len(function.inputs),
        "output_count": len(function.outputs),
        "operation_count": len(function.operations),
        "inputs": inputs,
    }
    if (
        summary["input_count"] != expected_input_count
        or summary["output_count"] != expected_output_count
    ):
        raise RuntimeError(f"Unexpected PaliGemma graph interface: {summary}")
    vision_input = inputs[0] if expected_input_count == 2 else inputs[1]
    if vision_input["uses"] < 1:
        raise RuntimeError(f"Vision embedding input is unused in graph: {vision_input}")
    if expected_input_count == 3 and vision_input["is_removable"]:
        raise RuntimeError(f"Vision embedding input is not dynamic in graph: {vision_input}")
    return summary


def tensor_stats(tensor: torch.Tensor) -> dict:
    value = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "mean": float(value.mean().item()),
        "std": float(value.std().item()),
        "absmax": float(value.abs().max().item()),
    }


def main() -> None:
    args = parse_args()
    if args.prompt_embedding_input and args.fixed_prompt is None:
        raise ValueError("--prompt-embedding-input requires --fixed-prompt")
    if args.compact_prefix and args.fixed_prompt is None:
        raise ValueError("--compact-prefix requires --fixed-prompt")
    precision_patch = install_pi0_attention_precision_patch(include_leap_build=True)
    model_dir = Path(args.model_dir).resolve()
    calibration_dir = Path(args.calibration_dir).resolve()
    vision_embeddings_dir = (
        Path(args.vision_embeddings_dir).resolve()
        if args.vision_embeddings_dir is not None
        else None
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / "pi0_gemma_llm_ptq"
    output_hbm = output_base.with_suffix(".hbm")
    if output_hbm.exists():
        raise FileExistsError(f"Refusing to overwrite existing model: {output_hbm}")

    run_state_path = output_dir / "run_state.json"
    model_file = model_dir / "model.safetensors"
    calibration_manifest = calibration_dir / "manifest.json"
    if not model_file.is_file():
        raise FileNotFoundError(model_file)
    if not calibration_manifest.is_file():
        raise FileNotFoundError(calibration_manifest)
    if vision_embeddings_dir is not None and not vision_embeddings_dir.is_dir():
        raise NotADirectoryError(vision_embeddings_dir)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {args.device}")

    dtype = torch.float16
    update_run_state(run_state_path, "loading_calibration")
    samples = load_calibration(
        calibration_dir,
        args.device,
        dtype,
        camera_count=1 if args.compact_prefix else 3,
    )
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError("No calibration samples loaded")

    model_siglip_module.load_file = load_pi0_checkpoint
    model_gemma_module.load_file = load_pi0_checkpoint
    update_run_state(run_state_path, "loading_models", sample_count=len(samples))
    print(f"Loading pi0 checkpoint: {model_file}", flush=True)
    siglip = (
        Siglip.build(str(model_file), args.vision_tokens_num)
        if vision_embeddings_dir is None
        else None
    )
    paligemma = LanguageModel.build(str(model_file), args.vision_tokens_num)
    available_layers = len(paligemma.model.layers)
    if not 1 <= args.num_hidden_layers <= available_layers:
        raise ValueError(
            f"--num-hidden-layers must be in [1, {available_layers}], "
            f"got {args.num_hidden_layers}"
        )
    paligemma.model.config.num_hidden_layers = args.num_hidden_layers
    paligemma.model_args.text_config.num_hidden_layers = args.num_hidden_layers
    expected_output_count = 1 + 2 * args.num_hidden_layers
    export_precision = configure_paligemma_export_precision(
        paligemma,
        attention_linear_mode=args.attention_linear_mode,
        attention_linear_last_layer_count=args.attention_linear_last_layer_count,
        mlp_linear_mode=args.mlp_linear_mode,
        mlp_down_linear_mode=args.mlp_down_linear_mode,
        mlp_down_linear_layer_count=args.mlp_down_linear_layer_count,
        attention_matmul_mode=args.attention_matmul_mode,
        fp16_last_layers=args.fp16_last_layers,
    )

    if siglip is not None:
        siglip.model.to(device=args.device, dtype=dtype)
    paligemma.model.to(device=args.device, dtype=dtype)
    if siglip is not None:
        siglip.model.compile_mode(False)
    paligemma.model.compile_mode(False)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer.padding_side = "right"
    fixed_token_array = None
    fixed_valid_token_length = None
    fixed_prompt_result = None
    fixed_embedding = None
    if args.fixed_prompt is not None:
        fixed_token_array, fixed_valid_token_length = tokenize(
            tokenizer,
            args.fixed_prompt,
        )
        if args.compact_prefix:
            fixed_token_array = fixed_token_array[:fixed_valid_token_length].copy()
        for sample_name, _images, prompt in samples:
            sample_tokens, sample_valid_length = tokenize(tokenizer, prompt)
            if args.compact_prefix:
                sample_tokens = sample_tokens[:sample_valid_length]
            if (
                sample_valid_length != fixed_valid_token_length
                or not np.array_equal(sample_tokens, fixed_token_array)
            ):
                raise RuntimeError(
                    f"Calibration sample {sample_name} does not match the fixed prompt"
                )
        fixed_tokens = torch.from_numpy(fixed_token_array).unsqueeze(0).to(args.device)
        with torch.inference_mode():
            if args.prompt_embedding_input:
                fixed_embedding = expose_prompt_embedding_input(
                    paligemma.model, fixed_tokens
                )
            else:
                fixed_embedding = freeze_prompt_embedding(paligemma.model, fixed_tokens)
        token_path = output_dir / "fixed_prompt_tokens.bin"
        embedding_path = output_dir / "fixed_prompt_embedding.bin"
        fixed_token_array.tofile(token_path)
        fixed_embedding.detach().cpu().numpy().astype(np.float16).tofile(embedding_path)
        fixed_prompt_result = {
            "text": args.fixed_prompt,
            "valid_token_length": fixed_valid_token_length,
            "token_ids": fixed_token_array.tolist(),
            "tokens_path": str(token_path),
            "tokens_sha256": sha256_file(token_path),
            "embedding_path": str(embedding_path),
            "embedding_sha256": sha256_file(embedding_path),
            "embedding": tensor_stats(fixed_embedding),
            "hbm_input_order": (
                ["prompt_embeddings", "vision_embeddings", "attention_mask"]
                if args.prompt_embedding_input
                else ["vision_embeddings", "attention_mask"]
            ),
            "prompt_embedding_input": args.prompt_embedding_input,
            "compact_prefix": args.compact_prefix,
        }
        write_json(output_dir / "fixed_prompt.json", fixed_prompt_result)
    if fixed_valid_token_length is None:
        _, position_valid_token_length = tokenize(tokenizer, samples[0][2])
    else:
        position_valid_token_length = fixed_valid_token_length
    if args.compact_prefix:
        position_layout = apply_compact_paligemma_position_layout(
            paligemma.model, position_valid_token_length
        )
    else:
        position_layout = apply_paligemma_position_layout(
            paligemma.model, position_valid_token_length
        )
    if fixed_prompt_result is not None:
        fixed_prompt_result["position_layout"] = position_layout
        write_json(output_dir / "fixed_prompt.json", fixed_prompt_result)
    vision_input_token_count = (
        VALID_VISION_TOKEN_COUNT
        if args.compact_prefix
        else paligemma.model_args.text_config.vision_token_num * 3
    )
    siglip_position_ids = torch.arange(0, 256).view(1, 256).to(args.device)
    calibration_stats = []
    reference_kv = None
    reference_sample = None
    sample_differences = []
    black_probe = None
    started = time.perf_counter()
    update_run_state(run_state_path, "calibrating", sample_count=len(samples))

    with torch.inference_mode():
        for sample_index, (sample_name, images, prompt) in enumerate(samples):
            if fixed_token_array is None:
                token_array, valid_token_length = tokenize(tokenizer, prompt)
            else:
                token_array = fixed_token_array
                valid_token_length = fixed_valid_token_length
            if valid_token_length != position_valid_token_length:
                raise RuntimeError(
                    "All calibration prompts must have the same valid token length "
                    "for the baked PaliGemma RoPE layout"
                )
            if vision_embeddings_dir is None:
                siglip_outputs = [
                    siglip.model.forward(image, siglip_position_ids) for image in images
                ]
                inputs_embeds = torch.cat(siglip_outputs, dim=1)
            else:
                inputs_embeds = load_vision_embeddings(
                    vision_embeddings_dir,
                    sample_name,
                    args.device,
                    dtype,
                    vision_input_token_count,
                )
            vision_token_length = inputs_embeds.shape[1]
            if vision_token_length != vision_input_token_count:
                raise RuntimeError(
                    f"Expected {vision_input_token_count} vision tokens, "
                    f"got {vision_token_length}"
                )
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
            if fixed_token_array is None:
                tokens = torch.from_numpy(token_array).unsqueeze(0).to(args.device)
                outputs = paligemma.model.forward(
                    tokens=tokens,
                    inputs_embeds=inputs_embeds,
                    attention_mask=gemma_mask,
                )
            elif args.prompt_embedding_input:
                outputs = paligemma.model.forward(
                    prompt_embeddings=fixed_embedding,
                    inputs_embeds=inputs_embeds,
                    attention_mask=gemma_mask,
                )
            else:
                outputs = paligemma.model.forward(
                    inputs_embeds=inputs_embeds,
                    attention_mask=gemma_mask,
                )
            if len(outputs) != expected_output_count:
                raise RuntimeError(
                    f"Expected {expected_output_count} PaliGemma outputs, "
                    f"got {len(outputs)}"
                )
            finite_flags = torch.stack([torch.isfinite(output).all() for output in outputs])
            if not bool(finite_flags.all().item()):
                raise FloatingPointError(f"Non-finite PaliGemma output at sample {sample_index}")

            kv0 = outputs[1].detach().float().cpu()
            if reference_kv is None:
                reference_kv = kv0
                reference_sample = sample_name
            else:
                difference = (kv0 - reference_kv).abs()
                sample_differences.append(
                    {
                        "sample": sample_name,
                        "reference_sample": reference_sample,
                        "mean_abs_diff": float(difference.mean().item()),
                        "max_abs_diff": float(difference.max().item()),
                        "identical": bool(torch.equal(kv0, reference_kv)),
                    }
                )

            if sample_index == 0:
                if vision_embeddings_dir is None:
                    black_images = (
                        [torch.full_like(images[0], -1.0)]
                        if args.compact_prefix
                        else [images[1], images[1], images[2]]
                    )
                    black_siglip = [
                        siglip.model.forward(image, siglip_position_ids)
                        for image in black_images
                    ]
                    black_embeds = torch.cat(black_siglip, dim=1)
                elif args.compact_prefix:
                    try:
                        black_embeds = load_vision_embeddings(
                            vision_embeddings_dir,
                            sample_name,
                            args.device,
                            dtype,
                            vision_input_token_count,
                            camera_index=1,
                        )
                    except ValueError:
                        black_embeds = torch.zeros_like(inputs_embeds)
                else:
                    camera_embeddings = torch.split(
                        inputs_embeds,
                        args.vision_tokens_num,
                        dim=1,
                    )
                    if len(camera_embeddings) != 3:
                        raise RuntimeError(
                            f"Expected three camera embeddings, got {len(camera_embeddings)}"
                        )
                    black_embeds = torch.cat(
                        [camera_embeddings[1], camera_embeddings[1], camera_embeddings[2]],
                        dim=1,
                    )
                if fixed_token_array is None:
                    black_outputs = paligemma.model.forward(
                        tokens=tokens,
                        inputs_embeds=black_embeds,
                        attention_mask=gemma_mask,
                    )
                elif args.prompt_embedding_input:
                    black_outputs = paligemma.model.forward(
                        prompt_embeddings=fixed_embedding,
                        inputs_embeds=black_embeds,
                        attention_mask=gemma_mask,
                    )
                else:
                    black_outputs = paligemma.model.forward(
                        inputs_embeds=black_embeds,
                        attention_mask=gemma_mask,
                    )
                black_kv0 = black_outputs[1].detach().float().cpu()
                black_difference = (kv0 - black_kv0).abs()
                black_probe = {
                    "sample": sample_name,
                    "mean_abs_diff": float(black_difference.mean().item()),
                    "max_abs_diff": float(black_difference.max().item()),
                    "identical": bool(torch.equal(kv0, black_kv0)),
                    "live_embedding": tensor_stats(inputs_embeds),
                    "black_embedding": tensor_stats(black_embeds),
                }
                if black_probe["identical"]:
                    raise RuntimeError("Float PaliGemma is insensitive to live versus black images")

            calibration_stats.append(
                {
                    "sample": sample_name,
                    "prompt_tokens": valid_token_length,
                    "vision_tokens": int(vision_token_length),
                    "inputs_embeds": tensor_stats(inputs_embeds),
                    "hidden_state": tensor_stats(outputs[0]),
                    "kv0": tensor_stats(outputs[1]),
                    "kv_last": tensor_stats(outputs[-1]),
                }
            )
            print(
                f"calibration {sample_index + 1}/{len(samples)}: "
                f"embed_absmax={calibration_stats[-1]['inputs_embeds']['absmax']:.4f} "
                f"kv0_absmax={calibration_stats[-1]['kv0']['absmax']:.4f}",
                flush=True,
            )

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    calibration_seconds = time.perf_counter() - started
    static_linear_count = finalize_static_linear_calibration(paligemma.model)
    calibration_result = {
        "model_dir": str(model_dir),
        "calibration_dir": str(calibration_dir),
        "vision_embeddings_dir": (
            str(vision_embeddings_dir) if vision_embeddings_dir is not None else None
        ),
        "vision_embeddings_source": (
            "precomputed_s600_siglip_hbm"
            if vision_embeddings_dir is not None
            else "float_siglip"
        ),
        "sample_count": len(samples),
        "vision_tokens_num": args.vision_tokens_num,
        "vision_input_tokens": vision_input_token_count,
        "compact_prefix": args.compact_prefix,
        "num_hidden_layers": args.num_hidden_layers,
        "static_linear_count": static_linear_count,
        "export_precision": export_precision,
        "device": args.device,
        "calibration_seconds": calibration_seconds,
        "fixed_prompt": fixed_prompt_result,
        "position_layout": position_layout,
        "black_probe": black_probe,
        "sample_differences": sample_differences,
        "samples": calibration_stats,
    }
    write_json(output_dir / "calibration_forward.json", calibration_result)

    if args.calibrate_only:
        update_run_state(run_state_path, "calibration_complete", calibration_seconds=calibration_seconds)
        print("Calibration-only run completed.", flush=True)
        return

    del siglip
    del tokenizer
    samples.clear()
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    paligemma.model.compile_mode(True)
    paligemma.model.to(device="cpu", dtype=dtype)
    if fixed_token_array is not None:
        paligemma.model.forward = paligemma.model.build
    expected_input_count = 3
    if fixed_token_array is None:
        input_types = paligemma.get_leap_input_types(
            paligemma.model_args.text_config.vision_token_num * 3,
            LANGUAGE_TOKEN_CAPACITY,
        )
    elif args.prompt_embedding_input:
        expected_input_count = 3
        input_types = prompt_embedding_input_types(
            vision_input_token_count,
            fixed_valid_token_length if args.compact_prefix else LANGUAGE_TOKEN_CAPACITY,
        )
    else:
        expected_input_count = 2
        input_types = fixed_prompt_input_types(
            vision_input_token_count,
            fixed_valid_token_length if args.compact_prefix else LANGUAGE_TOKEN_CAPACITY,
        )

    stage_times = {}
    update_run_state(run_state_path, "exporting_bc")
    stage_started = time.perf_counter()
    bc_path = output_base.with_suffix(".bc")
    bc_module = paligemma.model.export_module(input_types, "gemma", str(bc_path))
    stage_times["export_seconds"] = time.perf_counter() - stage_started
    bc_graph = graph_summary(
        bc_module,
        expected_input_count=expected_input_count,
        expected_output_count=expected_output_count,
    )
    write_json(output_dir / "bc_graph.json", bc_graph)

    update_run_state(run_state_path, "converting_bc", bc_graph=bc_graph)
    stage_started = time.perf_counter()
    converted_path = output_base.with_suffix(".convert.bc")
    conversion_kwargs = {
        "rmsnorm_version": "triton",
        "softmax_version": "vae",
        "enable_vpu": True,
        "enable_spu": True,
        "use_f16_quant_dequant_on_vae_always": True,
    }
    mlir_module = llm_convert(
        bc_module,
        args.march,
        rmsnorm_version="triton",
        softmax_version="vae",
    )
    mlir_module._use_f16_quant_dequant_on_vae_always = True
    mlir_module = hbdk_convert(
        mlir_module,
        args.march,
        enable_vpu=True,
        enable_spu=True,
    )
    hbdk_save(mlir_module, str(converted_path))
    stage_times["convert_seconds"] = time.perf_counter() - stage_started
    converted_graph = graph_summary(
        mlir_module,
        expected_input_count=expected_input_count,
        expected_output_count=expected_output_count,
    )
    write_json(output_dir / "converted_graph.json", converted_graph)
    del bc_module
    gc.collect()

    compile_kwargs = {
        "march": args.march,
        "jobs": args.jobs,
        "progress_bar": True,
        "max_time_per_fc": 0.0,
        "opt": args.compile_opt,
        "debug": False,
        "advice": 0.0,
        "balance": args.compile_balance,
        "input_no_padding": True,
        "output_no_padding": True,
        "enable_hpc": True,
        "core_num": 1,
        "max_l2m_size": args.compile_max_l2m_size,
    }
    update_run_state(
        run_state_path,
        "compiling_hbo",
        converted_graph=converted_graph,
        compile_kwargs=compile_kwargs,
    )
    stage_started = time.perf_counter()
    hbo_path = output_base.with_suffix(".hbo")
    hbo_model = paligemma.model.compile_hbo(
        mlir_module,
        str(hbo_path),
        **compile_kwargs,
    )
    stage_times["compile_hbo_seconds"] = time.perf_counter() - stage_started
    del mlir_module
    gc.collect()

    update_run_state(run_state_path, "linking_hbm")
    stage_started = time.perf_counter()
    paligemma.model.link_models([hbo_model], str(output_hbm))
    stage_times["link_seconds"] = time.perf_counter() - stage_started
    if output_hbm.stat().st_size > args.max_hbm_bytes:
        update_run_state(
            run_state_path,
            "hbm_too_large",
            hbm=str(output_hbm),
            hbm_size=output_hbm.stat().st_size,
            max_hbm_bytes=args.max_hbm_bytes,
        )
        raise RuntimeError(
            f"HBM size {output_hbm.stat().st_size} exceeds the S600 full-chain "
            f"limit {args.max_hbm_bytes}; reduce block precision coverage or "
            "increase compiler optimization."
        )

    artifact_paths = sorted(output_dir.glob("pi0_gemma_llm_ptq.*"))
    result = {
        **calibration_result,
        "stage_times": stage_times,
        "compile_kwargs": compile_kwargs,
        "conversion_kwargs": conversion_kwargs,
        "precision_patch": precision_patch,
        "bc_graph": bc_graph,
        "converted_graph": converted_graph,
        "model_sha256": sha256_file(model_file),
        "calibration_manifest_sha256": sha256_file(calibration_manifest),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "leap_llm": package_version("leap-llm"),
            "hbdk4_compiler": package_version("hbdk4-compiler"),
        },
        "artifacts": {
            path.name: {
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifact_paths
        },
    }
    manifest_path = output_dir / "quantization_manifest.json"
    write_json(manifest_path, result)
    update_run_state(
        run_state_path,
        "complete",
        manifest=str(manifest_path),
        hbm=str(output_hbm),
        hbm_sha256=result["artifacts"][output_hbm.name]["sha256"],
    )
    print(f"Quantization manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
