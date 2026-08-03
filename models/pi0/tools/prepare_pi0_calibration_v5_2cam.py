#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi0.modeling_pi0 import resize_with_pad_torch


TASK_DEFAULT = "Place the RDK camera box on top of the black MCU box."
PHASE_FRACTIONS = (0.05, 0.25, 0.50, 0.75, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build real SO100 calibration inputs for the S600 pi0 expert."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--camera-keys", nargs="+", default=["front"])
    parser.add_argument("--camera-slots", type=int, default=3)
    parser.add_argument("--task", default=TASK_DEFAULT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value) -> int | float:
    if isinstance(value, torch.Tensor):
        return value.item()
    return value


def prepare_rgb_image(image_chw: torch.Tensor) -> np.ndarray:
    image_bhwc = image_chw.float().permute(1, 2, 0).unsqueeze(0)
    resized = resize_with_pad_torch(image_bhwc, 224, 224)[0]
    return (
        resized.clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )


def load_state_stats(checkpoint: Path) -> tuple[np.ndarray, np.ndarray, Path]:
    state_files = sorted(
        checkpoint.glob("policy_preprocessor_step_*_normalizer_processor.safetensors")
    )
    if len(state_files) != 1:
        raise RuntimeError(
            f"Expected one preprocessor normalizer state file, found {state_files}"
        )
    tensors = load_file(str(state_files[0]))
    mean = tensors["observation.state.mean"].cpu().numpy().astype(np.float32)
    std = tensors["observation.state.std"].cpu().numpy().astype(np.float32)
    return mean, std, state_files[0]


def choose_indices(dataset: LeRobotDataset, sample_count: int) -> list[int]:
    episode_values = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
    episodes = np.unique(episode_values)
    episode_positions = np.rint(
        np.linspace(0, len(episodes) - 1, num=sample_count)
    ).astype(np.int64)

    selected = []
    for sample_index, episode_position in enumerate(episode_positions):
        episode = episodes[episode_position]
        episode_rows = np.flatnonzero(episode_values == episode)
        phase = PHASE_FRACTIONS[sample_index % len(PHASE_FRACTIONS)]
        local_index = int(round(phase * (len(episode_rows) - 1)))
        selected.append(int(episode_rows[local_index]))
    return selected


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    repo_id = args.repo_id or f"local/{dataset_root.name}"

    if args.samples < 1:
        raise ValueError("--samples must be positive")
    if not 1 <= len(args.camera_keys) <= args.camera_slots:
        raise ValueError("camera key count must be between 1 and --camera-slots")
    if args.camera_slots != 3:
        raise ValueError("S600 pi0 export currently requires exactly 3 physical camera slots")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")

    images_dir = output_dir / "images"
    text_dir = output_dir / "text"
    action_dir = output_dir / "action"
    images_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    action_dir.mkdir(parents=True, exist_ok=True)

    state_mean, state_std, normalizer_file = load_state_stats(checkpoint)
    dataset = LeRobotDataset(
        repo_id,
        root=dataset_root,
        video_backend=args.video_backend,
    )
    selected_indices = choose_indices(dataset, args.samples)
    empty_rgb = np.zeros((224, 224, 3), dtype=np.uint8)

    prompts = []
    records = []
    normalized_states = []
    noises = []

    for calibration_index, dataset_index in enumerate(selected_indices):
        sample = dataset[dataset_index]
        real_images = [
            prepare_rgb_image(sample[f"observation.images.{camera_key}"])
            for camera_key in args.camera_keys
        ]
        raw_state = sample["observation.state"].cpu().numpy().astype(np.float32)
        raw_action = sample["action"].cpu().numpy().astype(np.float32)
        normalized_state = (raw_state - state_mean[: raw_state.size]) / (
            state_std[: raw_state.size] + 1e-8
        )
        state = np.zeros(32, dtype=np.float32)
        state[: normalized_state.size] = normalized_state

        noise_seed = args.seed + calibration_index
        noise_rng = np.random.default_rng(noise_seed)
        x_t = noise_rng.standard_normal((50, 32)).astype(np.float32)

        sample_image_dir = images_dir / str(calibration_index)
        sample_action_dir = action_dir / str(calibration_index)
        sample_image_dir.mkdir()
        sample_action_dir.mkdir()

        image_paths = [
            sample_image_dir / f"image_{index}.jpg"
            for index in range(args.camera_slots)
        ]
        for image_path, image_rgb in zip(
            image_paths[: len(real_images)], real_images, strict=True
        ):
            Image.fromarray(image_rgb, mode="RGB").save(
                image_path, format="JPEG", quality=95, subsampling=0
            )
        for image_path in image_paths[len(real_images) :]:
            Image.fromarray(empty_rgb, mode="RGB").save(
                image_path, format="JPEG", quality=95, subsampling=0
            )

        np.save(sample_action_dir / "state.npy", state)
        np.save(sample_action_dir / "x_t.npy", x_t)
        np.save(sample_action_dir / "raw_state.npy", raw_state)
        np.save(sample_action_dir / "raw_action.npy", raw_action)

        sample_task = sample.get("task", args.task)
        task = sample_task if sample_task == args.task else args.task
        prompts.append({"text": task})
        normalized_states.append(state)
        noises.append(x_t)
        records.append(
            {
                "calibration_index": calibration_index,
                "dataset_index": dataset_index,
                "episode_index": int(scalar(sample["episode_index"])),
                "frame_index": int(scalar(sample["frame_index"])),
                "timestamp": float(scalar(sample["timestamp"])),
                "task": task,
                "raw_state": raw_state.tolist(),
                "normalized_state": state.tolist(),
                "raw_action": raw_action.tolist(),
                "noise_seed": noise_seed,
                "image_sha256": [sha256_file(path) for path in image_paths],
                "state_sha256": sha256_file(sample_action_dir / "state.npy"),
                "x_t_sha256": sha256_file(sample_action_dir / "x_t.npy"),
            }
        )

    text_path = text_dir / "calibration.json"
    text_path.write_text(json.dumps(prompts, indent=2) + "\n", encoding="utf-8")

    state_stack = np.stack(normalized_states)
    noise_stack = np.stack(noises)
    manifest = {
        "format": "pi0_s600_real_calibration_v2",
        "dataset_root": str(dataset_root),
        "dataset_repo_id": repo_id,
        "checkpoint": str(checkpoint),
        "normalizer_file": str(normalizer_file),
        "task": args.task,
        "sample_count": args.samples,
        "seed": args.seed,
        "sampling": {
            "episode_selection": "evenly spaced unique episode indices",
            "phase_fractions": list(PHASE_FRACTIONS),
        },
        "camera_keys": args.camera_keys,
        "valid_camera_slots": len(args.camera_keys),
        "camera_layout": [
            *[
                f"real {camera_key} camera resized with LeRobot resize_with_pad_torch"
                for camera_key in args.camera_keys
            ],
            *[
                "empty camera encoded as black and masked"
                for _ in range(args.camera_slots - len(args.camera_keys))
            ],
        ],
        "state_preprocessing": "checkpoint MEAN_STD normalization, then zero-pad 6 to 32",
        "noise": "deterministic standard normal N(0,1), shape [50,32]",
        "paths": {
            "images": str(images_dir),
            "text": str(text_path),
            "action": str(action_dir),
        },
        "summary": {
            "normalized_state_min": float(state_stack.min()),
            "normalized_state_max": float(state_stack.max()),
            "noise_mean": float(noise_stack.mean()),
            "noise_std": float(noise_stack.std()),
        },
        "records": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Calibration root: {output_dir}")
    print(f"Images: {images_dir}")
    print(f"Text: {text_path}")
    print(f"Expert action inputs: {action_dir}")
    print(f"Manifest: {manifest_path}")
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
