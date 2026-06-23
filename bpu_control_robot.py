#!/usr/bin/env python

# Copyright (c) 2025，MaChao D-Robotics.
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deploy an exported ACT policy on RDK BPU, mirroring lerobot-rollout ACT semantics.

Control loop matches ``SyncInferenceEngine.get_action`` + ``ACTPolicy.select_action``:
  - one BPU inference fills an action chunk (default 100 steps)
  - execute one action per control tick from the chunk queue
  - only re-infer when the queue is empty
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import time

import numpy as np
import torch
from torch import Tensor

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

try:
    from hbm_runtime import HB_HBMRuntime
except ImportError:
    import sys as _sys
    _bpu_ext_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpu_runtime", "build")
    if not os.path.exists(_bpu_ext_dir):
        _bpu_ext_dir = "/root/lerobot_act_bpu/build_pybind"
    _sys.path.insert(0, _bpu_ext_dir)
    from bpu_act_runtime import BPUACTRuntime as _BPUACTRuntime

    class HB_HBMRuntime:
        """Drop-in replacement for hbm_runtime.HB_HBMRuntime using C++ pybind11 extension."""

        def __init__(self, model_paths):
            self._rt = _BPUACTRuntime(model_paths)

        def run(self, inputs, model_name=""):
            output = self._rt.run(inputs, model_name=model_name)
            return {model_name: output}

logger = logging.getLogger(__name__)


def parse_max_relative_target(value):
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"none", "null", "off", "-1"}:
            return None
        return float(value)
    return float(value)


def detect_cameras_from_model(bpu_act_path: str) -> list[str]:
    camera_names = []
    for mean_file in glob.glob(os.path.join(bpu_act_path, "*_mean.npy")):
        filename = os.path.basename(mean_file)
        if filename.startswith("action_"):
            continue
        camera_name = filename.replace("_mean.npy", "")
        std_file = os.path.join(bpu_act_path, f"{camera_name}_std.npy")
        if os.path.exists(std_file):
            camera_names.append(camera_name)
    if not camera_names:
        raise ValueError(f"No camera normalization files found under {bpu_act_path}")
    return sorted(camera_names)


def detect_n_action_steps(bpu_act_path: str, override: int | None = None) -> int:
    if override is not None:
        return override
    ref_path = os.path.join(bpu_act_path, "new_actions.npy")
    if os.path.exists(ref_path):
        arr = np.load(ref_path)
        if arr.ndim == 3:
            return int(arr.shape[1])
    return 100


class BPUACTPolicy:
    """ACT policy backed by exported BPU models.

    ``select_action`` intentionally mirrors ``ACTPolicy.select_action`` from lerobot.
    """

    def __init__(self, bpu_act_model_path: str, n_action_steps: int, camera_names: list[str]):
        self.bpu_act_model_path = bpu_act_model_path
        self.n_action_steps = n_action_steps
        self.camera_names = camera_names
        self._action_queue: list[Tensor] = []
        self._inference_count = 0

        self.camera_params: dict[str, dict[str, Tensor]] = {}
        for camera_name in camera_names:
            std_path = os.path.join(bpu_act_model_path, f"{camera_name}_std.npy")
            mean_path = os.path.join(bpu_act_model_path, f"{camera_name}_mean.npy")
            self.camera_params[camera_name] = {
                "std": torch.tensor(np.load(std_path), dtype=torch.float32) + 1e-8,
                "mean": torch.tensor(np.load(mean_path), dtype=torch.float32),
            }

        self.state_mean = torch.tensor(
            np.load(os.path.join(bpu_act_model_path, "action_mean.npy")), dtype=torch.float32
        )
        self.state_std = torch.tensor(
            np.load(os.path.join(bpu_act_model_path, "action_std.npy")), dtype=torch.float32
        ) + 1e-8
        self.action_mean = torch.tensor(
            np.load(os.path.join(bpu_act_model_path, "action_mean_unnormalize.npy")), dtype=torch.float32
        )
        self.action_std = torch.tensor(
            np.load(os.path.join(bpu_act_model_path, "action_std_unnormalize.npy")), dtype=torch.float32
        )

        vision_path = os.path.join(bpu_act_model_path, "BPU_ACTPolicy_VisionEncoder.hbm")
        transformer_path = os.path.join(bpu_act_model_path, "BPU_ACTPolicy_TransformerLayers.hbm")
        self.bpu_policy = HB_HBMRuntime([vision_path, transformer_path])

    def reset(self) -> None:
        self._action_queue.clear()

    @property
    def queue_size(self) -> int:
        return len(self._action_queue)

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        if not self._action_queue:
            self._refill_action_queue(batch)
        return self._action_queue.pop(0)

    def _refill_action_queue(self, batch: dict[str, Tensor]) -> None:
        begin = time.perf_counter()
        batch = self._normalize_inputs(batch)

        state = batch["observation.state"].numpy().copy()
        vision_features = []
        for camera_name in self.camera_names:
            camera_input = batch[f"observation.images.{camera_name}"].numpy().copy()
            vision_output = self.bpu_policy.run(
                {"images": camera_input},
                model_name="BPU_ACTPolicy_VisionEncoder",
            )
            vision_features.append(
                next(iter(vision_output["BPU_ACTPolicy_VisionEncoder"].values()))
            )

        transformer_inputs = {"states": state}
        for camera_name, feature in zip(self.camera_names, vision_features, strict=True):
            transformer_inputs[f"{camera_name}_features"] = feature

        transformer_outputs = self.bpu_policy.run(
            transformer_inputs,
            model_name="BPU_ACTPolicy_TransformerLayers",
        )
        action_output = next(iter(transformer_outputs["BPU_ACTPolicy_TransformerLayers"].values()))
        actions = torch.from_numpy(np.asarray(action_output)).float()
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        if actions.ndim != 3:
            raise RuntimeError(f"Unexpected BPU action shape: {tuple(actions.shape)}")

        actions = actions[:, : self.n_action_steps]
        actions = actions * self.action_std + self.action_mean

        self._action_queue.clear()
        for step_idx in range(actions.shape[1]):
            self._action_queue.append(actions[0, step_idx].clone())

        if len(self._action_queue) != self.n_action_steps:
            raise RuntimeError(
                f"Action queue fill failed: expected {self.n_action_steps}, got {len(self._action_queue)} "
                f"(raw shape={tuple(torch.from_numpy(np.asarray(action_output)).shape)})"
            )

        elapsed_ms = 1000 * (time.perf_counter() - begin)
        self._inference_count += 1
        first = self._action_queue[0]
        last = self._action_queue[-1]
        step_delta = (last - first).abs().mean().item()
        logger.info(
            "BPU inference #%d: %.2f ms, queued=%d, chunk_step_delta_mean=%.4f deg, "
            "action[0][:3]=%s",
            self._inference_count,
            elapsed_ms,
            len(self._action_queue),
            step_delta,
            [round(v, 3) for v in first[:3].tolist()],
        )

    def _normalize_inputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)
        batch["observation.state"] = (batch["observation.state"] - self.state_mean) / self.state_std
        for camera_name in self.camera_names:
            key = f"observation.images.{camera_name}"
            params = self.camera_params[camera_name]
            batch[key] = (batch[key] - params["mean"]) / params["std"]
        return batch


def build_policy_batch(observation: dict, policy: BPUACTPolicy, motor_names: list[str]) -> dict[str, Tensor]:
    state = [observation[f"{motor}.pos"] for motor in motor_names]
    batch: dict[str, Tensor] = {
        "observation.state": torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    }
    for camera_name in policy.camera_names:
        frame = observation[camera_name]
        if not isinstance(frame, torch.Tensor):
            frame = torch.from_numpy(frame)
        batch[f"observation.images.{camera_name}"] = (
            frame.to(dtype=torch.float32).permute(2, 0, 1).contiguous().unsqueeze(0) / 255.0
        )
    return batch


def sanity_check_policy(policy: BPUACTPolicy, camera_names: list[str]) -> None:
    """Verify ACT chunk queue semantics before touching the robot."""
    batch = {
        "observation.state": torch.zeros(1, 6),
        **{
            f"observation.images.{name}": torch.rand(1, 3, 480, 640) / 255.0
            for name in camera_names
        },
    }
    policy.reset()
    inferences = 0
    for tick in range(min(10, policy.n_action_steps + 2)):
        before = policy._inference_count
        policy.select_action(batch)
        if policy._inference_count > before:
            inferences += 1
    expected = 1 if policy.n_action_steps >= 10 else policy.n_action_steps
    if inferences != expected:
        raise RuntimeError(
            f"ACT queue sanity check failed: {inferences} BPU inferences in 10 ticks, "
            f"expected {expected}. Do not pass --n-action-steps 1."
        )
    logger.info("ACT queue sanity check passed (%d inferences / 10 ticks)", inferences)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exported ACT policy on RDK BPU + SO100 follower.")
    parser.add_argument("--bpu-act-path", type=str, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--inference-time", type=int, default=1000, help="Run duration in seconds.")
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Actions executed per BPU inference (default: auto from new_actions.npy, else 100).",
    )
    parser.add_argument("--robot-port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-name", type=str, default="front")
    parser.add_argument(
        "--max-relative-target",
        type=parse_max_relative_target,
        default=None,
        help="Safety clamp in degrees (default: disabled, same as lerobot-rollout).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Log queue size every control tick.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    opt = parse_args()

    camera_names = detect_cameras_from_model(opt.bpu_act_path)
    if opt.camera_name not in camera_names:
        raise ValueError(
            f"--camera-name={opt.camera_name} not found in model cameras {camera_names}"
        )
    logger.info("Detected cameras from model: %s", camera_names)

    n_action_steps = detect_n_action_steps(opt.bpu_act_path, opt.n_action_steps)
    logger.info(
        "n_action_steps=%d (one BPU inference -> %d control ticks before re-infer)",
        n_action_steps,
        n_action_steps,
    )

    policy = BPUACTPolicy(opt.bpu_act_path, n_action_steps, camera_names)
    sanity_check_policy(policy, camera_names)
    policy.reset()

    robot = SO100Follower(
        SO100FollowerConfig(
            port=opt.robot_port,
            id="so100_follower",
            max_relative_target=opt.max_relative_target,
            cameras={
                opt.camera_name: OpenCVCameraConfig(
                    index_or_path=opt.camera_index,
                    width=opt.camera_width,
                    height=opt.camera_height,
                    fps=opt.fps,
                    warmup_s=10,
                    fourcc="MJPG",
                )
            },
        )
    )
    robot.connect()
    motor_names = list(robot.bus.motors.keys())

    total_ticks = opt.inference_time * opt.fps
    logger.info("Starting control loop for %d ticks @ %d fps", total_ticks, opt.fps)

    try:
        for tick in range(total_ticks):
            loop_start = time.perf_counter()

            observation = robot.get_observation()
            batch = build_policy_batch(observation, policy, motor_names)
            action_values = policy.select_action(batch)

            action = {
                f"{motor}.pos": action_values[i].item()
                for i, motor in enumerate(motor_names)
            }
            robot.send_action(action)

            if opt.debug or (tick + 1) % opt.fps == 0:
                logger.info(
                    "tick=%d queue_remaining=%d inference_count=%d action[:3]=%s",
                    tick,
                    policy.queue_size,
                    policy._inference_count,
                    [round(action_values[i].item(), 3) for i in range(min(3, len(action_values)))],
                )

            dt_s = time.perf_counter() - loop_start
            time.sleep(max(0.0, 1 / opt.fps - dt_s))
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
