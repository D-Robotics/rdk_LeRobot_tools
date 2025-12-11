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

# 注意: 此程序在RDK板端运行
# Attention: This program runs on RDK board.

import time
import numpy as np
from copy import copy
import argparse
import os
import glob

import torch
from torch import Tensor
from collections import deque

from lerobot.common.robot_devices.robots.utils import make_robot
from lerobot.common.robot_devices.control_utils import busy_wait

try:
    from hbm_runtime import HB_HBMRuntime
    print("using: hbm_runtime")
except ImportError:
    print("hbm_runtime not found, please check!")
    exit()

def detect_cameras_from_model(bpu_act_path):
    """Detect two camera names from model normalization files."""
    camera_names = []

    # Find all files ending with _mean.npy but not action_mean
    mean_files = glob.glob(os.path.join(bpu_act_path, "*_mean.npy"))

    for mean_file in mean_files:
        filename = os.path.basename(mean_file)
        if filename.startswith("action_"):
            continue  # Skip action-related files

        # Extract camera name (remove _mean.npy suffix)
        camera_name = filename.replace("_mean.npy", "")

        # Check if corresponding std file exists
        std_file = os.path.join(bpu_act_path, f"{camera_name}_std.npy")
        if os.path.exists(std_file):
            camera_names.append(camera_name)

    if len(camera_names) != 2:
        raise ValueError(f"Expected exactly 2 cameras, but found {len(camera_names)}: {camera_names}")

    return sorted(camera_names)  # Sort for consistent ordering

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bpu-act-path', type=str, default='/root/d-lerobot-runtime/outputs/bpu_output', help='Path to LeRobot ACT Policy model.')
    """
    # example: --bpu-act-path pretrained_model
    .
    |-- BPU_ACTPolicy_TransformerLayers.hbm
    |-- BPU_ACTPolicy_VisionEncoder.hbm
    |-- action_mean.npy
    |-- action_mean_unnormalize.npy
    |-- action_std.npy
    |-- action_std_unnormalize.npy
    |-- camera1_mean.npy    # camera names are auto-detected
    |-- camera1_std.npy
    |-- camera2_mean.npy
    `-- camera2_std.npy
    """
    parser.add_argument('--fps', type=int, default=30, help='')
    parser.add_argument('--inference-time', type=int, default=1000, help='seconds')
    parser.add_argument('--n-action-steps', type=int, default=50, help='')
    opt = parser.parse_args()

    # Auto-detect cameras from model files
    camera_names = detect_cameras_from_model(opt.bpu_act_path)
    print(f"Detected cameras from model: {camera_names}")

    robot = make_robot("so101")
    robot.connect()
    policy = RDK_ACTPolicy(opt.bpu_act_path, opt.n_action_steps, camera_names)
    # Copyright 2024 The HuggingFace Inc. team. All rights reserved.
    for _ in range(opt.inference_time * opt.fps):
        start_time = time.perf_counter()
        # Read the follower state and access the frames from the cameras
        observation = robot.capture_observation()
        # Convert to pytorch format: channel first and float32 in [0,1]
        # with batch dimension
        pred_action = predict_action(observation, policy)[0]
        # Remove batch dimension
        action = pred_action.squeeze(0)
        # Move to cpu, if not already the case
        action = action.to("cpu")
        # Order the robot to move
        robot.send_action(action)

        dt_s = time.perf_counter() - start_time
        busy_wait(1 / opt.fps - dt_s)
    robot.disconnect()

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
class RDK_ACTPolicy():
    def __init__(self, bpu_act_model_path, n_action_steps, camera_names):
        self.n_action_steps = n_action_steps
        self._action_queue = deque([], maxlen=self.n_action_steps)
        self.camera_names = camera_names

        print(f"Initializing BPU policy with cameras: {camera_names}")

        # Dynamically load normalization parameters for all cameras
        self.camera_params = {}
        for camera_name in camera_names:
            std_path = os.path.join(bpu_act_model_path, f"{camera_name}_std.npy")
            mean_path = os.path.join(bpu_act_model_path, f"{camera_name}_mean.npy")

            if os.path.exists(std_path) and os.path.exists(mean_path):
                self.camera_params[camera_name] = {
                    'std': torch.tensor(np.load(std_path), dtype=torch.float32) + 1e-8,
                    'mean': torch.tensor(np.load(mean_path), dtype=torch.float32)
                }
                print(f"Loaded normalization params for {camera_name}")
            else:
                raise FileNotFoundError(f"Missing normalization files for camera: {camera_name}")

        # Load action normalization parameters
        action_std_path = os.path.join(bpu_act_model_path, "action_std.npy")
        action_mean_path = os.path.join(bpu_act_model_path, "action_mean.npy")
        action_std_unnormalize_path = os.path.join(bpu_act_model_path, "action_std_unnormalize.npy")
        action_mean_unnormalize_path = os.path.join(bpu_act_model_path, "action_mean_unnormalize.npy")

        # Check all required files
        required_files = [action_std_path, action_mean_path, action_std_unnormalize_path, action_mean_unnormalize_path]
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")

        self.action_std = torch.tensor(np.load(action_std_path), dtype=torch.float32) + 1e-8
        self.action_mean = torch.tensor(np.load(action_mean_path), dtype=torch.float32)
        self.action_std_unnormalize = torch.tensor(np.load(action_std_unnormalize_path), dtype=torch.float32)
        self.action_mean_unnormalize = torch.tensor(np.load(action_mean_unnormalize_path), dtype=torch.float32)

        # Validate parameters
        for camera_name in camera_names:
            params = self.camera_params[camera_name]
            assert not torch.isinf(params['std']).any(), f"Invalid std for {camera_name}"
            assert not torch.isinf(params['mean']).any(), f"Invalid mean for {camera_name}"

        assert not torch.isinf(self.action_std).any(), "Invalid action_std"
        assert not torch.isinf(self.action_mean).any(), "Invalid action_mean"
        assert not torch.isinf(self.action_std_unnormalize).any(), "Invalid action_std_unnormalize"
        assert not torch.isinf(self.action_mean_unnormalize).any(), "Invalid action_mean_unnormalize"

        # Set model paths 
        bpu_act_policy_visionencoder_path = os.path.join(bpu_act_model_path,"BPU_ACTPolicy_VisionEncoder.hbm")
        bpu_act_policy_transformerlayers_path = os.path.join(bpu_act_model_path,"BPU_ACTPolicy_TransformerLayers.hbm")

        if not os.path.exists(bpu_act_policy_visionencoder_path):
            raise FileNotFoundError(f"Vision encoder model not found: {bpu_act_policy_visionencoder_path}")
        if not os.path.exists(bpu_act_policy_transformerlayers_path):
            raise FileNotFoundError(f"Transformer model not found: {bpu_act_policy_transformerlayers_path}")

        # load BPU model using HB_HBMRuntime
        self.bpu_policy = HB_HBMRuntime([
            bpu_act_policy_visionencoder_path,
            bpu_act_policy_transformerlayers_path
        ])
        self.cnt = 0
        print("BPU models loaded successfully")
            

    def bpu_select_action(self, batch: dict[str, Tensor]) -> Tensor:
        # normalize inputs
        batch = self.normalize_inputs(batch)

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            begin_time = time.time()

            # Prepare state input
            state = batch["observation.state"].numpy().copy()

            # Dynamically process all cameras through VisionEncoder
            vision_features = []
            for camera_name in self.camera_names:
                camera_input = batch[f'observation.images.{camera_name}'].numpy().copy()
                # Process through VisionEncoder
                vision_output = self.bpu_policy.run(
                    {"images": camera_input},
                    model_name="BPU_ACTPolicy_VisionEncoder"
                )
                vision_feature = next(iter(vision_output["BPU_ACTPolicy_VisionEncoder"].values()))
                vision_features.append(vision_feature)

            # Build TransformerLayers inputs
            transformer_inputs = {"states": state}
            for i, camera_name in enumerate(self.camera_names):
                transformer_inputs[f"{camera_name}_features"] = vision_features[i]

            # TransformerLayers inference
            transformer_outputs = self.bpu_policy.run(
                transformer_inputs,
                model_name="BPU_ACTPolicy_TransformerLayers"
            )

            # Extract action predictions
            action_output = next(iter(transformer_outputs["BPU_ACTPolicy_TransformerLayers"].values()))
            actions = torch.from_numpy(action_output)[:, :self.n_action_steps]

            print(f"{self.cnt} BPU ACT Policy Time : " + "\033[1;31m" + "%.2f ms"%(1000*(time.time() - begin_time)) + "\033[0m")
            self.cnt += 1
            actions = self.unnormalize_outputs({"action": actions})["action"]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def normalize_inputs(self, batch):
        # Normalize state
        batch["observation.state"] = (batch["observation.state"] - self.action_mean) / self.action_std

        # Dynamically normalize all camera images
        for camera_name in self.camera_names:
            if f'observation.images.{camera_name}' in batch:
                params = self.camera_params[camera_name]
                batch[f'observation.images.{camera_name}'] = (
                    batch[f'observation.images.{camera_name}'] - params['mean']
                ) / params['std']

        return batch
    
    def unnormalize_outputs(self, batch):
        batch["action"] = batch["action"] * self.action_std_unnormalize + self.action_mean_unnormalize
        return batch
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
def predict_action(observation, policy):
    observation = copy(observation)
    for name in observation:
        if "image" in name:
            observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].permute(2, 0, 1).contiguous()
        observation[name] = observation[name].unsqueeze(0)
        observation[name] = observation[name]
    action = policy.bpu_select_action(observation)
    return action
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
def _no_stats_error_str(name: str) -> str:
    return (
        f"`{name}` is infinity. You should either initialize with `stats` as an argument, or use a "
        "pretrained model."
    )
    
if __name__ == '__main__':
    main()