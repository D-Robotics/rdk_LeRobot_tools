#!/user/bin/env python

# Copyright (c) 2025，WuChao D-Robotics.
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

import torch
from torch import Tensor
from collections import deque

from lerobot.common.robot_devices.robots.utils import make_robot
from lerobot.common.robot_devices.control_utils import busy_wait

try:
    from libpycauchyS100tools import BPU_ACTPolicy
    print("using: libpycauchyS100tools")
except:
    from libpycauchytools import BPU_ACTPolicy
    print("using: libpycauchytools")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bpu-act-path', type=str, default='/root/lerobot_b/bpu_output_act_0417_2arms', help='Path to LeRobot ACT Policy model.')
    """ 
    # example: --bpy-act-path pretrained_model
    .
    |-- BPU_ACTPolicy_TransformerLayers.hbm
    |-- BPU_ACTPolicy_VisionEncoder.hbm
    |-- action_mean.npy
    |-- action_mean_unnormalize.npy
    |-- action_std.npy
    |-- action_std_unnormalize.npy
    |-- laptop_mean.npy
    |-- laptop_std.npy
    |-- phone_mean.npy
    `-- phone_std.npy
    """
    parser.add_argument('--fps', type=int, default=30, help='') 
    parser.add_argument('--inference-time', type=int, default=1000, help='seconds') 
    parser.add_argument('--n-action-steps', type=int, default=50, help='')
    opt = parser.parse_args([])
    
    robot = make_robot("so100")
    robot.connect()
    policy = RDK_ACTPolicy(opt.bpu_act_path, opt.n_action_steps)
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
    def __init__(self, bpu_act_model_path, n_action_steps):
        self.n_action_steps = n_action_steps
        self._action_queue = deque([], maxlen=self.n_action_steps)
        # Pre Process and Post Process
        self.laptop_std_path = os.path.join(bpu_act_model_path, "laptop_std.npy")
        self.laptop_mean_path = os.path.join(bpu_act_model_path, "laptop_mean.npy")
        self.phone_std_path = os.path.join(bpu_act_model_path, "phone_std.npy")
        self.phone_mean_path = os.path.join(bpu_act_model_path, "phone_mean.npy")
        self.action_std_path = os.path.join(bpu_act_model_path, "action_std.npy")
        self.action_mean_path = os.path.join(bpu_act_model_path, "action_mean.npy")
        self.action_std_unnormalize_path = os.path.join(bpu_act_model_path, "action_std_unnormalize.npy")
        self.action_mean_unnormalize_path = os.path.join(bpu_act_model_path, "action_mean_unnormalize.npy")
        self.bpu_act_policy_visionencoder_path = os.path.join(bpu_act_model_path,"BPU_ACTPolicy_VisionEncoder.hbm")
        self.bpu_act_policy_transformerlayers_path = os.path.join(bpu_act_model_path,"BPU_ACTPolicy_TransformerLayers.hbm")
        # check
        paths = {
            "laptop_std_path": self.laptop_std_path,
            "laptop_mean_path": self.laptop_mean_path,
            "phone_std_path": self.phone_std_path,
            "phone_mean_path": self.phone_mean_path,
            "action_std_path": self.action_std_path,
            "action_mean_path": self.action_mean_path,
            "action_std_unnormalize_path": self.action_std_unnormalize_path,
            "action_mean_unnormalize_path": self.action_mean_unnormalize_path,
            "bpu_act_policy_visionencoder_path": self.bpu_act_policy_visionencoder_path,
            "bpu_act_policy_transformerlayers_path": self.bpu_act_policy_transformerlayers_path,
        }
        for name, path in paths.items():
            assert os.path.exists(path), f"{name} not exist: {path}, please check!"
        
        self.laptop_std = torch.tensor(np.load(self.laptop_std_path), dtype=torch.float32) + 1e-8
        self.laptop_mean = torch.tensor(np.load(self.laptop_mean_path), dtype=torch.float32)
        self.phone_std = torch.tensor(np.load(self.phone_std_path), dtype=torch.float32) + 1e-8
        self.phone_mean = torch.tensor(np.load(self.phone_mean_path), dtype=torch.float32)
        self.action_std = torch.tensor(np.load(self.action_std_path), dtype=torch.float32) + 1e-8
        self.action_mean = torch.tensor(np.load(self.action_mean_path), dtype=torch.float32)
        self.action_std_unnormalize = torch.tensor(np.load(self.action_std_unnormalize_path), dtype=torch.float32)
        self.action_mean_unnormalize = torch.tensor(np.load(self.action_mean_unnormalize_path), dtype=torch.float32)

        assert not torch.isinf(self.laptop_std).any(), _no_stats_error_str("mean")
        assert not torch.isinf(self.laptop_mean).any(), _no_stats_error_str("std")
        assert not torch.isinf(self.phone_std).any(), _no_stats_error_str("mean")
        assert not torch.isinf(self.phone_mean).any(), _no_stats_error_str("std")
        assert not torch.isinf(self.action_std).any(), _no_stats_error_str("mean")
        assert not torch.isinf(self.action_mean).any(), _no_stats_error_str("std")
        assert not torch.isinf(self.action_std_unnormalize).any(), _no_stats_error_str("mean")
        assert not torch.isinf(self.action_mean_unnormalize).any(), _no_stats_error_str("std")
        
        # load BPU model
        self.bpu_policy = BPU_ACTPolicy(self.bpu_act_policy_visionencoder_path, self.bpu_act_policy_transformerlayers_path)
        self.cnt = 0

    def bpu_select_action(self, batch: dict[str, Tensor]) -> Tensor:
        # normalize inputs
        batch = self.normalize_inputs(batch)

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            begin_time = time.time()
            actions = self.bpu_model(batch)[:, :self.n_action_steps]
            print(f"{self.cnt} BPU ACT Policy Time: " + "\033[1;31m" + "%.2f ms"%(1000*(time.time() - begin_time)) + "\033[0m")
            self.cnt += 1
            actions = self.unnormalize_outputs({"action": actions})["action"]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
    def bpu_model(self, batch):
        state = batch["observation.state"].numpy().copy()
        laptop = batch['observation.images.laptop'].numpy().copy()
        phone = batch['observation.images.phone'].numpy().copy()
        actions = self.bpu_policy(state, laptop, phone)
        actions = torch.from_numpy(actions)
        return actions
    
    def normalize_inputs(self, batch):
        batch["observation.state"] = (batch["observation.state"] - self.action_mean) / self.action_std
        batch['observation.images.laptop'] = (batch['observation.images.laptop'] - self.laptop_mean) / self.laptop_std
        batch['observation.images.phone'] = (batch['observation.images.phone'] - self.phone_mean) / self.phone_std
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
