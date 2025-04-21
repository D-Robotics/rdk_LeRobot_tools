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
import torch
import numpy as np
from copy import copy
import argparse
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.robot_devices.robots.utils import make_robot
from lerobot.common.robot_devices.control_utils import busy_wait, is_headless, log_control_info

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, default='outputs/dp_so100_resnet18_0421_test1/pretrained_model', help='Path to LeRobot ACT Policy model.')
    """ 
    # example: --ckpt-path
    ./pretrained_model/
    ├── config.json
    ├── model.safetensors
    └── train_config.json
    """
    parser.add_argument('--fps', type=int, default=30, help='') 
    parser.add_argument('--inference-time', type=int, default=1000, help='seconds') 
    parser.add_argument('--n-action-steps', type=int, default=50, help='')
    opt = parser.parse_args()
    
    robot = make_robot("so100")
    robot.connect()
    policy = ACTPolicy.from_pretrained(opt.ckpt_path)
    policy.to("cpu")

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
        print(f"Inference time: {dt_s:.3f}s")
        busy_wait(1 / opt.fps - dt_s)

    robot.disconnect()

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
def predict_action(observation, policy):
    observation = copy(observation)
    for name in observation:
        if "image" in name:
            observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].permute(2, 0, 1).contiguous()
        observation[name] = observation[name].unsqueeze(0)
        observation[name] = observation[name]
    action = policy.select_action(observation)
    return action

if __name__ == '__main__':
    main()
