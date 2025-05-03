import time
import torch
import numpy as np
from copy import copy
import argparse
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.robot_devices.robots.utils import make_robot
from lerobot.common.robot_devices.control_utils import busy_wait, is_headless, log_control_info
from lerobot.common.robot_devices.robots.configs import THU_AIRBOTConfig
from lerobot.common.robot_devices.robots.thu_airbot import THU_AIRBOT

import math

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, 
                        default='/home/cauchy/lerobot_hp/test_weights_0502/airbot_2camera_1_104step_15frame_dp/115000', 
                        help='Path to LeRobot ACT Policy model.')
    """ 
    # example: --ckpt-path
    ./pretrained_model/
    ├── config.json
    ├── model.safetensors
    └── train_config.json
    """
    parser.add_argument('--fps', type=int, default=10, help='') 
    parser.add_argument('--inference-time', type=int, default=1000, help='seconds') 
    parser.add_argument('--n-action-steps', type=int, default=104, help='')
    parser.add_argument('--device', type=str, default='cuda', help='cpu, cuda')
    opt = parser.parse_args()
    device = torch.device(opt.device)
    
    
    cfg = THU_AIRBOTConfig()
    robot = THU_AIRBOT(cfg)
    # robot = make_robot("THU_AIRBOT")
    robot.connect()
    robot.set_state("fast")
    policy = DiffusionPolicy.from_pretrained(opt.ckpt_path)
    policy.to(device)  # cpu cuda

    for _ in range(opt.inference_time * opt.fps):
        start_time = time.perf_counter()
        # Read the follower state and access the frames from the cameras
        observation = robot.capture_observation()
        # Convert to pytorch format: channel first and float32 in [0,1]
        # with batch dimension
        pred_action = predict_action(observation, policy, device)[0]
        # Remove batch dimension
        action = pred_action.squeeze(0)
        # Move to cpu, if not already the case
        action = action.to("cpu")
        # Order the robot to move
        robot.send_action(action)
        # Print the action
        # print(f"Action: {action}")
        for i in range(14):
            print("%06.1f"%(action[i] / math.pi * 180), end=" ")
        print()
            

        dt_s = time.perf_counter() - start_time
        # print(f"Inference time: {dt_s:.3f}s")
        busy_wait(1 / opt.fps - dt_s)

    robot.disconnect()

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
def predict_action(observation, policy, device):
    observation = copy(observation)
    for name in observation:
        if "image" in name:
            observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].permute(2, 0, 1).contiguous()
        observation[name] = observation[name].unsqueeze(0)
        observation[name] = observation[name].to(device)
    action = policy.select_action(observation)
    return action

if __name__ == '__main__':
    main()
