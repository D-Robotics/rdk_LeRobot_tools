#!/user/bin/env python

# Copyright (c) 2025，WuChao && MaChao D-Robotics.
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

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun
from lerobot.record import record_loop

try:
    from hbm_runtime import HB_HBMRuntime
    print("using: hbm_runtime")
except ImportError:
    print("hbm_runtime not found, please check!")
    exit()

class RDK_ACTConfig:
    """Simple configuration class for RDK_ACTPolicy to match LeRobot policy interface."""
    def __init__(self, device="cpu", n_action_steps=50):
        self.device = device
        self.n_action_steps = n_action_steps
        self.use_amp = False  # Automatic Mixed Precision not used for BPU

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bpu-act-path', type=str, default='rdk_LeRobot_tools/bpu_output', help='Path to LeRobot ACT Policy model.')
    """ 
    # example: --bpu-act-path pretrained_model
    .
    |-- BPU_ACTPolicy_TransformerLayers.hbm
    |-- BPU_ACTPolicy_VisionEncoder.hbm
    |-- action_mean.npy
    |-- action_mean_unnormalize.npy
    |-- action_std.npy
    |-- action_std_unnormalize.npy
    |-- camera1_mean.npy
    |-- camera1_std.npy
    |-- camera2_mean.npy
    `-- camera2_std.npy
    """
    parser.add_argument('--fps', type=int, default=30, help='FPS for recording') 
    parser.add_argument('--num-episodes', type=int, default=10, help='Number of episodes to run')
    parser.add_argument('--episode-time', type=int, default=60, help='Episode time in seconds')
    parser.add_argument('--n-action-steps', type=int, default=50, help='Number of action steps')
    parser.add_argument('--task-description', type=str, default='My task description', help='Task description')
    parser.add_argument('--repo-id', type=str, default='<HF_USER>/eval_lerobot1', help='HuggingFace dataset repository ID')
    opt = parser.parse_args()
    
    # 动态检测相机配置文件
    camera_names = detect_cameras_from_model(opt.bpu_act_path)
    print(f"Detected cameras from model: {camera_names}")
    
    # TODO: 用户需要根据实际硬件配置填写相机配置
    # 请根据检测到的相机名称和实际硬件连接情况修改以下配置
    camera_config = {"top": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30),"right": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30)}
    
    # 检查是否所有检测到的相机都有配置
    missing_cameras = set(camera_names) - set(camera_config.keys())
    if missing_cameras:
        print(f"Error: Missing camera configurations for: {missing_cameras}")
        print("Please add configurations for these cameras in the camera_config section:")
        for camera in missing_cameras:
            print(f'    "{camera}": OpenCVCameraConfig(index_or_path=X, width=640, height=480, fps=30),')
        return
        
    # 检查是否有多余的配置
    extra_cameras = set(camera_config.keys()) - set(camera_names)
    if extra_cameras:
        print(f"Warning: Extra camera configurations (not used by model): {extra_cameras}")
        # 移除不需要的相机配置
        for camera in extra_cameras:
            del camera_config[camera]
    
    # 显示相机配置
    for camera_name, config in camera_config.items():
        print(f"Camera '{camera_name}' -> index/path: {config.index_or_path}")
    
    robot_config = SO100FollowerConfig(
        port="/dev/ttyACM0", id="follower_arm", cameras=camera_config
    )

    # Initialize the robot
    robot = SO100Follower(robot_config)

    # Initialize the BPU policy
    policy = RDK_ACTPolicy_Dynamic(opt.bpu_act_path, opt.n_action_steps, camera_names)

    # Configure the dataset features
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    # Create the dataset
    dataset = LeRobotDataset.create(
        repo_id=opt.repo_id,
        fps=opt.fps,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # Initialize the keyboard listener and rerun visualization
    _, events = init_keyboard_listener()
    _init_rerun(session_name="recording")

    # Connect the robot
    robot.connect()

    for episode_idx in range(opt.num_episodes):
        log_say(f"Running BPU inference (dynamic {len(camera_names)} cameras), recording eval episode {episode_idx + 1} of {opt.num_episodes}")

        # Run the policy inference loop
        record_loop(
            robot=robot,
            events=events,
            fps=opt.fps,
            policy=policy,
            dataset=dataset,
            control_time_s=opt.episode_time,
            single_task=opt.task_description,
            display_data=False,
        )

        dataset.save_episode()

    # Clean up
    robot.disconnect()
    # dataset.push_to_hub()

def detect_cameras_from_model(bpu_act_path):
    """动态检测模型支持的相机数量和名称"""
    camera_names = []
    
    # 查找所有以 *_mean.npy 结尾但不是 action_mean 的文件
    mean_files = glob.glob(os.path.join(bpu_act_path, "*_mean.npy"))
    
    for mean_file in mean_files:
        filename = os.path.basename(mean_file)
        if filename.startswith("action_"):
            continue  # 跳过action相关文件
        
        # 提取相机名称 (去掉_mean.npy后缀)
        camera_name = filename.replace("_mean.npy", "")
        
        # 检查对应的std文件是否存在
        std_file = os.path.join(bpu_act_path, f"{camera_name}_std.npy")
        if os.path.exists(std_file):
            camera_names.append(camera_name)
            
    if not camera_names:
        # 如果没有检测到，使用默认配置
        print("Warning: No camera configuration detected, using default top+right")
        camera_names = ["top", "right"]
        
    return sorted(camera_names)  # 排序确保顺序一致

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
class RDK_ACTPolicy_Dynamic():
    def __init__(self, bpu_act_model_path, n_action_steps, camera_names):
        self.config = RDK_ACTConfig(device="cpu", n_action_steps=n_action_steps)
        self.n_action_steps = n_action_steps
        self._action_queue = deque([], maxlen=self.n_action_steps)
        self.camera_names = camera_names
        
        print(f"Initializing BPU policy with cameras: {camera_names}")
        
        # 动态加载归一化参数
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
        
        # 加载动作归一化参数
        action_std_path = os.path.join(bpu_act_model_path, "action_std.npy")
        action_mean_path = os.path.join(bpu_act_model_path, "action_mean.npy")
        action_std_unnormalize_path = os.path.join(bpu_act_model_path, "action_std_unnormalize.npy")
        action_mean_unnormalize_path = os.path.join(bpu_act_model_path, "action_mean_unnormalize.npy")
        
        # 检查所有必需文件
        required_files = [action_std_path, action_mean_path, action_std_unnormalize_path, action_mean_unnormalize_path]
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        self.action_std = torch.tensor(np.load(action_std_path), dtype=torch.float32) + 1e-8
        self.action_mean = torch.tensor(np.load(action_mean_path), dtype=torch.float32)
        self.action_std_unnormalize = torch.tensor(np.load(action_std_unnormalize_path), dtype=torch.float32)
        self.action_mean_unnormalize = torch.tensor(np.load(action_mean_unnormalize_path), dtype=torch.float32)

        # 验证参数
        for camera_name in camera_names:
            params = self.camera_params[camera_name]
            assert not torch.isinf(params['std']).any(), f"Invalid std for {camera_name}"
            assert not torch.isinf(params['mean']).any(), f"Invalid mean for {camera_name}"
            
        assert not torch.isinf(self.action_std).any(), "Invalid action_std"
        assert not torch.isinf(self.action_mean).any(), "Invalid action_mean"
        assert not torch.isinf(self.action_std_unnormalize).any(), "Invalid action_std_unnormalize"
        assert not torch.isinf(self.action_mean_unnormalize).any(), "Invalid action_mean_unnormalize"
        
        # 加载BPU模型
        bpu_act_policy_visionencoder_path = os.path.join(bpu_act_model_path,"BPU_ACTPolicy_VisionEncoder.hbm")
        bpu_act_policy_transformerlayers_path = os.path.join(bpu_act_model_path,"BPU_ACTPolicy_TransformerLayers.hbm")
        
        if not os.path.exists(bpu_act_policy_visionencoder_path):
            raise FileNotFoundError(f"Vision encoder model not found: {bpu_act_policy_visionencoder_path}")
        if not os.path.exists(bpu_act_policy_transformerlayers_path):
            raise FileNotFoundError(f"Transformer model not found: {bpu_act_policy_transformerlayers_path}")
            
        self.bpu_policy = HB_HBMRuntime([
            bpu_act_policy_visionencoder_path,
            bpu_act_policy_transformerlayers_path
        ])
        self.cnt = 0
        print("BPU models loaded successfully")
    
    def reset(self):
        """Reset the policy state, clear action queue."""
        self._action_queue.clear()
        
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        # normalize inputs
        batch = self.normalize_inputs(batch)

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            begin_time = time.time()
            
            # 准备状态输入
            state = batch["observation.state"].numpy().copy()
            
            # 动态处理所有相机的视觉特征
            vision_features = []
            for camera_name in self.camera_names:
                camera_input = batch[f'observation.images.{camera_name}'].numpy().copy()
                # 通过VisionEncoder获取特征
                vision_output = self.bpu_policy.run({"images": camera_input}, model_name="BPU_ACTPolicy_VisionEncoder")
                vision_feature = next(iter(vision_output["BPU_ACTPolicy_VisionEncoder"].values()))
                vision_features.append(vision_feature)
            
            # 构建TransformerLayers的输入
            transformer_inputs = {"states": state}
            for i, camera_name in enumerate(self.camera_names):
                transformer_inputs[f"{camera_name}_features"] = vision_features[i]
            
            # TransformerLayers推理
            transformer_outputs = self.bpu_policy.run(transformer_inputs, model_name="BPU_ACTPolicy_TransformerLayers")
            
            # 提取动作预测
            action_output = next(iter(transformer_outputs["BPU_ACTPolicy_TransformerLayers"].values()))
            actions = torch.from_numpy(action_output)[:, :self.n_action_steps]
            
            print(f"{self.cnt} BPU ACT Policy Time (dynamic {len(self.camera_names)} cameras): " + "\033[1;31m" + "%.2f ms"%(1000*(time.time() - begin_time)) + "\033[0m")
            self.cnt += 1
            
            actions = self.unnormalize_outputs({"action": actions})["action"]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
    
    def normalize_inputs(self, batch):
        # 归一化状态
        batch["observation.state"] = (batch["observation.state"] - self.action_mean) / self.action_std
        
        # 动态归一化所有相机图像
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
def _no_stats_error_str(name: str) -> str:
    return (
        f"`{name}` is infinity. You should either initialize with `stats` as an argument, or use a "
        "pretrained model."
    )
    
if __name__ == '__main__':
    main()