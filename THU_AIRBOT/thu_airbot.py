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

# lerobot/common/robot_devices/robots/thu_airbot.py

import base64
import json
import os
import sys
from pathlib import Path

import copy

import cv2
import numpy as np
import torch
import zmq

from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.motors.feetech import TorqueMode
from lerobot.common.robot_devices.motors.utils import MotorsBus, make_motors_buses_from_configs

from lerobot.common.robot_devices.robots.feetech_calibration import run_arm_manual_calibration
from lerobot.common.robot_devices.robots.utils import get_arm_id
from lerobot.common.robot_devices.utils import RobotDeviceNotConnectedError

import time
import argparse
from airbot_py.arm import AIRBOTPlay, RobotMode, SpeedProfile

from lerobot.common.robot_devices.robots.configs import THU_AIRBOTConfig


PYNPUT_AVAILABLE = True
try:
    # Only import if there's a valid X server or if we're not on a Pi
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        print("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")

    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
except Exception as e:
    keyboard = None
    PYNPUT_AVAILABLE = False
    print(f"Could not import pynput: {e}")



class THU_AIRBOT:
    """
    THU_AIRBOT is a class for connecting to and controlling a remote mobile manipulator robot.
    The robot includes a three omniwheel mobile base and a remote follower arm.
    The leader arm is connected locally (on the laptop) and its joint positions are recorded and then
    forwarded to the remote follower arm (after applying a safety clamp).
    In parallel, keyboard teleoperation is used to generate raw velocity commands for the wheels.
    """

    def __init__(self, config: THU_AIRBOTConfig):
        """
        Expected keys in config:
          - ip, port, video_port for the remote connection.
          - calibration_dir, leader_arms, follower_arms, max_relative_target, etc.
        """
        self.robot_type = config.type
        self.config = config
        self.logs = {}
        # checks
        for left_name, right_name in zip(config.leader_arms, config.follower_arms):
            if config.leader_arms[left_name].port == config.follower_arms[right_name].port:
                raise ValueError(
                    "[Left] Lead port and follow port cannot be the same. Please use different ports."
                )
        # For teleoperation, the leader arm (local) is used to record the desired arm pose.
        self.leader_arms = {}
        for key, cfg in self.config.leader_arms.items():
            self.leader_arms[key] = AIRBOTPlay(url=cfg.ip,port=cfg.port)

        self.follower_arms = {}
        for key, cfg in self.config.follower_arms.items():
            self.follower_arms[key] = AIRBOTPlay(url=cfg.ip,port=cfg.port)

        self.cameras = make_cameras_from_configs(self.config.cameras)

        self.is_connected = False

        if PYNPUT_AVAILABLE:
            print("pynput is available - enabling local keyboard listener.")
            self.listener = keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release,
            )
            self.listener.start()
        else:
            print("pynput not available - skipping local keyboard listener.")
            self.listener = None
        ## pos limit
        nano = 0.01
        self.limit_max = [2.089, 0.181, 3.161, 3.012, 1.859, 3.017]
        self.limit_min = [-3.151, -2.963, -0.094, -3.012, -1.859, -3.017]
        for i in range(6):
            self.limit_max[i] = self.limit_max[i] - nano
            self.limit_min[i] = self.limit_min[i] + nano
        print(f"{self.limit_max = }")
        print(f"{self.limit_min = }")

    def get_motor_names(self, arms: dict[str, MotorsBus]) -> list:
        return [f"{arm}_{motor}" for arm, bus in arms.items() for motor in bus.motors]

    def get_motor_names(self, arms: dict[str, MotorsBus]) -> list:
        return [f"{arm}_{motor}" for arm, bus in arms.items() for motor in bus.motors]

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        follower_arm_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
            "eef",
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
            "eef",
        ]
        combined_names = follower_arm_names
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(combined_names),),
                "names": combined_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(combined_names),),
                "names": combined_names,
            },
        }

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    @property
    def available_arms(self):
        available = []
        for name in self.leader_arms:
            available.append(get_arm_id(name, "leader"))
        for name in self.follower_arms:
            available.append(get_arm_id(name, "follower"))
        return available

    def on_press(self, key):
        try:
            # Movement
            if key.char == self.teleop_keys["forward"]:
                self.pressed_keys["forward"] = True
            elif key.char == self.teleop_keys["backward"]:
                self.pressed_keys["backward"] = True
            elif key.char == self.teleop_keys["left"]:
                self.pressed_keys["left"] = True
            elif key.char == self.teleop_keys["right"]:
                self.pressed_keys["right"] = True
            elif key.char == self.teleop_keys["rotate_left"]:
                self.pressed_keys["rotate_left"] = True
            elif key.char == self.teleop_keys["rotate_right"]:
                self.pressed_keys["rotate_right"] = True

            # Quit teleoperation
            elif key.char == self.teleop_keys["quit"]:
                self.running = False
                return False

            # Speed control
            elif key.char == self.teleop_keys["speed_up"]:
                self.speed_index = min(self.speed_index + 1, 2)
                print(f"Speed index increased to {self.speed_index}")
            elif key.char == self.teleop_keys["speed_down"]:
                self.speed_index = max(self.speed_index - 1, 0)
                print(f"Speed index decreased to {self.speed_index}")

        except AttributeError:
            # e.g., if key is special like Key.esc
            if key == keyboard.Key.esc:
                self.running = False
                return False

    def on_release(self, key):
        try:
            if hasattr(key, "char"):
                if key.char == self.teleop_keys["forward"]:
                    self.pressed_keys["forward"] = False
                elif key.char == self.teleop_keys["backward"]:
                    self.pressed_keys["backward"] = False
                elif key.char == self.teleop_keys["left"]:
                    self.pressed_keys["left"] = False
                elif key.char == self.teleop_keys["right"]:
                    self.pressed_keys["right"] = False
                elif key.char == self.teleop_keys["rotate_left"]:
                    self.pressed_keys["rotate_left"] = False
                elif key.char == self.teleop_keys["rotate_right"]:
                    self.pressed_keys["rotate_right"] = False
        except AttributeError:
            pass

    def connect(self):
        # Leader
        for name in self.leader_arms:
            self.leader_arms[name].connect()
            print(f"Connecting {name} leader arm.")
            self.leader_arms[name].switch_mode(RobotMode.GRAVITY_COMP)

        # Follower
        for name in self.follower_arms:
            self.follower_arms[name].connect()
            print(f"Connecting {name} follower arm.")
            self.follower_arms[name].switch_mode(RobotMode.PLANNING_POS)
            self.follower_arms[name].set_speed_profile(SpeedProfile.FAST)

        # cameras
        for cam_name, cam in self.cameras.items():
            cam.connect()
            cam.async_read()
            print(f"camera_{cam_name} is connected.")
        
        self.safe_begin()
            
        self.is_connected = True

    def set_state(self, state):
        for name in self.follower_arms:
            if state == "slow":
                self.follower_arms[name].set_speed_profile(SpeedProfile.SLOW)
                print("set mode: ", state)
            elif state == "fast":
                self.follower_arms[name].set_speed_profile(SpeedProfile.FAST)
                print("set mode: ", state)
            elif state == "default":
                self.follower_arms[name].set_speed_profile(SpeedProfile.DEFAULT)
                print("set mode: ", state)
            else:
                assert RuntimeError(f"get {state}, set_state(state), state must be: \"slow\", \"fast\", \"default\". ")

    def _get_data(self):
        """
        Polls the video socket for up to 15 ms. If data arrives, decode only
        the *latest* message, returning frames, speed, and arm state. If
        nothing arrives for any field, use the last known values.
        """
        frames = {}
        present_speed = {}
        remote_arm_state_tensor = torch.zeros(6, dtype=torch.float32)

        # Poll up to 15 ms
        # poller = zmq.Poller()
        # poller.register(self.video_socket, zmq.POLLIN)
        # socks = dict(poller.poll(15))
        # if self.video_socket not in socks or socks[self.video_socket] != zmq.POLLIN:
        #     # No new data arrived → reuse ALL old data
        #     return (self.last_frames, self.last_present_speed, self.last_remote_arm_state)

        # Drain all messages, keep only the last
        last_msg = None
        # while True:
        #     try:
        #         obs_string = self.video_socket.recv_string(zmq.NOBLOCK)
        #         last_msg = obs_string
        #     except zmq.Again:
        #         break

        if not last_msg:
            # No new message → also reuse old
            return (self.last_frames, self.last_present_speed, self.last_remote_arm_state)

        # Decode only the final message
        try:
            observation = json.loads(last_msg)

            images_dict = observation.get("images", {})
            new_speed = observation.get("present_speed", {})
            new_arm_state = observation.get("follower_arm_state", None)

            # Convert images
            for cam_name, image_b64 in images_dict.items():
                if image_b64:
                    jpg_data = base64.b64decode(image_b64)
                    np_arr = np.frombuffer(jpg_data, dtype=np.uint8)
                    frame_candidate = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame_candidate is not None:
                        frames[cam_name] = frame_candidate

            # If remote_arm_state is None and frames is None there is no message then use the previous message
            if new_arm_state is not None and frames is not None:
                self.last_frames = frames

                remote_arm_state_tensor = torch.tensor(new_arm_state, dtype=torch.float32)
                self.last_remote_arm_state = remote_arm_state_tensor

                present_speed = new_speed
                self.last_present_speed = new_speed
            else:
                frames = self.last_frames

                remote_arm_state_tensor = self.last_remote_arm_state

                present_speed = self.last_present_speed

        except Exception as e:
            print(f"[DEBUG] Error decoding video message: {e}")
            # If decode fails, fall back to old data
            return (self.last_frames, self.last_present_speed, self.last_remote_arm_state)

        return frames, present_speed, remote_arm_state_tensor

    def _process_present_speed(self, present_speed: dict) -> torch.Tensor:
        state_tensor = torch.zeros(3, dtype=torch.int32)
        if present_speed:
            decoded = {key: THU_AIRBOT.raw_to_degps(value) for key, value in present_speed.items()}
            if "1" in decoded:
                state_tensor[0] = decoded["1"]
            if "2" in decoded:
                state_tensor[1] = decoded["2"]
            if "3" in decoded:
                state_tensor[2] = decoded["3"]
        return state_tensor

    def teleop_step(
        self, record_data: bool = False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("THU_AIRBOT is not connected. Run `connect()` first.")

        # speed_setting = self.speed_levels[self.speed_index]
        # xy_speed = speed_setting["xy"]  # e.g. 0.1, 0.25, or 0.4
        # theta_speed = speed_setting["theta"]  # e.g. 30, 60, or 90

        # Prepare to assign the position of the leader to the follower
        # leader arms
        leader_arm_positions = []
        for name in self.leader_arms:
            pos = self.leader_arms[name].get_joint_pos()
            for i in range(6):
                pos[i] = pos[i] if pos[i] < self.limit_max[i] else self.limit_max[i]
                pos[i] = pos[i] if pos[i] > self.limit_min[i] else self.limit_min[i]
            eef = self.leader_arms[name].get_eef_pos()
            pos = np.array(pos + eef, dtype=np.float32)
            print(f"{name}: {pos.shape = }")
            # print(name, " ", pos)
            pos_tensor = torch.from_numpy(pos).float()
            leader_arm_positions.extend(pos_tensor.tolist())

        self.send_action(leader_arm_positions)
        time.sleep(0.01)  # debug

        if not record_data:
            return

        obs_dict = self.capture_observation()

        arm_state_tensor = torch.tensor(leader_arm_positions, dtype=torch.float32)

        action_dict = {"action": arm_state_tensor}

        return obs_dict, action_dict

    def capture_observation(self) -> dict:
        """
        Capture observations from the remote robot: current follower arm positions,
        present wheel speeds (converted to body-frame velocities: x, y, theta),
        and a camera frame.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("Not connected. Run `connect()` first.")
        
        follower_arm_positions = []
        for name in self.follower_arms:
            pos = self.follower_arms[name].get_joint_pos()
            eef = self.follower_arms[name].get_eef_pos()
            pos = np.array(pos + eef, dtype=np.float32)
            pos_tensor = torch.from_numpy(pos).float()
            follower_arm_positions.extend(pos_tensor.tolist())
        
        follower_arm_positions = torch.tensor(follower_arm_positions, dtype=torch.float32)
        obs_dict = {"observation.state": follower_arm_positions}

        # Loop over each configured camera
        for cam_name, cam in self.cameras.items():
            
            # frame = cam.read()
            frame = cam.color_image
            if frame is None:
                print(f"[ERROR] observation.images.{cam_name} get None images, please check!")
                # Create a black image using the camera's configured width, height, and channels
                frame = np.zeros((cam.height, cam.width, cam.channels), dtype=np.uint8)
            obs_dict[f"observation.images.{cam_name}"] = torch.from_numpy(frame)

        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("Not connected. Run `connect()` first.")

        slice_begin = 0
        slice_end = 7
        for follower in self.follower_arms:
            self.follower_arms[follower].servo_joint_pos(action[slice_begin:slice_end-1])
            self.follower_arms[follower].servo_eef_pos(action[slice_end-1:slice_end])
            slice_begin += 7
            slice_end += 7

        return action

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("Not connected.")
        # Leader
        for name in self.leader_arms:
            self.leader_arms[name].switch_mode(RobotMode.GRAVITY_COMP)
            # self.leader_arms[name].set_speed_profile(SpeedProfile.SLOW)
            # self.follower_arms[name].servo_joint_pos([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            # self.leader_arms[name].set_speed_profile(SpeedProfile.DEFAULT)
            # self.leader_arms[name].switch_mode(RobotMode.GRAVITY_COMP)
            

        # Follower 
        for name in self.follower_arms:
            self.follower_arms[name].switch_mode(RobotMode.GRAVITY_COMP)
            # self.follower_arms[name].set_speed_profile(SpeedProfile.SLOW)
            # self.follower_arms[name].servo_joint_pos([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            # self.leader_arms[name].set_speed_profile(SpeedProfile.DEFAULT)
            # self.leader_arms[name].switch_mode(RobotMode.GRAVITY_COMP)

        if PYNPUT_AVAILABLE:
            self.listener.stop()

        self.is_connected = False
        print("[INFO] Disconnected from remote robot.")

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
        if PYNPUT_AVAILABLE:
            self.listener.stop()
    
    def safe_begin(self):
        for leader, follower in zip(self.leader_arms, self.follower_arms):
            self.leader_arms[leader].switch_mode(RobotMode.PLANNING_POS)
            if (
                sum(
                    [
                        abs(i - j)
                        for i, j in zip(self.leader_arms[leader].get_joint_pos(), self.follower_arms[follower].get_joint_pos())
                    ]
                )
                > 0.1
            ):
                self.follower_arms[follower].switch_mode(RobotMode.PLANNING_POS)
                self.follower_arms[follower].set_speed_profile(SpeedProfile.SLOW)
                self.follower_arms[follower].move_to_joint_pos(self.leader_arms[leader].get_joint_pos())
                # time.sleep(2)
            print(f"{follower} is ready to follow {leader}.")

            self.leader_arms[leader].switch_mode(RobotMode.GRAVITY_COMP)
            self.follower_arms[follower].switch_mode(RobotMode.SERVO_JOINT_POS)
            self.follower_arms[follower].set_speed_profile(SpeedProfile.FAST)
    

    @staticmethod
    def degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        speed_in_steps = abs(degps) * steps_per_deg
        speed_int = int(round(speed_in_steps))
        if speed_int > 0x7FFF:
            speed_int = 0x7FFF
        if degps < 0:
            return speed_int | 0x8000
        else:
            return speed_int & 0x7FFF

    @staticmethod
    def raw_to_degps(raw_speed: int) -> float:
        steps_per_deg = 4096.0 / 360.0
        magnitude = raw_speed & 0x7FFF
        degps = magnitude / steps_per_deg
        if raw_speed & 0x8000:
            degps = -degps
        return degps

    def body_to_wheel_raw(
        self,
        x_cmd: float,
        y_cmd: float,
        theta_cmd: float,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
        max_raw: int = 3000,
    ) -> dict:
        """
        Convert desired body-frame velocities into wheel raw commands.

        Parameters:
          x_cmd      : Linear velocity in x (m/s).
          y_cmd      : Linear velocity in y (m/s).
          theta_cmd  : Rotational velocity (deg/s).
          wheel_radius: Radius of each wheel (meters).
          base_radius : Distance from the center of rotation to each wheel (meters).
          max_raw    : Maximum allowed raw command (ticks) per wheel.

        Returns:
          A dictionary with wheel raw commands:
             {"left_wheel": value, "back_wheel": value, "right_wheel": value}.

        Notes:
          - Internally, the method converts theta_cmd to rad/s for the kinematics.
          - The raw command is computed from the wheels angular speed in deg/s
            using degps_to_raw(). If any command exceeds max_raw, all commands
            are scaled down proportionally.
        """
        # Convert rotational velocity from deg/s to rad/s.
        theta_rad = theta_cmd * (np.pi / 180.0)
        # Create the body velocity vector [x, y, theta_rad].
        velocity_vector = np.array([x_cmd, y_cmd, theta_rad])

        # Define the wheel mounting angles (defined from y axis cw)
        angles = np.radians(np.array([300, 180, 60]))
        # Build the kinematic matrix: each row maps body velocities to a wheel’s linear speed.
        # The third column (base_radius) accounts for the effect of rotation.
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # Compute each wheel’s linear speed (m/s) and then its angular speed (rad/s).
        wheel_linear_speeds = m.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius

        # Convert wheel angular speeds from rad/s to deg/s.
        wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

        # Scaling
        steps_per_deg = 4096.0 / 360.0
        raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]
        max_raw_computed = max(raw_floats)
        if max_raw_computed > max_raw:
            scale = max_raw / max_raw_computed
            wheel_degps = wheel_degps * scale

        # Convert each wheel’s angular speed (deg/s) to a raw integer.
        wheel_raw = [THU_AIRBOT.degps_to_raw(deg) for deg in wheel_degps]

        return {"left_wheel": wheel_raw[0], "back_wheel": wheel_raw[1], "right_wheel": wheel_raw[2]}

    def wheel_raw_to_body(
        self, wheel_raw: dict, wheel_radius: float = 0.05, base_radius: float = 0.125
    ) -> tuple:
        """
        Convert wheel raw command feedback back into body-frame velocities.

        Parameters:
          wheel_raw   : Dictionary with raw wheel commands (keys: "left_wheel", "back_wheel", "right_wheel").
          wheel_radius: Radius of each wheel (meters).
          base_radius : Distance from the robot center to each wheel (meters).

        Returns:
          A tuple (x_cmd, y_cmd, theta_cmd) where:
             x_cmd      : Linear velocity in x (m/s).
             y_cmd      : Linear velocity in y (m/s).
             theta_cmd  : Rotational velocity in deg/s.
        """
        # Extract the raw values in order.
        raw_list = [
            int(wheel_raw.get("left_wheel", 0)),
            int(wheel_raw.get("back_wheel", 0)),
            int(wheel_raw.get("right_wheel", 0)),
        ]

        # Convert each raw command back to an angular speed in deg/s.
        wheel_degps = np.array([THU_AIRBOT.raw_to_degps(r) for r in raw_list])
        # Convert from deg/s to rad/s.
        wheel_radps = wheel_degps * (np.pi / 180.0)
        # Compute each wheel’s linear speed (m/s) from its angular speed.
        wheel_linear_speeds = wheel_radps * wheel_radius

        # Define the wheel mounting angles (defined from y axis cw)
        angles = np.radians(np.array([300, 180, 60]))
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # Solve the inverse kinematics: body_velocity = M⁻¹ · wheel_linear_speeds.
        m_inv = np.linalg.inv(m)
        velocity_vector = m_inv.dot(wheel_linear_speeds)
        x_cmd, y_cmd, theta_rad = velocity_vector
        theta_cmd = theta_rad * (180.0 / np.pi)
        return (x_cmd, y_cmd, theta_cmd)
