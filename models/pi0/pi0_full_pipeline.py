#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import signal
import socket
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import msg_pb2

MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
MOTOR_CONFIRMATION = "ENABLE_SO100_MOTORS"
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
ACTION_HORIZON = 50
CONTROL_HZ = 30.0
GRIPPER_TELEMETRY_REGISTERS = (
    "Torque_Enable",
    "Present_Current",
    "Present_Load",
    "Present_Voltage",
    "Present_Temperature",
    "Status",
)
GRIPPER_CONFIG_REGISTERS = (
    "Max_Torque_Limit",
    "Torque_Limit",
    "Protection_Current",
    "Protective_Torque",
    "Protection_Time",
    "Overload_Torque",
    "Over_Current_Protection_Time",
    "Minimum_Startup_Force",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recv_exact(conn: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Pi0 engine disconnected")
        data.extend(chunk)
    return bytes(data)


def recv_message(conn: socket.socket):
    size = struct.unpack("!I", recv_exact(conn, 4))[0]
    if size <= 0 or size > MAX_MESSAGE_BYTES:
        raise ValueError(f"Invalid protobuf message size: {size}")
    message = msg_pb2.MultiModalInput()
    message.ParseFromString(recv_exact(conn, size))
    return message


def send_message(conn: socket.socket, message) -> None:
    payload = message.SerializeToString()
    conn.sendall(struct.pack("!I", len(payload)) + payload)


def add_tensor(container, dtype: int, shape, data: bytes) -> None:
    tensor = container.add()
    tensor.dtype = dtype
    tensor.shape.extend(shape)
    tensor.data = data


def image_to_chw_uint8(image: Any) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected HWC/CHW image, got {array.shape}")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got {array.shape}")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def build_request(
    state: np.ndarray,
    images_rgb: list[np.ndarray],
    task: str,
    sequence: int,
    reset: bool,
):
    request = msg_pb2.MultiModalInput()
    request.header.seq = sequence
    request.header.stamp.sec = int(time.time())
    request.header.frame_id = "so100_pi0_full_pipeline"
    request.header.reset = reset

    for image_rgb in images_rgb:
        image = image_to_chw_uint8(image_rgb)
        add_tensor(request.images, msg_pb2.Tensor.UINT8, image.shape, image.tobytes())

    prompt = task.encode("utf-8")
    add_tensor(request.languages, msg_pb2.Tensor.STRING, [len(prompt)], prompt)
    add_tensor(request.states, msg_pb2.Tensor.FLOAT64, state.shape, state.tobytes())
    return request


def parse_actions(response) -> np.ndarray:
    for tensor in list(response.states) + list(response.languages):
        shape = tuple(tensor.shape)
        if tensor.dtype == msg_pb2.Tensor.FLOAT64:
            values = np.frombuffer(tensor.data, dtype=np.float64)
        elif tensor.dtype == msg_pb2.Tensor.FP16:
            values = np.frombuffer(tensor.data, dtype=np.float16).astype(np.float64)
        else:
            continue
        if np.prod(shape, dtype=np.int64) != values.size:
            raise ValueError(f"Action shape/data mismatch: {shape}, {values.size}")
        actions = values.reshape(shape)
        if actions.ndim == 3:
            if actions.shape[0] != 1:
                raise ValueError(f"Expected batch size 1, got {actions.shape}")
            actions = actions[0]
        expected_shape = (ACTION_HORIZON, len(MOTOR_NAMES))
        if actions.shape != expected_shape:
            raise ValueError(
                f"Expected {expected_shape} SO100 action chunk, got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("Action contains NaN or Inf")
        return actions
    raise ValueError("Response contains no action tensor")


def infer(
    conn: socket.socket,
    state: np.ndarray,
    images: list[np.ndarray],
    task: str,
    sequence: int,
    reset: bool,
) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    send_message(conn, build_request(state, images, task, sequence, reset))
    actions = parse_actions(recv_message(conn))
    return actions, (time.perf_counter() - start) * 1000.0


def tail(path: Path, lines: int = 80) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def start_engine(args, log_path: Path):
    env = os.environ.copy()
    if args.fixed_noise:
        env["PI0_FIXED_NOISE_FILE"] = str(args.fixed_noise_file)
    else:
        env.pop("PI0_FIXED_NOISE_FILE", None)
    log_file = log_path.open("w")
    process = subprocess.Popen(
        [str(args.engine_runner), str(args.config)],
        cwd=SCRIPT_DIR,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, log_file


def stop_engine(process, log_file) -> None:
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
    if log_file is not None:
        log_file.close()


def accept_engine(server: socket.socket, process, log_path: Path, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    server.settimeout(0.5)
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Pi0 engine exited before connecting with status {process.returncode}:\n{tail(log_path)}"
            )
        try:
            return server.accept()
        except socket.timeout:
            pass
    raise TimeoutError(f"Timed out waiting for Pi0 engine:\n{tail(log_path)}")


def resolve_robot_port(requested: Path) -> Path:
    if requested.exists():
        return requested
    fallback = Path("/dev/ttyACM0")
    if fallback.exists():
        logging.warning("Robot port %s is missing; using %s", requested, fallback)
        return fallback
    raise FileNotFoundError(f"No SO100 serial device: {requested} or {fallback}")


def make_robot(args):
    sys.path.insert(0, str(args.lerobot_root / "src"))
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

    cameras = {
        args.camera_name: OpenCVCameraConfig(
            index_or_path=args.camera,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            warmup_s=args.camera_warmup_s,
            fourcc="MJPG",
        )
    }
    side_camera = getattr(args, "side_camera", None)
    if side_camera is not None:
        side_camera_name = getattr(args, "side_camera_name", "side")
        if side_camera_name in cameras:
            raise ValueError(f"Duplicate camera name: {side_camera_name}")
        cameras[side_camera_name] = OpenCVCameraConfig(
            index_or_path=side_camera,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            warmup_s=args.camera_warmup_s,
            fourcc="MJPG",
        )

    config = SO100FollowerConfig(
        port=str(resolve_robot_port(args.robot_port)),
        calibration_dir=args.calibration_dir,
        id=args.robot_id,
        max_relative_target=args.max_relative_target,
        cameras=cameras,
        use_degrees=True,
    )
    return SO100Follower(config)


def connect_components_readonly(robot) -> None:
    robot.bus.connect()
    if not robot.is_calibrated:
        raise RuntimeError(
            "SO100 calibration does not match the controller; refusing automatic calibration"
        )
    for camera in robot.cameras.values():
        camera.connect()
    if not robot.is_connected:
        raise RuntimeError("SO100 bus or camera failed to connect")


def disconnect_components(robot, torque_enabled: bool) -> None:
    if robot is None:
        return
    if torque_enabled and robot.bus.is_connected:
        try:
            robot.bus.disable_torque(num_retry=5)
        except Exception:
            logging.exception("Failed to disable SO100 torque during shutdown")
    for camera in robot.cameras.values():
        if camera.is_connected:
            try:
                camera.disconnect()
            except Exception:
                logging.exception("Failed to disconnect camera")
    if robot.bus.is_connected:
        robot.bus.disconnect(disable_torque=False)


def torque_state(robot) -> dict[str, int]:
    values = robot.bus.sync_read("Torque_Enable", normalize=False)
    return {name: int(values[name]) for name in MOTOR_NAMES}


def state_from_bus(robot) -> np.ndarray:
    positions = robot.bus.sync_read("Present_Position")
    return np.asarray([positions[name] for name in MOTOR_NAMES], dtype=np.float64)


def get_live_observation(robot, camera_names: tuple[str, ...]):
    observation = robot.get_observation()
    state = np.asarray(
        [observation[f"{name}.pos"] for name in MOTOR_NAMES], dtype=np.float64
    )
    images = [np.asarray(observation[name]).copy() for name in camera_names]
    return observation, state, images


def read_latest_images(robot, camera_names: tuple[str, ...]) -> list[np.ndarray]:
    return [np.asarray(robot.cameras[name].read_latest()).copy() for name in camera_names]


def save_rgb(path: Path, image: np.ndarray) -> None:
    array = np.asarray(image)
    if array.shape[-1] != 3:
        raise ValueError(f"Cannot save image with shape {array.shape}")
    cv2.imwrite(str(path), cv2.cvtColor(array, cv2.COLOR_RGB2BGR))


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def validate_artifacts(args) -> dict[str, str]:
    required = [
        args.config,
        args.engine_runner,
        args.norm_stats,
        args.calibration_file,
    ]
    if args.fixed_noise:
        required.append(args.fixed_noise_file)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    stats_hash = sha256(args.norm_stats)
    calibration_hash = sha256(args.calibration_file)
    if stats_hash != args.expected_stats_sha256:
        raise RuntimeError(
            f"SO100 stats hash mismatch: expected {args.expected_stats_sha256}, got {stats_hash}"
        )
    if calibration_hash != args.expected_calibration_sha256:
        raise RuntimeError(
            "SO100 calibration hash mismatch: "
            f"expected {args.expected_calibration_sha256}, got {calibration_hash}"
        )

    config = json.loads(args.config.read_text())
    if config.get("runtime") != "standalone_dnn":
        raise RuntimeError("Deployment config must use runtime=standalone_dnn")
    if config.get("server_ip") != args.host or int(config.get("server_port")) != args.port:
        raise RuntimeError(
            "Deployment config server endpoint does not match Python server: "
            f"{config.get('server_ip')}:{config.get('server_port')} vs {args.host}:{args.port}"
        )
    if int(config.get("exec_size")) != len(MOTOR_NAMES):
        raise RuntimeError(f"Expected exec_size=6, got {config.get('exec_size')}")
    if int(config.get("real_camera_num", 0)) != 2:
        raise RuntimeError(
            f"Expected real_camera_num=2, got {config.get('real_camera_num')}"
        )

    model_artifacts = {
        "siglip": (Path(config["siglip_hbm_path"]), args.expected_siglip_sha256),
        "paligemma": (
            Path(config["paligemma_hbm_path"]),
            args.expected_paligemma_sha256,
        ),
        "expert": (Path(config["action_hbm_path"]), args.expected_expert_sha256),
    }
    model_hashes = {}
    for name, (path, expected_hash) in model_artifacts.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"{name} HBM hash mismatch: expected {expected_hash}, got {actual_hash}"
            )
        model_hashes[f"{name}_sha256"] = actual_hash

    prompt_embedding_path = Path(config["prompt_embedding_path"])
    if not prompt_embedding_path.is_file():
        raise FileNotFoundError(prompt_embedding_path)
    prompt_embedding_hash = sha256(prompt_embedding_path)
    if prompt_embedding_hash != args.expected_prompt_embedding_sha256:
        raise RuntimeError(
            "Prompt embedding hash mismatch: "
            f"expected {args.expected_prompt_embedding_sha256}, got {prompt_embedding_hash}"
        )

    subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "validate_pi0_config.py"), str(args.config)],
        check=True,
    )
    hashes = {
        "stats_sha256": stats_hash,
        "calibration_sha256": calibration_hash,
        **model_hashes,
        "prompt_embedding_sha256": prompt_embedding_hash,
        "engine_runner_sha256": sha256(args.engine_runner),
    }
    if args.fixed_noise:
        fixed_noise_hash = sha256(args.fixed_noise_file)
        if fixed_noise_hash != args.expected_fixed_noise_sha256:
            raise RuntimeError(
                "Fixed noise hash mismatch: "
                f"expected {args.expected_fixed_noise_sha256}, got {fixed_noise_hash}"
            )
        hashes["fixed_noise_sha256"] = fixed_noise_hash
    return hashes


def action_summary(actions: np.ndarray, state: np.ndarray) -> dict[str, Any]:
    first_delta = actions[0] - state
    return {
        "shape": list(actions.shape),
        "first": actions[0].tolist(),
        "first_delta": first_delta.tolist(),
        "max_abs_first_delta": float(np.max(np.abs(first_delta))),
        "min": actions.min(axis=0).tolist(),
        "max": actions.max(axis=0).tolist(),
    }


def stitch_action_chunk(
    actions: np.ndarray, previous_target: np.ndarray | None, blend_steps: int
) -> tuple[np.ndarray, dict[str, Any] | None]:
    command_actions = actions.copy()
    command_actions[:, -1] = np.clip(command_actions[:, -1], 0.0, 100.0)
    if previous_target is None or blend_steps <= 0:
        return command_actions, None

    applied_steps = min(blend_steps, len(command_actions))
    offset = previous_target - command_actions[0]
    weights = np.linspace(1.0, 0.0, applied_steps, dtype=np.float64)
    command_actions[:applied_steps] += weights[:, None] * offset
    command_actions[:, -1] = np.clip(command_actions[:, -1], 0.0, 100.0)
    return command_actions, {
        "steps": applied_steps,
        "raw_first_delta": (actions[0] - previous_target).tolist(),
        "applied_first_delta": (command_actions[0] - previous_target).tolist(),
        "offset": offset.tolist(),
    }


def read_gripper_registers(robot, registers: tuple[str, ...]) -> dict[str, int]:
    values: dict[str, int] = {}
    for register in registers:
        value = int(
            robot.bus.read(
                register, "gripper", normalize=False, num_retry=2
            )
        )
        if register == "Present_Temperature" and not 0 <= value <= 100:
            value = int(
                robot.bus.read(
                    register, "gripper", normalize=False, num_retry=5
                )
            )
        values[register] = value
    return values


def verify_execute_request(args) -> None:
    if not args.execute:
        return
    if args.confirm != MOTOR_CONFIRMATION:
        raise RuntimeError(f"Motor execution requires --confirm {MOTOR_CONFIRMATION}")
    if args.force_model_actions:
        return
    if not args.fixed_noise:
        raise RuntimeError("Initial motor smoke requires fixed noise")
    if not args.vision_check:
        raise RuntimeError("Motor execution requires the live-vs-black vision check")


def enable_torque_holding_current(robot, current: np.ndarray) -> None:
    hold = {name: float(current[index]) for index, name in enumerate(MOTOR_NAMES)}
    robot.bus.sync_write("Goal_Position", hold)
    try:
        robot.bus.enable_torque(num_retry=5)
        enabled = torque_state(robot)
        if any(value != 1 for value in enabled.values()):
            raise RuntimeError(f"Failed to enable all SO100 motors: {enabled}")
    except Exception:
        robot.bus.disable_torque(num_retry=5)
        raise


def send_target(robot, target: np.ndarray) -> dict[str, float]:
    action = {
        f"{name}.pos": float(target[index]) for index, name in enumerate(MOTOR_NAMES)
    }
    return robot.send_action(action)


def parse_args():
    parser = argparse.ArgumentParser(
        description="One-command S600 standalone Pi0 + LeRobot SO100 pipeline"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=SCRIPT_DIR
        / "configs/deployments/pi0_full_v5_2cam_positionfp16_siglip_fixed16_paligemma_hbmkv_expert_20260801.json",
    )
    parser.add_argument(
        "--engine-runner",
        "--xlm-runner",
        dest="engine_runner",
        type=Path,
        default=SCRIPT_DIR / "run_pi0_standalone_config.sh",
    )
    parser.add_argument(
        "--fixed-noise-file",
        type=Path,
        default=SCRIPT_DIR / "configs/fixed_noise_cv_12345678_fp16.bin",
    )
    parser.add_argument(
        "--fixed-noise", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument(
        "--engine-connect-timeout-s",
        "--xlm-connect-timeout-s",
        dest="engine_connect_timeout_s",
        type=float,
        default=120.0,
    )

    parser.add_argument("--lerobot-root", type=Path, default=Path("/root/lerobot"))
    parser.add_argument(
        "--robot-port",
        type=Path,
        default=Path(
            "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A058697-if00"
        ),
    )
    parser.add_argument("--robot-id", default="so100_follower")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=SCRIPT_DIR / "calibration/robots/so_follower",
    )
    parser.add_argument(
        "--calibration-file",
        type=Path,
        default=SCRIPT_DIR / "calibration/robots/so_follower/so100_follower.json",
    )
    parser.add_argument("--camera", type=Path, default=Path("/dev/video0"))
    parser.add_argument("--camera-name", default="front")
    parser.add_argument("--side-camera", type=Path, default=Path("/dev/video2"))
    parser.add_argument("--side-camera-name", default="side")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-warmup-s", type=int, default=2)

    parser.add_argument(
        "--task", default="Place the RDK camera box on top of the black MCU box."
    )
    parser.add_argument("--max-chunks", type=int, default=1)
    parser.add_argument(
        "--prefetch-steps",
        type=int,
        default=0,
        help="Set to 0 for strict synchronous inference; positive values enable asynchronous prefetch",
    )
    parser.add_argument("--inter-chunk-settle-s", type=float, default=0.0)
    parser.add_argument(
        "--prefetch-terminal-state",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--prefetch-tracking-lag-steps", type=int, default=5)
    parser.add_argument("--chunk-blend-steps", type=int, default=0)
    parser.add_argument("--visual-refresh-chunks", type=int, default=1)
    parser.add_argument(
        "--retain-step-details", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--result-history-chunks", type=int, default=20)
    parser.add_argument("--save-artifact-every-chunks", type=int, default=1)
    parser.add_argument("--max-first-delta", type=float, default=20.0)
    parser.add_argument("--max-command-delta", type=float, default=20.0)
    parser.add_argument(
        "--stop-on-command-jump",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--gripper-telemetry-interval-steps", type=int, default=30)
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=None,
        help="LeRobot relative target clip in degrees; omit to send exact model targets",
    )
    parser.add_argument(
        "--vision-check", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--vision-action-threshold", type=float, default=1e-6)

    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force-model-actions", action="store_true")
    parser.add_argument("--confirm")
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=SCRIPT_DIR
        / "configs/norm_stats_pi0_full_v5_2cam_100ep_015000.json",
    )
    parser.add_argument(
        "--expected-stats-sha256",
        default="c9a4362acaf822d820ac6871a04184972fd2540d44b4abecce6cd362c04db452",
    )
    parser.add_argument(
        "--expected-calibration-sha256",
        default="dc14138f498fdebd3a333809b01a27c230f46218eaa31a2079c7623b993290b0",
    )
    parser.add_argument(
        "--expected-siglip-sha256",
        default="f954178339b758785f7e1935a92f233d0625b30bd4c070acd86d74f546da058a",
    )
    parser.add_argument(
        "--expected-paligemma-sha256",
        default="7750e1a5cea43c02aec9db6eea317ac91d3ce4e37fc196c503d3935e611cde1e",
    )
    parser.add_argument(
        "--expected-expert-sha256",
        default="db83230022823e53260523dd830f685d38f4d9a72396df7060d11f5cce754d2b",
    )
    parser.add_argument(
        "--expected-prompt-embedding-sha256",
        default="39ac72af2ea87575284a6059cb513287c140593b426c488cb34d135fa552ce87",
    )
    parser.add_argument(
        "--expected-fixed-noise-sha256",
        default="0170c20746b8b8c1268d7d10c8e331ad7555cf050efddab4dc96147275e39dbc",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    def stop_signal(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_signal)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, stop_signal)

    args = parse_args()
    verify_execute_request(args)
    camera_names = (args.camera_name, args.side_camera_name)
    if len(set(camera_names)) != 2:
        raise ValueError("front and side camera names must be different")
    if args.max_chunks < 0:
        raise ValueError("--max-chunks must be >= 0; zero means unlimited")
    if not 0 <= args.prefetch_steps <= ACTION_HORIZON:
        raise ValueError(f"--prefetch-steps must be between 0 and {ACTION_HORIZON}")
    if not 0 <= args.prefetch_tracking_lag_steps < ACTION_HORIZON:
        raise ValueError(
            f"--prefetch-tracking-lag-steps must be between 0 and {ACTION_HORIZON - 1}"
        )
    if not 0 <= args.chunk_blend_steps <= ACTION_HORIZON:
        raise ValueError(
            f"--chunk-blend-steps must be between 0 and {ACTION_HORIZON}"
        )
    if args.gripper_telemetry_interval_steps < 0:
        raise ValueError("--gripper-telemetry-interval-steps must be non-negative")
    if args.inter_chunk_settle_s < 0:
        raise ValueError("--inter-chunk-settle-s must be non-negative")
    if args.visual_refresh_chunks < 0:
        raise ValueError("--visual-refresh-chunks must be non-negative")
    if args.result_history_chunks <= 0:
        raise ValueError("--result-history-chunks must be positive")
    if args.save_artifact_every_chunks < 0:
        raise ValueError("--save-artifact-every-chunks must be >= 0")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or SCRIPT_DIR / "diagnostics" / f"full_pipeline_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "engine.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "pipeline.log"),
        ],
    )

    artifact_hashes = validate_artifacts(args)
    robot = None
    engine_process = None
    engine_log = None
    torque_enabled = False
    result: dict[str, Any] = {
        "mode": "execute" if args.execute else "dry_run",
        "inference_mode": (
            "synchronous" if args.prefetch_steps == 0 else "asynchronous_prefetch"
        ),
        "task": args.task,
        "output_dir": str(output_dir),
        "artifact_hashes": artifact_hashes,
        "fixed_noise": args.fixed_noise,
        "fixed_noise_file": str(args.fixed_noise_file) if args.fixed_noise else None,
        "runtime": "standalone_dnn",
        "engine_runner": str(args.engine_runner),
        "protocol_image_count": len(camera_names),
        "model_visual_slot_count": 3,
        "masked_empty_slot_count": 3 - len(camera_names),
        "camera_names": list(camera_names),
        "action_horizon": ACTION_HORIZON,
        "control_hz": CONTROL_HZ,
        "force_model_actions": args.force_model_actions,
        "prefetch_steps": args.prefetch_steps,
        "inter_chunk_settle_s": args.inter_chunk_settle_s,
        "prefetch_terminal_state": args.prefetch_terminal_state,
        "prefetch_tracking_lag_steps": args.prefetch_tracking_lag_steps,
        "chunk_blend_steps": args.chunk_blend_steps,
        "visual_refresh_chunks": args.visual_refresh_chunks,
        "stop_on_command_jump": args.stop_on_command_jump,
        "gripper_telemetry_interval_steps": args.gripper_telemetry_interval_steps,
        "max_relative_target": args.max_relative_target,
        "retain_step_details": args.retain_step_details,
        "result_history_chunks": args.result_history_chunks,
        "chunk_log": str(output_dir / "chunks.jsonl"),
        "step_log": str(output_dir / "control_steps.jsonl"),
        "chunk_count": 0,
        "chunks": [],
    }

    try:
        robot = make_robot(args)
        connect_components_readonly(robot)
        initial_torque = torque_state(robot)
        result["initial_torque_enable"] = initial_torque
        if any(initial_torque.values()):
            raise RuntimeError(
                f"Refusing startup because SO100 torque is already enabled: {initial_torque}"
            )
        try:
            result["gripper_protection_config"] = read_gripper_registers(
                robot, GRIPPER_CONFIG_REGISTERS
            )
        except Exception as error:
            result["gripper_protection_config_error"] = str(error)
            logging.warning("Could not read gripper protection registers: %s", error)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((args.host, args.port))
            server.listen(1)
            engine_process, engine_log = start_engine(args, log_path)
            conn, address = accept_engine(
                server, engine_process, log_path, args.engine_connect_timeout_s
            )
            logging.info("Standalone Pi0 engine connected from %s", address)

            sequence = 1
            chunk_index = 0
            next_control_tick = None
            last_commanded_target = None
            period_s = 1.0 / CONTROL_HZ
            with (
                conn,
                ThreadPoolExecutor(max_workers=1) as inference_pool,
                (output_dir / "chunks.jsonl").open("a", buffering=1) as chunk_log,
                (output_dir / "control_steps.jsonl").open(
                    "a", buffering=64 * 1024
                ) as step_log,
            ):
                _, state, images = get_live_observation(robot, camera_names)
                actions, inference_ms = infer(
                    conn,
                    state,
                    images,
                    args.task,
                    sequence,
                    reset=True,
                )
                sequence += 1

                vision = None
                vision_pass = True
                if args.vision_check:
                    black_images = [np.zeros_like(image) for image in images]
                    for camera_name, black_image in zip(camera_names, black_images):
                        save_rgb(
                            output_dir / f"vision_black_{camera_name}.jpg",
                            black_image,
                        )
                    black_actions, black_ms = infer(
                        conn,
                        state,
                        black_images,
                        args.task,
                        sequence,
                        reset=True,
                    )
                    sequence += 1
                    np.save(output_dir / "vision_black_actions.npy", black_actions)
                    action_diff = np.abs(actions - black_actions)
                    vision = {
                        "live_inference_ms": inference_ms,
                        "black_inference_ms": black_ms,
                        "mean_abs_action_diff": float(action_diff.mean()),
                        "max_abs_action_diff": float(action_diff.max()),
                    }
                    vision_pass = (
                        vision["max_abs_action_diff"]
                        > args.vision_action_threshold
                    )
                    vision["pass"] = vision_pass
                result["initial_vision_check"] = vision

                while args.max_chunks == 0 or chunk_index < args.max_chunks:
                    command_actions, chunk_blend = stitch_action_chunk(
                        actions,
                        last_commanded_target if args.execute else None,
                        args.chunk_blend_steps,
                    )
                    save_artifacts = (
                        args.save_artifact_every_chunks > 0
                        and chunk_index % args.save_artifact_every_chunks == 0
                    )
                    if save_artifacts:
                        for camera_name, image in zip(camera_names, images):
                            save_rgb(
                                output_dir
                                / f"chunk_{chunk_index:05d}_{camera_name}.jpg",
                                image,
                            )
                        np.save(
                            output_dir / f"chunk_{chunk_index:05d}_actions.npy", actions
                        )
                        if args.execute:
                            np.save(
                                output_dir
                                / f"chunk_{chunk_index:05d}_command_actions.npy",
                                command_actions,
                            )

                    summary = action_summary(actions, state)
                    first_delta_pass = (
                        summary["max_abs_first_delta"] <= args.max_first_delta
                    )
                    chunk_record = {
                        "chunk": chunk_index,
                        "state": state.tolist(),
                        "inference_ms": inference_ms,
                        "actions": summary,
                        "command_actions": action_summary(command_actions, state),
                        "chunk_blend": chunk_blend,
                        "gripper_range_clipped_steps": int(
                            np.count_nonzero(
                                (actions[:, -1] < 0.0) | (actions[:, -1] > 100.0)
                            )
                        ),
                        "vision_check": vision,
                        "first_delta_pass": first_delta_pass,
                        "executed_step_count": 0,
                        "max_abs_delta": 0.0,
                        "max_abs_command_delta": 0.0,
                        "max_control_lag_ms": 0.0,
                        "mean_control_lag_ms": 0.0,
                        "last_current": None,
                        "last_target": None,
                    }
                    if args.retain_step_details:
                        chunk_record["executed_steps"] = []
                    result["chunks"].append(chunk_record)
                    if chunk_index == 0:
                        result["initial_first_delta_pass"] = first_delta_pass

                    logging.info(
                        "chunk=%d inference=%.1fms state=%s action0=%s max_first_delta=%.3f vision_pass=%s",
                        chunk_index,
                        inference_ms,
                        np.round(state, 3).tolist(),
                        np.round(actions[0], 3).tolist(),
                        summary["max_abs_first_delta"],
                        vision_pass,
                    )

                    has_next_chunk = (
                        args.max_chunks == 0 or chunk_index + 1 < args.max_chunks
                    )
                    next_future = None
                    next_state = None
                    next_images = None
                    control_lag_total_ms = 0.0

                    if args.execute:
                        if not torque_enabled:
                            if args.force_model_actions:
                                logging.warning(
                                    "Force-model-actions active: bypassing startup vision and first-delta gates"
                                )
                            else:
                                if not vision_pass:
                                    raise RuntimeError(
                                        "Vision A/B gate failed; refusing to enable SO100 torque"
                                    )
                                if not first_delta_pass:
                                    raise RuntimeError(
                                        "First-action safety gate failed; refusing to enable SO100 torque"
                                    )
                            enable_torque_holding_current(robot, state)
                            torque_enabled = True
                            last_commanded_target = state.copy()
                            logging.warning("SO100 torque enabled")

                        next_control_tick = time.perf_counter()
                        for step_index, target in enumerate(command_actions):
                            sleep_s = next_control_tick - time.perf_counter()
                            if sleep_s > 0:
                                time.sleep(sleep_s)
                            step_started = time.perf_counter()
                            control_lag_ms = float(
                                max(0.0, step_started - next_control_tick) * 1000.0
                            )
                            current = state_from_bus(robot)
                            tracking_delta = np.abs(target - current)
                            if last_commanded_target is None:
                                raise RuntimeError("Missing previous SO100 command state")
                            command_delta = np.abs(target - last_commanded_target)
                            if (
                                not args.force_model_actions
                                and np.max(command_delta) > args.max_command_delta
                            ):
                                message = (
                                    "Command jump safety gate failed at "
                                    f"chunk={chunk_index} step={step_index}: "
                                    f"{command_delta.tolist()}"
                                )
                                if args.stop_on_command_jump:
                                    raise RuntimeError(message)
                                logging.warning("%s; continuing", message)
                            sent = send_target(robot, target)
                            last_commanded_target = target.copy()
                            gripper_telemetry = None
                            global_step_index = (
                                chunk_index * ACTION_HORIZON + step_index
                            )
                            if (
                                args.gripper_telemetry_interval_steps > 0
                                and global_step_index
                                % args.gripper_telemetry_interval_steps
                                == 0
                            ):
                                try:
                                    gripper_telemetry = read_gripper_registers(
                                        robot, GRIPPER_TELEMETRY_REGISTERS
                                    )
                                    if gripper_telemetry["Torque_Enable"] != 1:
                                        logging.error(
                                            "Gripper torque disabled during execution: %s",
                                            gripper_telemetry,
                                        )
                                except Exception as error:
                                    logging.warning(
                                        "Could not read gripper telemetry: %s", error
                                    )
                            step_record = {
                                "chunk": chunk_index,
                                "step": step_index,
                                "current": current.tolist(),
                                "target": target.tolist(),
                                "sent": sent,
                                "max_abs_delta": float(np.max(tracking_delta)),
                                "max_abs_command_delta": float(np.max(command_delta)),
                                "control_lag_ms": control_lag_ms,
                            }
                            if gripper_telemetry is not None:
                                step_record["gripper_telemetry"] = gripper_telemetry
                            step_log.write(
                                json.dumps(step_record, separators=(",", ":")) + "\n"
                            )
                            if args.retain_step_details:
                                chunk_record["executed_steps"].append(step_record)

                            chunk_record["executed_step_count"] = step_index + 1
                            chunk_record["max_abs_delta"] = max(
                                chunk_record["max_abs_delta"],
                                step_record["max_abs_delta"],
                            )
                            chunk_record["max_abs_command_delta"] = max(
                                chunk_record["max_abs_command_delta"],
                                step_record["max_abs_command_delta"],
                            )
                            chunk_record["max_control_lag_ms"] = max(
                                chunk_record["max_control_lag_ms"], control_lag_ms
                            )
                            control_lag_total_ms += control_lag_ms
                            chunk_record["mean_control_lag_ms"] = (
                                control_lag_total_ms / (step_index + 1)
                            )
                            chunk_record["last_current"] = step_record["current"]
                            chunk_record["last_target"] = step_record["target"]

                            remaining_steps = ACTION_HORIZON - step_index - 1
                            if (
                                has_next_chunk
                                and args.prefetch_steps > 0
                                and next_future is None
                                and remaining_steps == args.prefetch_steps
                            ):
                                next_chunk_index = chunk_index + 1
                                refresh_visual = (
                                    args.visual_refresh_chunks > 0
                                    and next_chunk_index % args.visual_refresh_chunks == 0
                                )
                                prefetch_model_action_index = None
                                if args.prefetch_terminal_state:
                                    prefetch_model_action_index = (
                                        ACTION_HORIZON
                                        - 1
                                        - args.prefetch_tracking_lag_steps
                                    )
                                    next_state = actions[
                                        prefetch_model_action_index
                                    ].copy()
                                else:
                                    next_state = current.copy()
                                next_images = read_latest_images(robot, camera_names)
                                next_future = inference_pool.submit(
                                    infer,
                                    conn,
                                    next_state,
                                    next_images,
                                    args.task,
                                    sequence,
                                    refresh_visual,
                                )
                                sequence += 1
                                chunk_record["prefetch_started_step"] = step_index
                                chunk_record["prefetch_model_state"] = next_state.tolist()
                                chunk_record["prefetch_model_action_index"] = (
                                    prefetch_model_action_index
                                )
                                chunk_record["prefetch_visual_refresh"] = refresh_visual

                            next_control_tick += period_s

                        step_log.flush()

                    next_actions = None
                    next_inference_ms = None
                    if has_next_chunk and next_future is not None:
                        wait_started = time.perf_counter()
                        next_actions, next_inference_ms = next_future.result()
                        chunk_record["next_prefetch_wait_ms"] = float(
                            (time.perf_counter() - wait_started) * 1000.0
                        )

                    result["chunk_count"] = chunk_index + 1
                    chunk_log.write(
                        json.dumps(chunk_record, separators=(",", ":")) + "\n"
                    )
                    if len(result["chunks"]) > args.result_history_chunks:
                        result["chunks"] = result["chunks"][
                            -args.result_history_chunks :
                        ]
                    write_json_atomic(output_dir / "result.json", result)

                    chunk_index += 1
                    if not has_next_chunk:
                        break

                    if next_future is not None:
                        state = state_from_bus(robot)
                        images = next_images
                        actions = next_actions
                        inference_ms = next_inference_ms
                    else:
                        if args.inter_chunk_settle_s > 0:
                            logging.info(
                                "Waiting %.3fs for SO100 to settle before next observation",
                                args.inter_chunk_settle_s,
                            )
                            time.sleep(args.inter_chunk_settle_s)
                        _, state, images = get_live_observation(robot, camera_names)
                        next_chunk_index = chunk_index + 1
                        refresh_visual = (
                            args.visual_refresh_chunks > 0
                            and next_chunk_index % args.visual_refresh_chunks == 0
                        )
                        actions, inference_ms = infer(
                            conn,
                            state,
                            images,
                            args.task,
                            sequence,
                            reset=refresh_visual,
                        )
                        sequence += 1
                    vision = None
                    vision_pass = True

        result["completed"] = True
        result["motor_actions_sent"] = bool(args.execute and torque_enabled)
        write_json_atomic(output_dir / "result.json", result)
        logging.info("PIPELINE_COMPLETE output=%s", output_dir)
        if not args.force_model_actions:
            initial_vision = result.get("initial_vision_check")
            if args.vision_check and not initial_vision["pass"]:
                return 2
            if not result.get("initial_first_delta_pass", False):
                return 3
        return 0
    except KeyboardInterrupt:
        result["interrupted"] = True
        logging.warning("Interrupted; shutting down")
        return 130
    except Exception as error:
        result["error"] = str(error)
        logging.exception("PIPELINE_FAILED")
        return 1
    finally:
        if output_dir.exists():
            result["final_torque_disable_requested"] = torque_enabled
            write_json_atomic(output_dir / "result.json", result)
        disconnect_components(robot, torque_enabled)
        stop_engine(engine_process, engine_log)


if __name__ == "__main__":
    raise SystemExit(main())
