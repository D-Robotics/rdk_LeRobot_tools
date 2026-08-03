#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import http.client
import json
import logging
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import cv2
import numpy as np

from pi0_full_pipeline import (
    ACTION_HORIZON,
    MOTOR_CONFIRMATION,
    MOTOR_NAMES,
    connect_components_readonly,
    disconnect_components,
    enable_torque_holding_current,
    get_live_observation,
    make_robot,
    save_rgb,
    send_target,
    state_from_bus,
    torque_state,
    write_json_atomic,
)


SCRIPT_DIR = Path(__file__).resolve().parent


class RemotePi0:
    def __init__(self, server_url: str, token: str, timeout_s: float) -> None:
        parsed = urlsplit(server_url)
        if parsed.scheme != "http" or parsed.hostname is None:
            raise ValueError("--server-url must be an http URL")
        self.server_url = server_url.rstrip("/")
        self.base_path = parsed.path.rstrip("/")
        self.host = parsed.hostname
        self.port = parsed.port or 80
        self.token = token
        self.timeout_s = timeout_s
        self.lock = threading.Lock()
        self.connection = self._new_connection()

    def _new_connection(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_s)

    def _request(
        self, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        data: bytes | None = None
        headers = {"Accept": "application/json"}
        method = "GET"
        if payload is not None:
            method = "POST"
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
            headers["Authorization"] = f"Bearer {self.token}"
        headers["Connection"] = "keep-alive"
        with self.lock:
            for attempt in range(2):
                try:
                    self.connection.request(
                        method, f"{self.base_path}{path}", body=data, headers=headers
                    )
                    response = self.connection.getresponse()
                    body = response.read()
                    if response.status >= 400:
                        text = body.decode("utf-8", errors="replace")
                        raise RuntimeError(
                            f"Remote Pi0 HTTP {response.status}: {text}"
                        )
                    return json.loads(body)
                except (BrokenPipeError, ConnectionError, http.client.HTTPException):
                    self.connection.close()
                    if attempt == 1:
                        raise
                    self.connection = self._new_connection()
        raise RuntimeError("Remote Pi0 request failed")

    def health(self) -> dict[str, Any]:
        return self._request("/health")

    def infer(
        self,
        state: np.ndarray,
        images_rgb: dict[str, np.ndarray],
        task: str,
        request_id: int,
        noise_mode: str,
        noise_seed: int | None,
        jpeg_quality: int,
        wire_width: int,
        wire_height: int,
    ) -> tuple[np.ndarray, dict[str, Any], float]:
        if not images_rgb:
            raise ValueError("At least one camera image is required")
        encoded_images: dict[str, str] = {}
        wire_image_shapes: dict[str, list[int]] = {}
        wire_jpeg_bytes: dict[str, int] = {}
        for camera_key, image_rgb in images_rgb.items():
            wire_image = np.asarray(image_rgb)
            if wire_width > 0 and wire_height > 0:
                wire_image = cv2.resize(
                    wire_image,
                    (wire_width, wire_height),
                    interpolation=cv2.INTER_AREA,
                )
            image_bgr = cv2.cvtColor(wire_image, cv2.COLOR_RGB2BGR)
            encoded_ok, encoded = cv2.imencode(
                ".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            if not encoded_ok:
                raise RuntimeError(f"Failed to JPEG-encode camera: {camera_key}")
            encoded_images[camera_key] = base64.b64encode(encoded).decode("ascii")
            wire_image_shapes[camera_key] = list(wire_image.shape)
            wire_jpeg_bytes[camera_key] = int(encoded.size)
        payload = {
            "request_id": request_id,
            "task": task,
            "state": np.asarray(state, dtype=np.float32).tolist(),
            "images_jpeg_b64": encoded_images,
            "noise_mode": noise_mode,
            "noise_seed": noise_seed,
        }
        started = time.perf_counter()
        response = self._request("/infer", payload)
        round_trip_ms = (time.perf_counter() - started) * 1000.0
        response["wire_image_shapes"] = wire_image_shapes
        response["wire_jpeg_bytes"] = wire_jpeg_bytes
        actions = np.asarray(response["actions"], dtype=np.float64)
        expected = (ACTION_HORIZON, len(MOTOR_NAMES))
        if actions.shape != expected:
            raise ValueError(f"Unexpected remote action shape: {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("Remote actions contain NaN or Inf")
        return actions, response, round_trip_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SO100 with original Pi0 inference hosted on a GPU server"
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:31001")
    parser.add_argument(
        "--token-file", type=Path, default=SCRIPT_DIR / "configs/remote_pi0_token"
    )
    parser.add_argument("--request-timeout-s", type=float, default=10.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--wire-width", type=int, default=640)
    parser.add_argument("--wire-height", type=int, default=480)
    parser.add_argument("--noise-mode", choices=("random", "fixed"), default="random")
    parser.add_argument("--noise-seed", type=int)

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
    parser.add_argument("--camera", type=Path, default=Path("/dev/video0"))
    parser.add_argument("--camera-name", default="front")
    parser.add_argument("--side-camera", type=Path)
    parser.add_argument("--side-camera-name", default="side")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-warmup-s", type=int, default=2)
    parser.add_argument(
        "--task", default="Place the RDK camera box on top of the black MCU box."
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=1,
        help="Number of action chunks to run; 0 means unlimited until SIGTERM/KeyboardInterrupt",
    )
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument(
        "--prefetch-remaining-steps",
        type=int,
        default=0,
        help="0 uses exact boundary observations with a short inference pause",
    )
    parser.add_argument(
        "--prefetch-terminal-state",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-relative-target", type=float, default=None)
    parser.add_argument("--save-artifact-every-chunks", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    return parser.parse_args()


def main() -> int:
    def stop_signal(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_signal)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, stop_signal)

    args = parse_args()
    if args.execute and args.confirm != MOTOR_CONFIRMATION:
        raise RuntimeError(f"Motor execution requires --confirm {MOTOR_CONFIRMATION}")
    if args.control_hz <= 0:
        raise ValueError("--control-hz must be positive")
    if not 0 <= args.prefetch_remaining_steps < ACTION_HORIZON:
        raise ValueError("--prefetch-remaining-steps must be in [0, 49]")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100]")
    if args.wire_width <= 0 or args.wire_height <= 0:
        raise ValueError("--wire-width and --wire-height must be positive")
    if args.side_camera is not None and args.side_camera_name == args.camera_name:
        raise ValueError("Front and side camera names must differ")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        SCRIPT_DIR / "diagnostics" / f"remote_torch_pi0_{stamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "pipeline.log"),
        ],
    )

    token = args.token_file.read_text().strip()
    if not token:
        raise ValueError("Remote Pi0 token file is empty")
    remote = RemotePi0(args.server_url, token, args.request_timeout_s)
    health = remote.health()
    logging.info("Remote Pi0 health: %s", health)
    camera_names = [args.camera_name]
    if args.side_camera is not None:
        camera_names.append(args.side_camera_name)
    server_camera_keys = health.get("camera_keys")
    if server_camera_keys is not None and server_camera_keys != camera_names:
        raise RuntimeError(
            f"Server camera keys {server_camera_keys} do not match client {camera_names}"
        )

    result: dict[str, Any] = {
        "mode": "execute" if args.execute else "dry_run",
        "server_url": args.server_url,
        "server_health": health,
        "task": args.task,
        "noise_mode": args.noise_mode,
        "noise_seed": args.noise_seed,
        "control_hz": args.control_hz,
        "action_horizon": ACTION_HORIZON,
        "prefetch_remaining_steps": args.prefetch_remaining_steps,
        "prefetch_terminal_state": args.prefetch_terminal_state,
        "max_relative_target": args.max_relative_target,
        "camera_names": camera_names,
        "output_dir": str(output_dir),
        "chunks": [],
        "chunk_count": 0,
    }
    robot = None
    torque_enabled = False
    request_id = 1

    try:
        robot = make_robot(args)
        connect_components_readonly(robot)
        initial_torque = torque_state(robot)
        result["initial_torque_enable"] = initial_torque
        if any(initial_torque.values()):
            raise RuntimeError(
                f"Refusing startup because SO100 torque is already enabled: {initial_torque}"
            )

        observation = robot.get_observation()
        state = np.asarray(
            [observation[f"{name}.pos"] for name in MOTOR_NAMES], dtype=np.float64
        )
        images = {
            camera_name: np.asarray(observation[camera_name]).copy()
            for camera_name in camera_names
        }
        actions, remote_meta, round_trip_ms = remote.infer(
            state,
            images,
            args.task,
            request_id,
            args.noise_mode,
            args.noise_seed,
            args.jpeg_quality,
            args.wire_width,
            args.wire_height,
        )
        request_id += 1

        with (
            ThreadPoolExecutor(max_workers=1) as inference_pool,
            (output_dir / "chunks.jsonl").open("a", buffering=1) as chunk_log,
            (output_dir / "control_steps.jsonl").open(
                "a", buffering=64 * 1024
            ) as step_log,
        ):
            chunk_index = 0
            while args.max_chunks <= 0 or chunk_index < args.max_chunks:
                if (
                    args.save_artifact_every_chunks > 0
                    and chunk_index % args.save_artifact_every_chunks == 0
                ):
                    for camera_name, image in images.items():
                        save_rgb(
                            output_dir / f"chunk_{chunk_index:05d}_{camera_name}.jpg",
                            image,
                        )
                    np.save(
                        output_dir / f"chunk_{chunk_index:05d}_actions.npy", actions
                    )

                first_delta = actions[0] - state
                chunk_record: dict[str, Any] = {
                    "chunk": chunk_index,
                    "state": state.tolist(),
                    "remote": remote_meta,
                    "round_trip_ms": round_trip_ms,
                    "actions": {
                        "shape": list(actions.shape),
                        "first": actions[0].tolist(),
                        "last": actions[-1].tolist(),
                        "first_delta": first_delta.tolist(),
                        "max_abs_first_delta": float(np.max(np.abs(first_delta))),
                        "min": actions.min(axis=0).tolist(),
                        "max": actions.max(axis=0).tolist(),
                    },
                    "executed_step_count": 0,
                    "max_control_lag_ms": 0.0,
                    "mean_control_lag_ms": 0.0,
                }
                result["chunks"].append(chunk_record)
                logging.info(
                    "chunk=%d server=%.1fms roundtrip=%.1fms state=%s action0=%s delta=%.3f",
                    chunk_index,
                    remote_meta["inference_ms"],
                    round_trip_ms,
                    np.round(state, 3).tolist(),
                    np.round(actions[0], 3).tolist(),
                    chunk_record["actions"]["max_abs_first_delta"],
                )

                has_next_chunk = args.max_chunks <= 0 or chunk_index + 1 < args.max_chunks
                next_future = None
                next_images = None
                control_lag_total_ms = 0.0

                if args.execute:
                    if not torque_enabled:
                        enable_torque_holding_current(robot, state)
                        torque_enabled = True
                        logging.warning(
                            "SO100 torque enabled; exact remote model actions active"
                        )

                    period_s = 1.0 / args.control_hz
                    next_control_tick = time.perf_counter()
                    for step_index, target in enumerate(actions):
                        sleep_s = next_control_tick - time.perf_counter()
                        if sleep_s > 0:
                            time.sleep(sleep_s)
                        step_started = time.perf_counter()
                        control_lag_ms = float(
                            max(0.0, step_started - next_control_tick) * 1000.0
                        )
                        current = state_from_bus(robot)
                        sent = send_target(robot, target)
                        step_record = {
                            "chunk": chunk_index,
                            "step": step_index,
                            "current": current.tolist(),
                            "target": target.tolist(),
                            "sent": sent,
                            "max_abs_delta": float(np.max(np.abs(target - current))),
                            "control_lag_ms": control_lag_ms,
                        }
                        step_log.write(
                            json.dumps(step_record, separators=(",", ":")) + "\n"
                        )
                        chunk_record["executed_step_count"] = step_index + 1
                        chunk_record["last_current"] = step_record["current"]
                        chunk_record["last_target"] = step_record["target"]
                        chunk_record["max_control_lag_ms"] = max(
                            chunk_record["max_control_lag_ms"], control_lag_ms
                        )
                        control_lag_total_ms += control_lag_ms
                        chunk_record["mean_control_lag_ms"] = (
                            control_lag_total_ms / (step_index + 1)
                        )

                        remaining_steps = ACTION_HORIZON - step_index - 1
                        if (
                            has_next_chunk
                            and args.prefetch_remaining_steps > 0
                            and remaining_steps == args.prefetch_remaining_steps
                            and next_future is None
                        ):
                            prefetch_state = (
                                actions[-1].copy()
                                if args.prefetch_terminal_state
                                else current.copy()
                            )
                            next_images = {
                                camera_name: np.asarray(
                                    robot.cameras[camera_name].read_latest()
                                ).copy()
                                for camera_name in camera_names
                            }
                            seed = (
                                None
                                if args.noise_seed is None
                                else args.noise_seed + request_id - 1
                            )
                            next_future = inference_pool.submit(
                                remote.infer,
                                prefetch_state,
                                next_images,
                                args.task,
                                request_id,
                                args.noise_mode,
                                seed,
                                args.jpeg_quality,
                                args.wire_width,
                                args.wire_height,
                            )
                            chunk_record["prefetch_started_step"] = step_index
                            chunk_record["prefetch_state"] = prefetch_state.tolist()
                            request_id += 1

                        next_control_tick += period_s

                    step_log.flush()

                result["chunk_count"] = chunk_index + 1
                chunk_log.write(
                    json.dumps(chunk_record, separators=(",", ":")) + "\n"
                )
                write_json_atomic(output_dir / "result.json", result)
                if not has_next_chunk:
                    break

                if next_future is not None:
                    wait_started = time.perf_counter()
                    actions, remote_meta, round_trip_ms = next_future.result()
                    chunk_record["next_prefetch_wait_ms"] = (
                        time.perf_counter() - wait_started
                    ) * 1000.0
                    state = state_from_bus(robot)
                    images = next_images
                else:
                    observation = robot.get_observation()
                    state = np.asarray(
                        [observation[f"{name}.pos"] for name in MOTOR_NAMES],
                        dtype=np.float64,
                    )
                    images = {
                        camera_name: np.asarray(observation[camera_name]).copy()
                        for camera_name in camera_names
                    }
                    seed = (
                        None
                        if args.noise_seed is None
                        else args.noise_seed + request_id - 1
                    )
                    actions, remote_meta, round_trip_ms = remote.infer(
                        state,
                        images,
                        args.task,
                        request_id,
                        args.noise_mode,
                        seed,
                        args.jpeg_quality,
                        args.wire_width,
                        args.wire_height,
                    )
                    request_id += 1
                chunk_index += 1

        result["completed"] = True
        result["motor_actions_sent"] = bool(args.execute and torque_enabled)
        write_json_atomic(output_dir / "result.json", result)
        logging.info("PIPELINE_COMPLETE output=%s", output_dir)
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
        result["final_torque_disable_requested"] = torque_enabled
        write_json_atomic(output_dir / "result.json", result)
        disconnect_components(robot, torque_enabled)


if __name__ == "__main__":
    raise SystemExit(main())
