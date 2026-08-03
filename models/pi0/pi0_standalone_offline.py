#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from pi0_full_pipeline import build_request, parse_actions, recv_message, send_message


SCRIPT_DIR = Path(__file__).resolve().parent


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one deterministic standalone S600 Pi0 inference without a robot"
    )
    parser.add_argument("--front", type=Path, required=True)
    parser.add_argument("--side", type=Path, required=True)
    parser.add_argument("--state", type=float, nargs=6, required=True)
    parser.add_argument(
        "--task", default="Place the RDK camera box on top of the black MCU box."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=SCRIPT_DIR
        / "configs/deployments/pi0_full_v5_2cam_positionfp16_siglip_fixed16_paligemma_hbmkv_expert_20260801.json",
    )
    parser.add_argument(
        "--engine-runner",
        type=Path,
        default=SCRIPT_DIR / "run_pi0_standalone_config.sh",
    )
    parser.add_argument(
        "--fixed-noise-file",
        type=Path,
        default=SCRIPT_DIR / "configs/fixed_noise_cv_12345678_fp16.bin",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--connect-timeout-s", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (
        args.front,
        args.side,
        args.config,
        args.engine_runner,
        args.fixed_noise_file,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    config = json.loads(args.config.read_text())
    host = str(config["server_ip"])
    port = int(config["server_port"])
    state = np.asarray(args.state, dtype=np.float64)
    images = [load_rgb(args.front), load_rgb(args.side)]

    environment = os.environ.copy()
    environment["PI0_FIXED_NOISE_FILE"] = str(args.fixed_noise_file.resolve())
    engine_log_path = args.output_dir / "engine.log"
    engine_log = engine_log_path.open("w")
    process = None
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((host, port))
            server.listen(1)
            process = subprocess.Popen(
                [str(args.engine_runner), str(args.config)],
                cwd=SCRIPT_DIR,
                env=environment,
                stdout=engine_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + args.connect_timeout_s
            server.settimeout(0.5)
            connection = None
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Engine exited with {process.returncode}:\n"
                        + engine_log_path.read_text(errors="replace")[-8000:]
                    )
                try:
                    connection, _ = server.accept()
                    break
                except socket.timeout:
                    pass
            if connection is None:
                raise TimeoutError("Timed out waiting for standalone Pi0 engine")
            with connection:
                started = time.perf_counter()
                send_message(
                    connection,
                    build_request(state, images, args.task, sequence=1, reset=True),
                )
                actions = parse_actions(recv_message(connection))
                inference_ms = (time.perf_counter() - started) * 1000.0

        np.save(args.output_dir / "actions.npy", actions)
        result = {
            "front": str(args.front.resolve()),
            "side": str(args.side.resolve()),
            "state": state.tolist(),
            "task": args.task,
            "inference_ms": inference_ms,
            "first_action": actions[0].tolist(),
            "shape": list(actions.shape),
        }
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        return 0
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        engine_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
