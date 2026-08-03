#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import logging
import signal
import socket
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors


MAX_REQUEST_BYTES = 4 * 1024 * 1024


class Pi0Runtime:
    def __init__(
        self,
        checkpoint: Path,
        device: str,
        dtype: str,
        fixed_noise_file: Path | None,
        camera_keys: list[str],
    ) -> None:
        self.checkpoint = checkpoint.resolve()
        self.device = device
        self.dtype = dtype
        self.lock = threading.Lock()
        self.request_count = 0
        self.last_inference_ms: float | None = None
        if not camera_keys or len(set(camera_keys)) != len(camera_keys):
            raise ValueError("--camera-keys must contain unique camera names")
        self.camera_keys = tuple(camera_keys)

        config = PreTrainedConfig.from_pretrained(self.checkpoint)
        config.device = device
        config.dtype = dtype
        config.gradient_checkpointing = False
        config.compile_model = False
        self.config = config

        policy_class = get_policy_class(config.type)
        load_started = time.perf_counter()
        self.policy = (
            policy_class.from_pretrained(self.checkpoint, config=config)
            .to(device)
            .eval()
        )
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config, pretrained_path=self.checkpoint
        )
        self.load_seconds = time.perf_counter() - load_started

        self.fixed_noise: torch.Tensor | None = None
        if fixed_noise_file is not None:
            values = np.fromfile(fixed_noise_file, dtype=np.float16)
            expected = config.chunk_size * config.max_action_dim
            if values.size != expected:
                raise ValueError(
                    f"Fixed noise has {values.size} values; expected {expected}"
                )
            values = values.reshape(1, config.chunk_size, config.max_action_dim)
            self.fixed_noise = torch.from_numpy(values.astype(np.float32)).to(device)

        logging.info(
            "Loaded original Pi0 checkpoint=%s device=%s dtype=%s in %.3fs",
            self.checkpoint,
            self.device,
            self.dtype,
            self.load_seconds,
        )

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "checkpoint": str(self.checkpoint),
            "device": self.device,
            "dtype": self.dtype,
            "load_seconds": self.load_seconds,
            "request_count": self.request_count,
            "last_inference_ms": self.last_inference_ms,
            "fixed_noise_available": self.fixed_noise is not None,
            "chunk_size": self.config.chunk_size,
            "action_dim": 6,
            "camera_keys": list(self.camera_keys),
        }

    def _decode_image(self, encoded: str, camera_key: str) -> torch.Tensor:
        compressed = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
        image_bgr = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Failed to decode camera image: {camera_key}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(image_rgb.copy()).permute(2, 0, 1).float() / 255.0

    def _make_noise(self, mode: str, seed: int | None) -> torch.Tensor | None:
        if mode == "random":
            if seed is None:
                return None
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            return torch.randn(
                (1, self.config.chunk_size, self.config.max_action_dim),
                generator=generator,
                device=self.device,
                dtype=torch.float32,
            )
        if mode == "fixed":
            if self.fixed_noise is None:
                raise ValueError("Server was not started with --fixed-noise-file")
            return self.fixed_noise.clone()
        raise ValueError(f"Unsupported noise_mode: {mode}")

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_started = time.perf_counter()
        state = np.asarray(payload.get("state"), dtype=np.float32)
        if state.shape != (6,) or not np.isfinite(state).all():
            raise ValueError(f"Expected finite state shape (6,), got {state.shape}")
        task = payload.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        encoded_images = payload.get("images_jpeg_b64")
        if encoded_images is None and len(self.camera_keys) == 1:
            encoded_images = {self.camera_keys[0]: payload.get("image_jpeg_b64")}
        if not isinstance(encoded_images, dict):
            raise ValueError("images_jpeg_b64 must be a camera-name mapping")
        missing_cameras = [
            key for key in self.camera_keys if not isinstance(encoded_images.get(key), str)
        ]
        if missing_cameras:
            raise ValueError(f"Missing camera JPEG data: {missing_cameras}")
        noise_mode = str(payload.get("noise_mode", "random"))
        noise_seed = payload.get("noise_seed")
        if noise_seed is not None:
            noise_seed = int(noise_seed)

        images = {
            key: self._decode_image(encoded_images[key], key) for key in self.camera_keys
        }
        observation = {
            **{
                f"observation.images.{key}": image.unsqueeze(0)
                for key, image in images.items()
            },
            "observation.state": torch.from_numpy(state).unsqueeze(0),
            "task": [task],
        }

        with self.lock:
            batch = self.preprocessor(observation)
            noise = self._make_noise(noise_mode, noise_seed)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            inference_started = time.perf_counter()
            with torch.inference_mode():
                normalized_actions = self.policy.predict_action_chunk(batch, noise=noise)
                action_tensor = self.postprocessor(normalized_actions)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            inference_ms = (time.perf_counter() - inference_started) * 1000.0

        actions = action_tensor.detach().cpu().float().numpy()
        if actions.shape == (1, self.config.chunk_size, 6):
            actions = actions[0]
        expected_shape = (self.config.chunk_size, 6)
        if actions.shape != expected_shape:
            raise ValueError(f"Expected action shape {expected_shape}, got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("Model produced NaN or Inf")

        self.request_count += 1
        self.last_inference_ms = inference_ms
        return {
            "request_id": payload.get("request_id"),
            "actions": actions.tolist(),
            "shape": list(actions.shape),
            "noise_mode": noise_mode,
            "noise_seed": noise_seed,
            "camera_keys": list(self.camera_keys),
            "camera_shapes": {
                key: list(image.shape) for key, image in images.items()
            },
            "inference_ms": inference_ms,
            "total_ms": (time.perf_counter() - request_started) * 1000.0,
            "first_delta": (actions[0] - state).tolist(),
            "max_abs_first_delta": float(np.max(np.abs(actions[0] - state))),
        }


class Pi0RequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "Pi0TorchServer/1.0"

    @property
    def runtime(self) -> Pi0Runtime:
        return self.server.runtime

    @property
    def auth_token(self) -> str:
        return self.server.auth_token

    def log_message(self, format_string: str, *args: Any) -> None:
        logging.info("%s - %s", self.client_address[0], format_string % args)

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        self._write_json(HTTPStatus.OK, self.runtime.health())

    def do_POST(self) -> None:
        if self.path != "/infer":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if self.headers.get("Authorization") != f"Bearer {self.auth_token}":
            self._write_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
                raise ValueError(f"Invalid Content-Length: {content_length}")
            payload = json.loads(self.rfile.read(content_length))
            response = self.runtime.infer(payload)
            logging.info(
                "request=%s inference=%.1fms total=%.1fms first_delta=%.3f",
                response.get("request_id"),
                response["inference_ms"],
                response["total_ms"],
                response["max_abs_first_delta"],
            )
            self._write_json(HTTPStatus.OK, response)
        except Exception as error:
            logging.exception("Inference request failed")
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": f"{type(error).__name__}: {error}"},
            )


class Pi0HttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        runtime: Pi0Runtime,
        auth_token: str,
    ) -> None:
        super().__init__(address, Pi0RequestHandler)
        self.runtime = runtime
        self.auth_token = auth_token

    def get_request(self):
        request, address = super().get_request()
        request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return request, address


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the original unquantized LeRobot Pi0 policy over HTTP"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--fixed-noise-file", type=Path)
    parser.add_argument("--camera-keys", nargs="+", default=["front"])
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )

    auth_token = args.token_file.read_text().strip()
    if not auth_token:
        raise ValueError("Token file is empty")
    runtime = Pi0Runtime(
        checkpoint=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        fixed_noise_file=args.fixed_noise_file,
        camera_keys=args.camera_keys,
    )
    server = Pi0HttpServer((args.host, args.port), runtime, auth_token)

    def stop_server(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)
    logging.info("Listening on http://%s:%d", args.host, args.port)
    server.serve_forever(poll_interval=0.2)
    server.server_close()
    logging.info("Server stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
