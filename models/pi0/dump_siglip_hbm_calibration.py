#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import socket
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np


CAMERA_COUNT = 2
ACTION_SIZE = 6
ACTION_HORIZON = 50
VISION_SHAPE = (1, 768, 2048)
VISION_BYTES = int(np.prod(VISION_SHAPE)) * np.dtype(np.float16).itemsize
EMPTY_SLOT_BYTES = 256 * 2048 * np.dtype(np.float16).itemsize
EXPERT_KV_COUNT = 36
EXPERT_KV_SHAPE = (1, 816, 256)
EXPERT_KV_BYTES = int(np.prod(EXPERT_KV_SHAPE)) * np.dtype(np.float16).itemsize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Feed real two-camera calibration samples through the S600 SigLIP HBM "
            "and save the three-slot FP16 embeddings consumed by PaliGemma."
        )
    )
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--engine-dump-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--prompt")
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--dump-timeout-s", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-first-request-dump", action="store_true")
    parser.add_argument("--save-expert-kv", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_sort_key(path: Path) -> tuple[int, int | str]:
    return (0, int(path.name)) if path.name.isdigit() else (1, path.name)


def recv_exact(connection: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            raise ConnectionError("S600 engine disconnected")
        data.extend(chunk)
    return bytes(data)


def recv_message(connection: socket.socket, msg_pb2):
    size = struct.unpack("!I", recv_exact(connection, 4))[0]
    message = msg_pb2.MultiModalInput()
    message.ParseFromString(recv_exact(connection, size))
    return message


def send_message(connection: socket.socket, message) -> None:
    payload = message.SerializeToString()
    connection.sendall(struct.pack("!I", len(payload)) + payload)


def add_tensor(container, dtype: int, shape: tuple[int, ...], data: bytes) -> None:
    tensor = container.add()
    tensor.dtype = dtype
    tensor.shape.extend(shape)
    tensor.data = data


def read_rgb_chw(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


def build_request(
    msg_pb2,
    sample_dir: Path,
    raw_state: np.ndarray,
    prompt: str,
    sequence: int,
):
    request = msg_pb2.MultiModalInput()
    request.header.seq = sequence
    request.header.stamp.sec = int(time.time())
    request.header.frame_id = "pi0_siglip_real_calibration"
    request.header.reset = True

    for camera_index in range(CAMERA_COUNT):
        image = read_rgb_chw(sample_dir / f"image_{camera_index}.jpg")
        add_tensor(request.images, msg_pb2.Tensor.UINT8, image.shape, image.tobytes())

    encoded_prompt = prompt.encode("utf-8")
    add_tensor(
        request.languages,
        msg_pb2.Tensor.STRING,
        (len(encoded_prompt),),
        encoded_prompt,
    )
    state = np.ascontiguousarray(raw_state.astype(np.float64, copy=False))
    add_tensor(request.states, msg_pb2.Tensor.FLOAT64, state.shape, state.tobytes())
    return request


def validate_response(msg_pb2, response) -> None:
    if len(response.states) != 1:
        raise RuntimeError(f"Expected one action tensor, got {len(response.states)}")
    tensor = response.states[0]
    if tensor.dtype != msg_pb2.Tensor.FLOAT64:
        raise RuntimeError(f"Expected FLOAT64 actions, got dtype={tensor.dtype}")
    actions = np.frombuffer(tensor.data, dtype=np.float64)
    expected = ACTION_HORIZON * ACTION_SIZE
    if actions.size != expected:
        raise RuntimeError(f"Expected {expected} action values, got {actions.size}")
    if not np.isfinite(actions).all():
        raise RuntimeError("S600 action response contains NaN or Inf")


def wait_for_dump(path: Path, timeout_s: float, expected_bytes: int = VISION_BYTES) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size == expected_bytes:
            return
        time.sleep(0.05)
    size = path.stat().st_size if path.exists() else None
    raise TimeoutError(f"Timed out waiting for {path}; size={size}")


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    calibration_dir = args.calibration_dir.resolve()
    images_dir = calibration_dir / "images"
    action_dir = calibration_dir / "action"
    source_manifest_path = calibration_dir / "manifest.json"
    if not images_dir.is_dir() or not action_dir.is_dir():
        raise NotADirectoryError(calibration_dir)

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    prompt = args.prompt or source_manifest.get("task")
    if not prompt:
        raise ValueError("A prompt is required")
    if source_manifest.get("camera_keys") != ["front", "side"]:
        raise ValueError(
            f"Expected front/side calibration cameras, got {source_manifest.get('camera_keys')}"
        )

    sample_dirs = sorted(
        [path for path in images_dir.iterdir() if path.is_dir()],
        key=numeric_sort_key,
    )
    if not sample_dirs:
        raise RuntimeError(f"No calibration samples found in {images_dir}")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    engine_dump_dir = args.engine_dump_dir.resolve()
    engine_dump_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import msg_pb2

    records: list[dict] = []
    empty_slot_hashes: set[str] = set()
    request_index = 0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        if args.ready_file is not None:
            args.ready_file.parent.mkdir(parents=True, exist_ok=True)
            args.ready_file.write_text("ready\n", encoding="utf-8")
        print(
            f"SIGLIP_CALIBRATION_SERVER_READY host={args.host} port={args.port} "
            f"samples={len(sample_dirs)}",
            flush=True,
        )
        connection, address = server.accept()
        print(f"S600_ENGINE_CONNECTED address={address}", flush=True)
        with connection:
            for sample_index, sample_dir in enumerate(sample_dirs):
                sample_name = sample_dir.name
                sample_output_dir = output_dir / sample_name
                output_path = sample_output_dir / "paligemma_inputs_embeds.bin"
                if args.resume and output_path.is_file():
                    if output_path.stat().st_size != VISION_BYTES:
                        raise ValueError(f"Invalid existing embedding: {output_path}")
                    with output_path.open("rb") as stream:
                        stream.seek(VISION_BYTES - EMPTY_SLOT_BYTES)
                        empty_hash = hashlib.sha256(stream.read()).hexdigest()
                    empty_slot_hashes.add(empty_hash)
                    records.append(
                        {
                            "sample": sample_name,
                            "status": "reused",
                            "embedding_sha256": sha256_file(output_path),
                            "empty_slot_sha256": empty_hash,
                        }
                    )
                    continue

                raw_state_path = action_dir / sample_name / "raw_state.npy"
                raw_state = np.load(raw_state_path).reshape(-1)
                if raw_state.shape != (ACTION_SIZE,):
                    raise ValueError(
                        f"Expected six raw state values in {raw_state_path}, got {raw_state.shape}"
                    )

                request = build_request(
                    msg_pb2,
                    sample_dir,
                    raw_state,
                    prompt,
                    sequence=sample_index + 1,
                )
                send_message(connection, request)
                response = recv_message(connection, msg_pb2)
                validate_response(msg_pb2, response)

                request_dump_dir = engine_dump_dir / f"request_{request_index:06d}"
                source_path = request_dump_dir / "paligemma_vision_fp16.bin"
                wait_for_dump(source_path, args.dump_timeout_s)
                kv_source_paths = [
                    request_dump_dir / f"expert_kv_{index:02d}_fp16.bin"
                    for index in range(EXPERT_KV_COUNT)
                ]
                if args.save_expert_kv:
                    for kv_source_path in kv_source_paths:
                        wait_for_dump(
                            kv_source_path,
                            args.dump_timeout_s,
                            expected_bytes=EXPERT_KV_BYTES,
                        )
                sample_output_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path, output_path)
                embedding_sha256 = sha256_file(output_path)
                with output_path.open("rb") as stream:
                    stream.seek(VISION_BYTES - EMPTY_SLOT_BYTES)
                    empty_hash = hashlib.sha256(stream.read()).hexdigest()
                empty_slot_hashes.add(empty_hash)
                kv_hashes = []
                if args.save_expert_kv:
                    for kv_index, kv_source_path in enumerate(kv_source_paths):
                        kv_output_path = (
                            sample_output_dir / f"expert_kv_{kv_index:02d}_fp16.bin"
                        )
                        if kv_source_path.stat().st_size != EXPERT_KV_BYTES:
                            raise ValueError(
                                f"Invalid Expert KV dump size: {kv_source_path}"
                            )
                        shutil.copyfile(kv_source_path, kv_output_path)
                        kv_hashes.append(sha256_file(kv_output_path))
                records.append(
                    {
                        "sample": sample_name,
                        "status": "generated",
                        "request_index": request_index,
                        "source_images": [
                            {
                                "path": str(sample_dir / f"image_{camera_index}.jpg"),
                                "sha256": sha256_file(
                                    sample_dir / f"image_{camera_index}.jpg"
                                ),
                            }
                            for camera_index in range(CAMERA_COUNT)
                        ],
                        "raw_state_path": str(raw_state_path),
                        "raw_state": raw_state.astype(float).tolist(),
                        "embedding_sha256": embedding_sha256,
                        "empty_slot_sha256": empty_hash,
                    }
                )
                if args.save_expert_kv:
                    records[-1]["expert_kv_sha256"] = kv_hashes
                if not (args.keep_first_request_dump and request_index == 0):
                    shutil.rmtree(request_dump_dir)
                request_index += 1
                print(
                    f"SAMPLE_COMPLETE {sample_index + 1}/{len(sample_dirs)} "
                    f"sample={sample_name} sha256={embedding_sha256}",
                    flush=True,
                )
                write_json_atomic(
                    output_dir / "manifest.partial.json",
                    {
                        "format": "pi0_s600_siglip_hbm_calibration_v1",
                        "completed": len(records),
                        "sample_count": len(sample_dirs),
                        "records": records,
                    },
                )

    if len(empty_slot_hashes) != 1:
        raise RuntimeError(
            f"Masked empty-camera embedding changed across samples: {sorted(empty_slot_hashes)}"
        )
    manifest = {
        "format": "pi0_s600_siglip_hbm_calibration_v1",
        "source_calibration_dir": str(calibration_dir),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "prompt": prompt,
        "protocol_real_camera_count": CAMERA_COUNT,
        "model_camera_slots": 3,
        "embedding_shape": list(VISION_SHAPE),
        "embedding_dtype": "float16",
        "embedding_bytes": VISION_BYTES,
        "sample_count": len(sample_dirs),
        "generated_request_count": request_index,
        "empty_slot_sha256": next(iter(empty_slot_hashes)),
        "expert_kv": {
            "saved": args.save_expert_kv,
            "count": EXPERT_KV_COUNT,
            "shape": list(EXPERT_KV_SHAPE),
            "dtype": "float16",
            "bytes_per_tensor": EXPERT_KV_BYTES,
        },
        "records": records,
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    partial_manifest = output_dir / "manifest.partial.json"
    if partial_manifest.exists():
        partial_manifest.unlink()
    print(json.dumps({key: manifest[key] for key in manifest if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
