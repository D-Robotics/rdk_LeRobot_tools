#!/usr/bin/env python3
import argparse
import html
import socketserver
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler

import cv2


class Camera:
    def __init__(self, device: str, width: int, height: int, fps: int, quality: int):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.quality = quality
        self.condition = threading.Condition()
        self.frame = None
        self.error = None
        self.running = True
        self.thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.thread.start()

    def open_camera(self):
        capture = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open {self.device}")
        return capture

    def capture_loop(self):
        capture = None
        while self.running:
            try:
                if capture is None or not capture.isOpened():
                    capture = self.open_camera()
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError("Camera frame read failed")
                ok, encoded = cv2.imencode(
                    ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.quality]
                )
                if not ok:
                    raise RuntimeError("JPEG encoding failed")
                with self.condition:
                    self.frame = encoded.tobytes()
                    self.error = None
                    self.condition.notify_all()
            except Exception as exc:
                self.error = str(exc)
                if capture is not None:
                    capture.release()
                    capture = None
                time.sleep(1)
        if capture is not None:
            capture.release()

    def wait_frame(self, previous):
        with self.condition:
            self.condition.wait_for(
                lambda: not self.running or (self.frame is not None and self.frame is not previous),
                timeout=5,
            )
            return self.frame


class ThreadedHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    camera: Camera = None

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>S600 SO100 Front Camera</title>
<style>body{{margin:0;background:#111;color:#eee;font-family:sans-serif;text-align:center}}
h1{{font-size:20px}}img{{max-width:100vw;max-height:calc(100vh - 70px);object-fit:contain}}
small{{color:#aaa}}</style></head>
<body><h1>S600 SO100 Front Camera</h1><img src="/stream.mjpg">
<br><small>{html.escape(self.camera.device)} · {self.camera.width}×{self.camera.height} · target {self.camera.fps} FPS</small></body></html>"""
            body = page.encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/snapshot.jpg":
            frame = self.camera.wait_frame(None)
            if frame is None:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, self.camera.error or "No frame")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(frame)
            return
        if self.path == "/stream.mjpg":
            self.send_response(HTTPStatus.OK)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            previous = None
            try:
                while True:
                    frame = self.camera.wait_frame(previous)
                    if frame is None:
                        continue
                    previous = frame
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, fmt, *args):
        print(f"{self.client_address[0]} - {fmt % args}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=80)
    args = parser.parse_args()
    camera = Camera(args.device, args.width, args.height, args.fps, args.quality)
    Handler.camera = camera
    with ThreadedHTTPServer((args.host, args.port), Handler) as server:
        print(f"MJPEG_SERVER_READY http://{args.host}:{args.port}", flush=True)
        try:
            server.serve_forever()
        finally:
            camera.running = False


if __name__ == "__main__":
    main()
