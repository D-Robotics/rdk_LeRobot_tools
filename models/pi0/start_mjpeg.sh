#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$ROOT/mjpeg_server.pid"
LOG_FILE="$ROOT/mjpeg_server.log"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "MJPEG server already running: PID $(cat "$PID_FILE")"
  exit 0
fi
rm -f "$PID_FILE"
nohup python3 "$ROOT/mjpeg_server.py" --host 0.0.0.0 --port 8080 --device /dev/video0 --width 640 --height 480 --fps 15 >"$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
for _ in $(seq 1 50); do
  grep -q MJPEG_SERVER_READY "$LOG_FILE" && break
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    cat "$LOG_FILE"
    exit 1
  fi
  sleep 0.1
done
cat "$LOG_FILE"
