#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$ROOT/mjpeg_server.pid"
if [[ ! -f "$PID_FILE" ]]; then
  echo "MJPEG server is not running"
  exit 0
fi
pid="$(cat "$PID_FILE")"
kill "$pid" 2>/dev/null || true
for _ in $(seq 1 30); do
  kill -0 "$pid" 2>/dev/null || break
  sleep 0.1
done
rm -f "$PID_FILE"
echo "MJPEG server stopped"
