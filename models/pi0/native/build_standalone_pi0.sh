#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK="${D_ROBOTICS_LLM_SDK_ROOT:-/root/D-Robotics_LLM_S600_1.0.2_SDK/oellm_runtime}"
VENDOR="$ROOT/native/vendor/openexplorer_llm"
SOURCE="$ROOT/native/standalone/pi0_standalone.cc"
PROTO_ROOT="$ROOT/native/sdk_1.0.2/pi0_demo"
OUTPUT_DIR="$ROOT/native/install/bin"
OUTPUT="$OUTPUT_DIR/pi0_standalone_sdk102"
TEMP="$OUTPUT.tmp"

mkdir -p "$OUTPUT_DIR"

g++ -std=c++17 -O3 -g -pthread \
  -include "$ROOT/native/standalone/dnn_sdk102_compat.h" \
  "$SOURCE" \
  "$PROTO_ROOT/proto/msg.pb.cc" \
  "$VENDOR/xlm/src/utils/model_manager.cc" \
  "$VENDOR/xlm/src/utils/xlm_utils.cc" \
  -I"$ROOT/native/standalone" \
  -I"$PROTO_ROOT" \
  -I/usr/include/opencv4 \
  -I/usr/include/eigen3 \
  -I"$SDK/include/protobuf/include" \
  -I"$VENDOR/xlm/include" \
  -I"$ROOT/native/compat" \
  -L"$SDK/lib" \
  -Wl,-rpath,"$SDK/lib" \
  -lprotobuf \
  -ldnn \
  -lhbucp \
  -lhlog \
  -lrt \
  -ldl \
  -lopencv_world \
  -o "$TEMP"

chmod 0755 "$TEMP"
mv -f "$TEMP" "$OUTPUT"
echo "$OUTPUT"
