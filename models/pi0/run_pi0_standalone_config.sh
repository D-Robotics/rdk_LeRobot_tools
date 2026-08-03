#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK="${D_ROBOTICS_LLM_SDK_ROOT:-/root/D-Robotics_LLM_S600_1.0.2_SDK/oellm_runtime}"
CONFIG="${1:?usage: run_pi0_standalone_config.sh CONFIG_JSON [ENGINE_ARGS...]}"
shift
ENGINE="${PI0_STANDALONE_BIN:-$ROOT/native/install/bin/pi0_standalone_sdk102}"

python3 "$ROOT/validate_pi0_config.py" "$CONFIG"
[[ -x "$ENGINE" ]]

export LD_LIBRARY_PATH="$SDK/lib:${LD_LIBRARY_PATH:-}"
export HB_DNN_USER_DEFINED_L2M_SIZES="${HB_DNN_USER_DEFINED_L2M_SIZES:-6:6:6:6}"

cd "$ROOT"
exec "$ENGINE" --config "$CONFIG" "$@"
