#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_dir="$root/diagnostics/live_final_fixed16_sync_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$run_dir"
printf '%s\n' "$$" > "$run_dir/pipeline.pid"

cd "$root"
exec /home/sunrise/lerobot/.venv/bin/python -u pi0_full_pipeline.py \
  --config configs/deployments/pi0_full_v5_2cam_positionfp16_siglip_fixed16_paligemma_hbmkv_expert_20260801.json \
  --expected-stats-sha256 c9a4362acaf822d820ac6871a04184972fd2540d44b4abecce6cd362c04db452 \
  --expected-calibration-sha256 dc14138f498fdebd3a333809b01a27c230f46218eaa31a2079c7623b993290b0 \
  --expected-siglip-sha256 f954178339b758785f7e1935a92f233d0625b30bd4c070acd86d74f546da058a \
  --expected-paligemma-sha256 7750e1a5cea43c02aec9db6eea317ac91d3ce4e37fc196c503d3935e611cde1e \
  --expected-expert-sha256 db83230022823e53260523dd830f685d38f4d9a72396df7060d11f5cce754d2b \
  --expected-prompt-embedding-sha256 39ac72af2ea87575284a6059cb513287c140593b426c488cb34d135fa552ce87 \
  --max-chunks 0 \
  --prefetch-steps 0 \
  --no-fixed-noise \
  --save-artifact-every-chunks 0 \
  --output-dir "$run_dir/output" \
  --execute \
  --force-model-actions \
  --confirm ENABLE_SO100_MOTORS \
  "$@"
