English | [简体中文](./README_CN.md)

# Pi0 Policy on RDK S600

This document covers Pi0 post-training, chain-aware quantization, standalone HBM deployment, offline validation, and synchronous SO100 dual-camera inference.

The Pi0 implementation starts at `models/pi0/`. It was validated with a real SO100 follower, two USB cameras, and the following strictly synchronous path:

```text
front + side + 6-D joint state + fixed task prompt
  ↓
SigLIP HBM × 3 slots
  ├─ front: real image
  ├─ side: real image
  └─ third slot: masked empty image
  ↓
PaliGemma HBM → 36 KV tensors
  ↓
Action Expert HBM × 10 flow-matching steps
  ↓
[50, 6] absolute SO100 targets
  ↓
30 Hz SO100Follower.send_action()
```

The production launcher uses `prefetch_steps=0`: each action chunk is inferred from the latest state and both latest images, executed for 50 control steps, and only then is the next observation captured. PaliGemma KV is reused only across the ten Expert denoise steps of the same request; it is regenerated for every new image pair.

## Verified Baseline

- LeRobot: v0.5.2.
- Robot: SO100 follower, 6 action dimensions in degrees.
- Cameras: front + side real cameras, plus one masked empty Pi0 slot.
- Training: 100 episodes, 15,000 BF16 steps, batch size 1; SigLIP, PaliGemma, and Expert all trainable.
- Runtime: D-Robotics LLM S600 SDK 1.0.2, `nash-p`, no runtime dependency on `libxlm.so`.
- Continuous hardware validation: 64 synchronous chunks, with a fresh front/side image pair per chunk.
- One fixed real-input BF16/HBM comparison: MAE `0.4405°`, RMSE `0.5903°`, max error `1.6888°`, relative L2 `0.852%`, cosine `0.999981858`.

The numerical comparison is a chain-integrity check for one fixed input, not a task-success-rate report.

## Directory Layout

```text
models/pi0/
├─ pi0_full_pipeline.py         # cameras, SO100 lifecycle, TCP server, 30 Hz control
├─ pi0_standalone_offline.py    # no-serial offline full-chain smoke test
├─ pi0_torch_server.py          # BF16 server reference
├─ run_live_sync.sh             # verified synchronous hardware launcher
├─ validate_pi0_config.py       # deployment artifact/hash validation
├─ configs/                     # deployment metadata, stats, quantization manifests
├─ native/                      # standalone C++ runtime and SDK compatibility layer
└─ tools/                       # training, calibration, three-stage quantization
```

The repository intentionally does **not** track HBM files, datasets, real calibration images/KV, tokens, robot calibration, diagnostic dumps, or compiled binaries.

## Required On-Board Layout

```text
/root/rdk_LeRobot_tools
/home/sunrise/lerobot
/root/D-Robotics_LLM_S600_1.0.2_SDK/oellm_runtime
/root/pi0_models/versions
```

The final deployment config is:

```text
models/pi0/configs/deployments/pi0_full_v5_2cam_positionfp16_siglip_fixed16_paligemma_hbmkv_expert_20260801.json
```

Its companion manifest records the expected model sources, file sizes, and SHA256 values. Do not replace one HBM independently without rebuilding or revalidating the downstream chain.

## Build and Non-Actuating Validation

Build the standalone runtime:

```bash
cd /root/rdk_LeRobot_tools
git checkout s600
bash models/pi0/native/build_standalone_pi0.sh
```

Run static checks and validate the final deployment bundle:

```bash
cd /root/rdk_LeRobot_tools/models/pi0
bash -n run_live_sync.sh run_pi0_standalone_config.sh native/build_standalone_pi0.sh
/home/sunrise/lerobot/.venv/bin/python -m py_compile \
  pi0_full_pipeline.py pi0_standalone_offline.py pi0_torch_server.py validate_pi0_config.py
/home/sunrise/lerobot/.venv/bin/python validate_pi0_config.py \
  configs/deployments/pi0_full_v5_2cam_positionfp16_siglip_fixed16_paligemma_hbmkv_expert_20260801.json
```

Offline dual-image smoke test; this does not open the robot serial port:

```bash
/home/sunrise/lerobot/.venv/bin/python -u pi0_standalone_offline.py \
  --front /path/to/front.jpg \
  --side /path/to/side.jpg \
  --state 0 0 0 0 0 0 \
  --output-dir diagnostics/offline_smoke_$(date +%Y%m%d_%H%M%S)
```

The expected output shape is `[50, 6]`.

## Continuous Hardware Inference

Before starting, verify `/dev/ttyACM0` is the follower, `/dev/video0` and `/dev/video2` are the intended cameras, and the robot workspace is clear:

```bash
cd /root/rdk_LeRobot_tools/models/pi0
./run_live_sync.sh
```

Use `Ctrl+C` to stop. The Python controller owns the process group and disables torque during normal shutdown. The launcher deliberately uses `--force-model-actions`: it does not add a 1°/20° relative target limiter or chunk blending, so a human operator must supervise the robot.

## Training and Chain-Aware Quantization Tools

| Stage | Entry point | Important constraint |
| --- | --- | --- |
| Full post-training | `models/pi0/tools/train_pi0_full_v5_2cam_100ep.sh` | Two real cameras require all three model parts to remain trainable. |
| Real calibration set | `models/pi0/tools/prepare_pi0_calibration_v5_2cam.py` | Use representative front + side frames and the matching checkpoint stats. |
| SigLIP | `models/pi0/tools/quantize_siglip_real_calib.py` | Preserve position embedding precision. |
| PaliGemma | `models/pi0/tools/quantize_paligemma_real_calib.py` | Calibrate with real SigLIP HBM outputs. |
| Expert | `models/pi0/tools/quantize_expert_real_calib.py` | Calibrate with the final PaliGemma HBM's real 36 KV tensors. |

The required order is `SigLIP HBM → dump real vision embeddings → PaliGemma HBM → dump real KV → Expert HBM`. Calibrating all three stages independently with floating-point intermediates can compile successfully while still producing poor closed-loop behavior.
