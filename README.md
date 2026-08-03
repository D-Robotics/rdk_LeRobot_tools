English | [简体中文](./README_CN.md)

# RDK LeRobot Tools

RDK LeRobot Tools is a deployment toolkit for running policies trained with [Hugging Face LeRobot](https://github.com/huggingface/lerobot) on D-Robotics RDK devices. It collects the model-export, quantization, compilation, board-runtime, and robot-control code needed to move a policy from training to real hardware.

The `s600` branch currently maintains two policy backends for RDK S600:

| Policy | Repository support | Documentation |
| --- | --- | --- |
| ACT | ONNX export, BPU compilation, calibration, and SO100 on-board inference | [ACT guide](./models/act/README.md) |
| Pi0 | Post-training helpers, three-stage chain-aware quantization, standalone HBM runtime, and synchronous SO100 dual-camera inference | [Pi0 guide](./models/pi0/README.md) |

## Repository Layout

```text
models/
├─ act/       # ACT implementation and documentation
└─ pi0/       # Pi0 implementation and documentation
doc/          # Shared workflow notes and assets
```

The root-level `bpu_control_robot.py` and `export_bpu_actpolicy.py` files remain as compatibility launchers for existing ACT commands. New model-specific code and documentation should live under its corresponding `models/<policy>/` directory.

Large generated assets are intentionally not tracked, including checkpoints, HBM/ONNX binaries, datasets, calibration samples, robot calibration files, diagnostic dumps, tokens, and build output. Each model guide documents the required local runtime layout.
