English| [简体中文](./README_CN.md)
# RDK LeRobot Tools

**This `s600` branch is intended for exporting LeRobot v0.5.2 ACT policies and deploying them on RDK S600 BPU.**

**Note: For S600, use `nash-p` and the OE 3.7.0 S100/S600 toolchain.**

This repository provides a set of tools to export ACT policy models trained with the [LeRobot](https://github.com/D-Robotics/lerobot) framework and deploy them to D-Robotics RDK S600, utilizing the BPU for efficient inference.

For the full workflow documentation, see: 👉 *[Full Workflow Guide](./doc/WORKFLOW_GUIDE_EN.md)*

## Directory Structure

*   `export_bpu_actpolicy.py`: **Model Export Script** (runs on the development machine/training server). Used to convert PyTorch weights to ONNX and generate configuration files and scripts required for BPU compilation.
*   `bpu_export_config_s600_calfix.yaml`: **Recommended S600 config** for LeRobot v0.5.2 / SO100 ACT / `nash-p`.
*   `bpu_export_config.yaml`: Generic template for other platforms (defaults to `nash-e`; do not use on S600).
*   `bpu_control_robot.py`: **On-Board Deployment Script** (runs on the RDK board). Loads the compiled BPU model and controls the robot.

## 1. Environment Preparation

### 1.1 Development Machine (For Model Conversion)

This branch was verified with **LeRobot v0.5.2**. The verified environment used:

```text
datasets 4.8.5
torch 2.7.1+cu126
onnxruntime 1.26.0
onnx 1.21.0
numpy 2.2.6
```

A dedicated conda environment is recommended:

```bash
conda activate lerobot
cd /path/to/lerobot
pip install -e ".[feetech]"
```

You may also use the LeRobot repository provided by D-Robotics to set up the development environment:
👉 **https://github.com/D-Robotics/lerobot**

```bash
    git clone https://github.com/D-Robotics/lerobot.git
    cd lerobot
    git clone https://github.com/D-Robotics/rdk_LeRobot_tools.git
    pip install -e ".[feetech]"
```

Install the following Python packages for ONNX export and processing:

```bash
pip install onnx onnxsim termcolor tqdm safetensors
```

*Note: Model compilation (ONNX -> HBM) needs to be performed in the Docker toolchain environment (OpenExplorer) provided by D-Robotics.*

### 1.2 RDK Board (For Model Deployment)

The on-board runtime environment has high requirements for stability. Please be sure to Clone the specified version of the `D-Robotics/lerobot` repository:

1.  **Install LeRobot (D-Robotics Fork Version)**:
    ```bash
    git clone https://github.com/D-Robotics/lerobot.git
    cd lerobot
    pip install -e ".[feetech]"
    # This branch was verified with datasets 4.8.5.
    # If you use another LeRobot version, make sure the checkpoint and dataset can be loaded correctly.
    ```

2.  **Install BPU Inference Library**:
    ```bash
    pip install hbm-runtime
    ```

## 2. Model Export and Compilation (Executed on Development Machine)

This process is divided into two steps: first exporting ONNX and configuration, and then using the Horizon toolchain to compile into a BPU model.

### Step 1: Export ONNX and Configuration

1.  **Modify Configuration File**:
    For S600 / SO100 ACT, edit `bpu_export_config_s600_calfix.yaml` directly:
    *   `dataset.root`: Dataset root used during training.
    *   `act_path`: Trained ACT checkpoint path.
    *   `export_path`: Export output directory.
    *   `type`: Must be `nash-p` for S600.
    *   `cal_num`: Recommended `100`.

2.  **Run Export Script**:
    ```bash
    cd rdk_LeRobot_tools
    python export_bpu_actpolicy.py --config bpu_export_config_s600_calfix.yaml
    ```
    After successful execution, ONNX files, calibration data, and `build_all.sh` are generated under `export_path`.

    **Important Note:** The export script reads image, state, and action normalization statistics from the LeRobot v0.5.2 checkpoint processor safetensors. Image calibration tensors are first ensured to be `0..1` float inputs, then normalized with `(image - mean) / std`, matching the board runtime path `uint8 -> /255.0 -> normalize`.

### Step 2: Compile BPU Model

Enter the toolchain Docker environment and run the compilation script generated in the previous step:

```bash
cd /path/to/bpu_export_act_so100_s600_calfix
bash build_all.sh
```

For S600, use the OE 3.7.0 S100/S600 Docker toolchain, for example:

```bash
docker run --rm \
  -v /path/to/bpu_export_act_so100_s600_calfix:/workspace \
  -w /workspace \
  registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0 \
  bash build_all.sh
```

After compilation is complete, a **`bpu_output`** folder is generated under the export directory. This folder contains:
*   `.hbm` / `.bin`: The compiled BPU model files (executable on the BPU).
*   `.npy`: Normalization parameters required for runtime.
*   `new_actions.npy`: Model inference results before conversion (used for precision verification).

**Please copy the `bpu_output` folder to the RDK board.**


```bpu_output/
    |-- BPU_ACTPolicy_TransformerLayers.hbm
    |-- BPU_ACTPolicy_VisionEncoder.hbm
    |-- action_mean.npy
    |-- action_mean_unnormalize.npy
    |-- action_std.npy
    |-- action_std_unnormalize.npy
    |-- front_mean.npy
    |-- front_std.npy
    |-- new_actions.npy
    `-- ...
```

## 3. On-Board Inference (Executed on RDK)

The core of on-board inference is using the **`bpu_control_robot.py`** script.

### Prerequisites
1.  Installed LeRobot from the `D-Robotics/lerobot` repository and `hbm-runtime`.
2.  Transferred the **`bpu_output`** folder (containing the quantized `.hbm` model and calibration parameters) to the board.
3.  **Hardware Configuration**: Pass robot port, camera index, and camera name via CLI. Calibration files are saved by `lerobot-calibrate`.

### Run Steps

1.  Connect the robot. The current script uses **SO100Follower** by default.
2.  Run the control script:

    ```bash
    cd rdk_LeRobot_tools
    python bpu_control_robot.py \
      --bpu-act-path ../bpu_output \
      --robot-port /dev/ttyACM0 \
      --camera-index 0 \
      --camera-name front \
      --fps 30 \
      --inference-time 60
    ```

### Common Parameters

*   `--bpu-act-path`: BPU model folder path (must contain `.hbm` and `.npy` files).
*   `--robot-port`: Follower serial port, default `/dev/ttyACM0`.
*   `--camera-index` / `--camera-name`: Camera index and name, must match `bpu_output/*_mean.npy`.
*   `--fps`: Control loop frequency (default 30Hz).
*   `--inference-time`: Duration of automatic operation (seconds).

## Notes

*   **Model Compatibility**: On-board execution must use `.hbm` / `.bin` models quantized and compiled by the OE toolchain, and cannot directly run ONNX or PyTorch models.
*   **Robot Configuration**: `bpu_control_robot.py` is currently hardcoded to `SO100Follower`. For SO-101 deployment, update the robot type in code first.
*   **S600 calibration consistency**: For LeRobot v0.5.2 ACT models, image calibration must match runtime preprocessing: `uint8 -> /255.0 -> (image - mean) / std`. If `0..255` images are normalized directly with ImageNet mean/std, the Vision/Transformer quantization ranges will be wrong and BPU actions may diverge significantly from PyTorch/ONNX.
*   **ACT chunking**: ACT outputs a 100-step action chunk in one inference. Do not pass `--n-action-steps 1` to work around deployment issues.
