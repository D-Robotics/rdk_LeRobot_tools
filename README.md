English| [简体中文](./README_CN.md)
# RDK LeRobot Tools

**This is the STABLE version, primarily adapted for older versions of LeRobot (compatible with v2.1 datasets). For newer versions of LeRobot, please switch to the corresponding branch.**

**Note: This tool has currently only verified the deployment effect of ACT models on RDK S100. The effects on other hardware platforms or model architectures are not guaranteed.**

This repository provides a set of tools to export ACT policy models trained based on the [LeRobot](https://github.com/D-Robotics/lerobot) framework and deploy them to D-Robotics RDK S100, utilizing the BPU for efficient inference.

For the full workflow documentation, see: 👉 *[Full Workflow Guide](./doc/WORKFLOW_GUIDE_EN.md)*

## Directory Structure

*   `damo/`: Toolkit adapted for the DAMO Developer Matrix - LeYun Embodied Intelligence Development Platform.
*   `export_bpu_actpolicy.py`: **Model Export Script** (runs on the development machine/training server). Used to convert PyTorch weights to ONNX and generate configuration files and scripts required for BPU compilation.
*   `bpu_export_config.yaml`: Model export configuration file.
*   `bpu_control_robot.py`: **On-Board Deployment Script** (runs on the RDK board). Loads the compiled BPU model and controls the robot.

## 1. Environment Preparation

### 1.1 Development Machine (For Model Conversion)

It is **strictly recommended** to use the LeRobot repository provided by D-Robotics to set up the development environment to ensure the best compatibility:
👉 **https://github.com/D-Robotics/lerobot**

```bash
    git clone https://github.com/D-Robotics/lerobot.git
    cd lerobot
    git clone https://github.com/D-Robotics/rdk_LeRobot_tools.git
    pip install -e ".[feetech]"
```

This version is compatible with v2.1 datasets. The export tools in this repository will work as long as they can load historical versions of v2.1 datasets.

Or clone the official repository and switch to the corresponding older-version branch:
```bash
    git clone https://github.com/huggingface/lerobot.git
    cd lerobot
    git checkout 8cfab3882480bdde38e42d93a9752de5ed42cae2  # 切换到 v2.1 版本对应的 commit
    git clone https://github.com/D-Robotics/rdk_LeRobot_tools.git
    pip install -e ".[feetech]"
```

**Special Note:** Newer versions of LeRobot may have compatibility issues with older code, leading to errors. The D-Robotics fork of the LeRobot repository has already locked the `datasets` library version. If you clone other versions of the LeRobot repository and encounter compatibility problems, you might need to manually downgrade the `datasets` library to `datasets==2.19.0` to avoid compatibility issues.

Install the following Python packages for ONNX export and processing:

```bash
pip install onnx onnxsim termcolor tqdm
```

*Note: Model compilation (ONNX -> HBM) needs to be performed in the Docker toolchain environment (OpenExplorer) provided by D-Robotics.*

### 1.2 RDK Board (For Model Deployment)

The on-board runtime environment has high requirements for stability. Please be sure to Clone the specified version of the `D-Robotics/lerobot` repository:

1.  **Install LeRobot (D-Robotics Fork Version)**:
    ```bash
    git clone https://github.com/D-Robotics/lerobot.git
    cd lerobot
    pip install -e .
    # The D-Robotics fork version has locked the datasets dependency, so no manual action is needed.
    # If you are using other LeRobot repository versions and encounter compatibility issues, you might need to manually install datasets==2.19.0.
    ```

2.  **Install BPU Inference Library**:
    ```bash
    pip install hbm-runtime
    ```

## 2. Model Export and Compilation (Executed on Development Machine)

This process is divided into two steps: first exporting ONNX and configuration, and then using the Horizon toolchain to compile into a BPU model.

### Step 1: Export ONNX and Configuration

1.  **Modify Configuration File**:
    Edit `bpu_export_config.yaml` and modify the following key fields according to the actual situation:
    *   `dataset.root`: The root directory of the dataset used during training.
    *   `act_path`: The path to the trained ACT model checkpoint (containing `config.json` and `model.safetensors`).
    *   `type`: BPU platform type. The script will automatically adjust compilation parameters based on this type.
        *   `nash-e` / `nash-m` / `nash-p`: Suitable for Nash architectures such as **RDK S100**.
        *   `bayes` / `bayes-e`: Suitable for Bayes architectures such as **RDK X5**.

2.  **Run Export Script**:
    ```bash
    python export_bpu_actpolicy.py --config bpu_export_config.yaml
    ```
    After successful execution, the ONNX model, calibration data, and compilation script (`build_all.sh`) will be generated under `bpu_export_output` (or the directory specified in the configuration).

    **Important Note:** For newer versions of LeRobot (v2.1), to avoid missing key errors (such as `policy.type`) during export, please make sure to uncomment the corresponding `policy` and `dataset` sections in `bpu_export_config.yaml`. Please refer to the instructions in the `bpu_export_config.yaml` template file for details.

### Step 2: Compile BPU Model

Enter the toolchain Docker environment and run the compilation script generated in the previous step:

```bash
cd bpu_export_output
bash build_all.sh
```

After compilation is complete, a **`bpu_output`** folder will be generated in the `bpu_export_output` directory. This folder contains:
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
    |-- camera1_mean.npy    # camera names are auto-detected
    |-- camera1_std.npy
    |-- camera2_mean.npy
    `-- camera2_std.npy
```

## 3. On-Board Inference (Executed on RDK)

The core of on-board inference is using the **`bpu_control_robot.py`** script.

### Prerequisites
1.  Installed LeRobot from the `D-Robotics/lerobot` repository and `hbm_runtime`.
2.  Transferred the **`bpu_output`** folder (containing the quantized `.hbm` model and calibration parameters) to the board.
3.  **Hardware Configuration**: Please refer to the official data collection and teleoperation steps to complete the `config` file configuration, ensuring that the **robot arm port number**, **camera port number**, and **calibration file** are configured correctly.

### Run Steps

1.  Connect the robot (default configuration is `so101`).
2.  Run the control script, specifying the model path:

    ```bash
    # Assuming bpu_output is in the current directory
    python bpu_control_robot.py --bpu-act-path ./bpu_output
    ```

### Common Parameters

*   `--bpu-act-path`: BPU model folder path (must contain `.hbm` and `.npy` files).
*   `--fps`: Control loop frequency (default 30Hz).
*   `--inference-time`: Duration of automatic operation (seconds).

## 4. DAMO Platform Model Adaptation

If you are using data collected and models trained on the **DAMO Developer Matrix - LeYun Embodied Intelligence Development Platform**, you **must** adapt the dataset format before exporting the BPU model.

**Operation Steps:**

1.  **Backup Data**: This operation will directly modify the source files, so please be sure to backup your dataset folder first.
2.  **Edit Script**: Open `damo/replace.py` and modify `folder_path` to your dataset path.
3.  **Run Conversion**: `python damo/replace.py`

## Notes

*   **Model Compatibility**: On-board execution must use `.hbm` / `.bin` models quantized and compiled by the OE toolchain, and cannot directly run ONNX or PyTorch models.
*   **Robot Configuration**: `bpu_control_robot.py` connects to the `so101` robot by default. If you need to change it, please modify `make_robot("so101")` in the code.
