English| [简体中文](./WORKFLOW_GUIDE_CN.md)
# LeRobot + D-Robotics RDK End-to-End Workflow Guide (Detailed)

This document, based on the [D-Robotics/lerobot](https://github.com/D-Robotics/lerobot) repository and this toolchain, provides detailed steps to implement an ACT policy on the **SO-101 Robot Arm** from scratch and deploy it to **RDK S100/S100P**.

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="imgs/so101-leader.webp" width="80%" />
        <br /><b>Leader Arm</b>
      </td>
      <td align="center">
        <img src="imgs/so101.webp" width="80%" />
        <br /><b>Follower Arm</b>
      </td>
    </tr>
  </table>
</div>

> **🚀 Core Recommendation: RDK S100/S100P Full-Stack Solution**
> 
> **RDK S100/S100P is not just an inference terminal; it is a full-featured edge computing platform!**
> Apart from model training (which requires a GPU), you can complete all the following tasks directly on the RDK:
> *   ✅ **Hardware Calibration**
> *   ✅ **Teleoperation Testing**
> *   ✅ **Data Collection**
> *   ✅ **BPU Model Inference**
>
> We strongly recommend leveraging the portability of the RDK to connect the robot arm directly for data collection and debugging.

> **Version Statement**:
> *   **Repository**: It is strictly recommended to use [D-Robotics/lerobot](https://github.com/D-Robotics/lerobot).
> *   **Hardware**: This document is verified only for **RDK S100** or **RDK S100P** + **SO-101 Robot Arm**. Other hardware platforms (like RDK X5) are not fully verified.
> *   **Stability**: This is the Stable version, compatible with LeRobot v2.1 dataset format.

---

## 1. Environment Setup (Development Machine & RDK)

We need to prepare two environments:
*   **Development Machine (PC/Server)**: Responsible for **Model Training** and **Model Export/Compilation** (NVIDIA GPU required).
*   **RDK Board**: Responsible for **Calibration, Data Collection, Teleoperation**, and **Final Inference**.

### 1.1 Development Machine Environment (For Training)

Ubuntu 20.04/22.04 + NVIDIA GPU is recommended.

```bash
# 1. Clone D-Robotics repository
git clone https://github.com/D-Robotics/lerobot.git
cd lerobot
git clone https://github.com/D-Robotics/rdk_LeRobot_tools.git

# 2. Install dependencies
pip install -e .
pip install onnx onnxsim termcolor tqdm
```

### 1.2 RDK Board Environment (For Collection & Inference)

SSH into RDK S100/S100P:

```bash
# 1. Clone D-Robotics LeRobot as well
git clone https://github.com/D-Robotics/lerobot.git
cd lerobot
pip install -e ".[feetech]"

# 2. Install BPU runtime (required for inference, recommended to install)
pip install hbm-runtime
```

---

## 2. Hardware Configuration & Assembly (SO-101)

**Tip: Operations in this chapter can be performed on the development machine or directly on the RDK S100 via screen or SSH!**

### 2.1 Set Motor IDs

Before assembly, you need to set the ID for each motor. The SO-101 requires 6 motors for both Leader and Follower arms, with IDs 1-6 respectively.

**Steps:**
1.  Connect **only one** motor to the adapter board at a time.
2.  Run the following command to set the ID (e.g., set to 1):
    ```bash
    python lerobot/scripts/configure_motor.py \
      --port /dev/ttyUSB0 \
      --brand feetech \
      --model sts3215 \
      --baudrate 1000000 \
      --ID 1
    ```
3.  Unplug the current motor, plug in the new one, and repeat the steps to set IDs to 2, 3, 4, 5, 6.

**Demo Video:**
<video controls width="100%" src="https://github.com/user-attachments/assets/b31c115f-e706-4dcd-b7f1-4535da62416d" type="video/mp4"></video>

### 2.2 Assembly Instructions

Please refer to the [Official SO-ARM100 Guide](https://github.com/TheRobotStudio/SO-ARM100) for 3D printed parts assembly.

| Leader-Arm Axis | Motor | Gear Ratio |
|-----------------|:-------:|:----------:|
| Base / Shoulder Yaw | 1 | 1 / 191 |
| Shoulder Pitch      | 2 | 1 / 345 |
| Elbow               | 3 | 1 / 191 |
| Wrist Roll          | 4 | 1 / 147 |
| Wrist Pitch         | 5 | 1 / 147 |
| Gripper             | 6 | 1 / 147 |

**Key Joint Assembly Demos:**

*   **Joint 1 (Base)**:
    <video controls width="100%" src="https://github.com/user-attachments/assets/b0ee9dee-a2d0-445b-8489-02ebecb3d639" type="video/mp4"></video>

*   **Joint 2 (Shoulder)**:
    <video controls width="100%" src="https://github.com/user-attachments/assets/32453dc2-5006-4140-9f56-f0d78eae5155" type="video/mp4"></video>

*   **Joint 3 (Elbow)**:
    <video controls width="100%" src="https://github.com/user-attachments/assets/7384b9a7-a946-440c-b292-91391bcc4d6b" type="video/mp4"></video>

*   **Joint 4 (Wrist Roll)**:
    <video controls width="100%" src="https://github.com/user-attachments/assets/dca78ad0-7c36-4bdf-8162-c9ac42a1506f" type="video/mp4"></video>

*   **Joint 5 (Wrist Pitch)**:
    <video controls width="100%" src="https://github.com/user-attachments/assets/55f5d245-976d-49ff-8b4a-59843c441b12" type="video/mp4"></video>

*   **Gripper (Follower)**:
    <video controls width="100%" src="https://github.com/user-attachments/assets/6f766aa9-cfae-4388-89e7-0247f198c086" type="video/mp4"></video>

*   **Handle (Leader)**:
    <video controls width="100%" src="https://github.com/user-attachments/assets/1308c93d-2ef1-4560-8e93-a3812568a202" type="video/mp4"></video>

*   **Wiring**:
    <video controls width="100%" src="https://github.com/user-attachments/assets/4c2cacfd-9276-4ee4-8bf2-ba2492667b78" type="video/mp4"></video>

### 2.3 Find Ports & Modify Config (Recommended on RDK S100)

Connect the assembled Leader and Follower arms to the USB ports of RDK S100.

```bash
python lerobot/scripts/find_motors_bus_port.py
```
Note the output ports, e.g., `/dev/ttyUSB0` and `/dev/ttyUSB1`.

**Modify Config File**:
Find the `So101RobotConfig` class in `lerobot/common/robot_devices/robots/configs.py`, or directly modify the YAML config `lerobot/configs/robot/so101.yaml`.

```python
    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/ttyUSB0",  <-- Change to Leader port
                motors={...},
            ),
        }
    )
    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/ttyUSB1",  <-- Change to Follower port
                motors={...},
            ),
        }
    )
```

**Demo Video:**
<video controls width="100%" src="https://github.com/user-attachments/assets/fc45d756-31bb-4a61-b973-a87d633d08a7" type="video/mp4"></video>

---

## 3. Calibration

**Recommended to run directly on RDK S100.**
Calibration is crucial for synchronizing Leader and Follower arms. **Must be run when the robot arm is in the Zero position (fully extended straight).**

### 3.1 Manual Calibration (Follower)

Move the follower arm to the following positions sequentially:

| 1. Middle | 2. Zero | 3. Rotated | 4. Rest |
| :---: | :---: | :---: | :---: |
| <img src="imgs/follower_middle.webp" width="100%"/> | <img src="imgs/follower_zero.webp" width="100%"/> | <img src="imgs/follower_rotated.webp" width="100%"/> | <img src="imgs/follower_rest.webp" width="100%"/> |

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --robot.cameras='{}' \
  --control.type=calibrate \
  --control.arms='["main_follower"]'
```

### 3.2 Manual Calibration (Leader)

Move the leader arm to the following positions sequentially:

| 1. Middle | 2. Zero | 3. Rotated | 4. Rest |
| :---: | :---: | :---: | :---: |
| <img src="imgs/leader_middle.webp" width="100%"/> | <img src="imgs/leader_zero.webp" width="100%"/> | <img src="imgs/leader_rotated.webp" width="100%"/> | <img src="imgs/leader_rest.webp" width="100%"/> |

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --robot.cameras='{}' \
  --control.type=calibrate \
  --control.arms='["main_leader"]'
```

---

## 4. Camera Configuration

**Recommended to run directly on RDK S100.**

### 4.1 Find Camera Indices

Connect all USB cameras to RDK and run:

```bash
python lerobot/common/robot_devices/cameras/opencv.py \
    --images-dir outputs/images_from_opencv_cameras
```
Check the generated images to confirm which view corresponds to `camera_00/01`.

### 4.2 Modify Config

Update in `lerobot/common/robot_devices/robots/configs.py` or `so101.yaml`:

```python
        cameras={
            "laptop": OpenCVCameraConfig(
                camera_index=0,  <-- Confirm index
                fps=30,
                width=640,
                height=480,
            ),
            "phone": OpenCVCameraConfig(
                camera_index=1,  <-- Confirm index
                fps=30,
                width=640,
                height=480,
            ),
        },
```

---

## 5. Data Collection (Data Collection)

**Recommended to run directly on RDK S100.**
Collecting high-quality demonstration data is key to training success. It is recommended to collect **50+** successful trajectories.

### 5.1 Run Collection Script

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=record \
  --control.fps=30 \
  --control.root=data/so101_pick_place \
  --control.repo_id=my_id/so101_pick_place \
  --control.tags='["so101","tutorial"]' \
  --control.warmup-time-s=5 \
  --control.episode-time-s=40 \
  --control.reset-time-s=5 \
  --control.num-episodes=50
```

### 5.2 Key Parameters

| Parameter | Meaning | Recommendation/Note |
| :--- | :--- | :--- |
| `--robot.type` | Robot type | `so101` |
| `--fps` | Frame rate | `30` (Standard for ACT) |
| `--root` | Local save path | E.g., `data/task_name` |
| `--repo-id` | Hugging Face Repo ID | Format `user/dataset_name` |
| `--warmup-time-s` | Warmup time | `5`s. Time to adjust pose before recording |
| `--episode-time-s` | Max duration per episode | `30-40`s for simple tasks |
| `--reset-time-s` | Reset time | `5`s. Time to reset object after recording |
| `--num-episodes` | Total episodes | `50+` recommended |

### 5.3 Keyboard Controls

*   **Right Arrow (`->`)**: Stop current episode early and go to Reset.
*   **Left Arrow (`<-`)**: Discard current episode and re-record.
*   **ESC**: Stop collection task early and save data.

### 5.4 Data Verification

After collection, verify that the data is valid (images clear, motion synchronized).
## 6. Model Training (ACT Policy)

**This step must be run on the Development Machine (with GPU).**
Copy the collected `data/so101_pick_place` folder from RDK to the development machine.

### 6.1 Modify Training Configuration (Recommended)

We recommend directly modifying the `TrainPipelineConfig` class in `lerobot/configs/train.py` to set default parameters. Or you can refer to the original tutorial to use `config.yaml`, etc.

Here is an example modification:

```python
# ... (Imports at the top of the file are omitted) ...

    # Core Training Parameters
    seed: int | None = 1000
    # Number of workers for the dataloader.
    num_workers: int = 4
    batch_size: int = 8
    steps: int = 100_000
    eval_freq: int = 20_000
    log_freq: int = 200

# ... (Rest of the file is omitted) ...
```

### 6.2 Start Training & Advanced Options

**Standard Start Command**:

```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/so101_test \
  --dataset.root=data/so101_pick_place \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_test \
  --job_name=act_so101_test \
  --policy.device=cuda \
  --wandb.enable=true
```

**Parameter Details**:
*   `--dataset.repo_id`: Specifies the dataset ID. If `root` is specified, this is a placeholder.
*   `--dataset.root`: Local dataset path (e.g., `data/so101_pick_place`).
*   `--policy.type=act`: Specifies using the ACT policy. This policy automatically loads configurations from `configuration_act.py` and adapts the network structure based on robot information saved in your dataset.
*   `--policy.device=cuda`: Specifies the training device. Use `cuda` for NVIDIA GPUs, `mps` for Apple Silicon.
*   `--wandb.enable=true`: Enables Weights and Biases for visualizing training plots (requires running `wandb login` first).

**Resume Training**:

If training is interrupted, you can resume by specifying the checkpoint's configuration file path. For example, to resume from the `last` checkpoint of the `act_so101_test` task:

```bash
python lerobot/scripts/train.py \
  --config_path=outputs/train/act_so101_test/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

**Monitor Training**:
*   Observe the Loss values in the terminal output; they should show a downward trend.
*   After training is complete, the weights will be saved in `outputs/train/act_so101_test/checkpoints`.

---

## 7. Model Export & BPU Compilation

**This step is performed on the Development Machine.**

### 7.1 Configure Export Parameters

Edit `rdk_LeRobot_tools/bpu_export_config.yaml`:

```yaml
dataset:
  root: "data/so101_pick_place"
act_path: "outputs/train/act_so101/checkpoints/050000/pretrained_model"
type: "nash-e" # RDK S100/S100P
```

### 7.2 Export ONNX

```bash
# 1. Export ONNX (Development Machine)
python export_bpu_actpolicy.py --config bpu_export_config.yaml
```
*Success indicator: A `bpu_export_output` directory is generated, containing `build_all.sh` and calibration data.*

### 7.3 Compile BPU Model (OpenExplorer Docker Environment)

1.  **Install Docker**
    *   Follow official instructions to install and verify: [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)
    *   Verify:  
        ```bash
        sudo docker --version
        sudo docker run --rm hello-world
        ```

2.  **Get and Load Offline Image** (Recommend CPU Image, select according to RDK model)
    *   Download page: [https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview#docker-%E9%95%9C%E5%83%8F](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview#docker-%E9%95%9C%E5%83%8F)
    *   Load image:
        ```bash
        sudo docker load -i ai_toolchain_ubuntu_22_s100_xxx.tar
        ```

3.  **Start Container** (Recommended Parameters)
    *   **Note**: Mount the host's working directory into the container and increase shared memory to avoid memory/IPC issues.
    *   **Example** (Map `/home/user/rdk_workspace` to `/workspace`):
        ```bash
        sudo docker run -it --rm \
         --network host \
         --shm-size=15g \
         -v /home/user/rdk_workspace:/workspace \
         --workdir /workspace \
         <docker-image-name> /bin/bash
        ```
    *   **Common Replacements**:
        - `<docker-image-name>` Replace with the loaded image name (check with `docker images`).

4.  **Compile Model Inside Container**
    *   Enter the mounted directory and execute the build script:
        ```bash
        cd /workspace/bpu_export_output
        bash build_all.sh
        ```
    *   Compilation output is usually located in a subdirectory under `bpu_export_output/` (confirm via script output).

5.  **Common Issues & Troubleshooting**
    *   **Permission Issues**: Permission errors when copying files back to the host; check file ownership or use `sudo chown -R`.
    *   **Insufficient Disk Space**: Compilation generates large temporary files; ensure the host has enough disk space.
    *   **Memory/IPC Errors**: Increase `--shm-size` (e.g., 15g) or appropriately increase container memory limits.
    *   **Uncertain Image Name**: Run `sudo docker images` to view the loaded image tags and IDs.
    *   If you need to keep container artifacts long-term, do not use `--rm` or write outputs to the host mounted directory.

**Example Full Workflow:**
```bash
# 1. Load image (Host)
sudo docker load -i ai_toolchain_ubuntu_22_s100_xxx.tar

# 2. Start container and mount current project directory (Host)
sudo docker run -it --rm --network host --shm-size=15g \
  -v "$(pwd)":/workspace --workdir /workspace <docker-image-name> /bin/bash

# 3. Compile inside container (Container)
cd /workspace/bpu_export_output
bash build_all.sh
```

Expected artifacts:

```
bpu_output/
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

After completion, copy the generated `bpu_output` folder to the RDK board for deployment.

---

## 8. Board Deployment & Inference (RDK S100/S100P)

### Prerequisites
1.  Installed LeRobot from `D-Robotics/lerobot` repository and `hbm_runtime`.
2.  Transferred the **`bpu_output`** folder (containing quantized `.hbm` models and calibration parameters) to the board.
3.  **Hardware Config**: Ensure **robot arm ports**, **camera ports**, and **calibration files** are correctly configured by referring to the Data Collection and Teleoperation steps above.

### Run BPU Accelerated Inference

This is the final step to deploy the trained model to the RDK.

1.  **File Transfer**: Copy the `bpu_output` folder generated on the development machine to the RDK board.
2.  **Run Inference**:

    ```bash
    cd rdk_LeRobot_tools
    
    python bpu_control_robot.py \
      --bpu-act-path ../bpu_output \
      --fps 30 \
      --inference-time 60
    ```

### Troubleshooting

*   **Robot Not Moving**: Check `ls /dev/ttyUSB*`; check sudo permissions.
*   **Camera Error**: Confirm that the auto-detected camera indices are correct.