English| [简体中文](./WORKFLOW_GUIDE_CN.md)
# LeRobot + D-Robotics RDK End-to-End Workflow Guide (Detailed)

This document, based on the [D-Robotics/lerobot](https://github.com/D-Robotics/lerobot) repository and this toolchain, provides detailed steps to implement an ACT policy on the **SO-101 Robot Arm** from scratch and deploy it to **RDK S600**. For SO-101 assembly, motor setup, and calibration, also refer to the official Hugging Face [SO-101 documentation](https://huggingface.co/docs/lerobot/so101).

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/SO101_Leader.webp" width="80%" />
        <br /><b>Leader Arm</b>
      </td>
      <td align="center">
        <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/SO101_Follower.webp" width="80%" />
        <br /><b>Follower Arm</b>
      </td>
    </tr>
  </table>
</div>

> **🚀 Core Recommendation: RDK S600 Full-Stack Solution**
> 
> **RDK S600 is not just an inference terminal; it is a full-featured edge computing platform!**
> Apart from model training (which requires a GPU), you can complete all the following tasks directly on the RDK:
> *   ✅ **Hardware Calibration**
> *   ✅ **Teleoperation Testing**
> *   ✅ **Data Collection**
> *   ✅ **BPU Model Inference**
>
> We strongly recommend leveraging the portability of the RDK to connect the robot arm directly for data collection and debugging.

> **Version Statement**:
> *   **LeRobot**: This branch was verified with LeRobot v0.5.2.
> *   **Key Python packages**: `datasets 4.8.5`, `torch 2.7.1+cu126`, `onnxruntime 1.26.0`, `onnx 1.21.0`, `numpy 2.2.6`.
> *   **Hardware**: This document targets **RDK S600 + SO-101/SO100 single-arm ACT**.
> *   **BPU target**: S600 uses `nash-p`; OE 3.7.0 S100/S600 toolchain is recommended.

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
cd rdk_LeRobot_tools && git checkout s600 && cd ..

# 2. Install dependencies
conda activate lerobot
pip install -e ".[feetech]"
pip install onnx onnxsim termcolor tqdm safetensors
```

### 1.2 RDK Board Environment (For Collection & Inference)

SSH into RDK S600:

```bash
# 1. Clone D-Robotics LeRobot and this tools repo
git clone https://github.com/D-Robotics/lerobot.git
cd lerobot
git clone https://github.com/D-Robotics/rdk_LeRobot_tools.git
cd rdk_LeRobot_tools && git checkout s600 && cd ..
pip install -e ".[feetech]"

# 2. Install BPU runtime (required for inference, recommended to install)
pip install hbm-runtime
```

---

## 2. Hardware Configuration & Assembly (SO-101)

**Tip: Operations in this chapter can be performed on the development machine or directly on the RDK S600 via screen or SSH!**

### 2.1 Set Motor IDs

Before assembly, you need to set the ID for each motor. The SO-101 requires 6 motors for both Leader and Follower arms, with IDs 1-6 respectively.

**Steps:**
1.  Connect **only one** motor to the adapter board at a time.
2.  Use the LeRobot v0.5.2 motor setup command and follow the prompts to connect each motor one by one. Follower example:
    ```bash
    lerobot-setup-motors \
      --robot.type=so101_follower \
      --robot.port=/dev/ttyACM0
    ```
3.  Leader example:
    ```bash
    lerobot-setup-motors \
      --teleop.type=so101_leader \
      --teleop.port=/dev/ttyACM1
    ```
4.  Follow the CLI prompts, connecting only the requested motor each time, until motor IDs and baudrate are configured.

**Motor setup demo video:**
<video controls width="100%" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/setup_motors_so101_2.mp4" type="video/mp4"></video>

### 2.2 Assembly Instructions

Please refer to the [Official SO-ARM100 Guide](https://github.com/TheRobotStudio/SO-ARM100) and the Hugging Face [SO-101 documentation](https://huggingface.co/docs/lerobot/so101) for assembly.

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
    <video controls width="100%" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/Joint1_v2.mp4" type="video/mp4"></video>

*   **Joint 2 (Shoulder)**:
    <video controls width="100%" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/Joint2_v2.mp4" type="video/mp4"></video>

*   **Joint 3 (Elbow)**:
    <video controls width="100%" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/Joint3_v2.mp4" type="video/mp4"></video>

*   **Joint 4 (Wrist Roll)**:
    <video controls width="100%" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/Joint4_v2.mp4" type="video/mp4"></video>

*   **Joint 5 (Wrist Pitch)**:
    <video controls width="100%" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/Joint5_v2.mp4" type="video/mp4"></video>

*   **Gripper (Follower)**:
    <video controls width="100%" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/Gripper_v2.mp4" type="video/mp4"></video>

*   **Handle (Leader)**:
    <video controls width="100%" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/Leader_v2.mp4" type="video/mp4"></video>

### 2.3 Find Ports (Recommended on RDK S600)

Connect the assembled Leader and Follower arms to the USB ports of RDK S600. LeRobot v0.5.2 recommends:

```bash
lerobot-find-port
```
Follow the unplug/replug prompts and record the detected ports, e.g. `/dev/ttyACM0` and `/dev/ttyACM1`. On Linux, if serial permissions are insufficient, temporarily run:

```bash
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```

Pass the port via command-line options in collection or inference scripts. This tool's `bpu_control_robot.py` defaults to `--robot-port /dev/ttyACM0`.

---

## 3. Calibration

**Recommended to run directly on RDK S600.**
Calibration is crucial for synchronizing arms and making the trained policy transferable. LeRobot v0.5.2 uses `lerobot-calibrate`, following the same flow as the official Hugging Face [SO-101 documentation](https://huggingface.co/docs/lerobot/so101):

1. Move the arm so every joint is near the middle of its range.
2. Press Enter, then move each joint through its full range of motion.

**Calibration demo video:**

<video controls width="100%" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/calibrate_so101_2.mp4" type="video/mp4"></video>

### 3.1 Calibrate Follower

```bash
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=s600_follower
```

### 3.2 Calibrate Leader

```bash
lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=s600_leader
```

---

## 4. Camera Configuration

**Recommended to run directly on RDK S600.**
LeRobot v0.5.2 no longer edits cameras through `configs.py` / `so101.yaml`. Pass camera settings directly via `--robot.cameras` in `lerobot-record` or via `--camera-index` / `--camera-name` in `bpu_control_robot.py`.

### 4.1 Find Camera Indices

Connect all USB cameras to RDK and run:

```bash
lerobot-find-cameras
```

Note each camera's `index_or_path`, e.g. `0`, `1`.

### 4.2 Configure Cameras on the CLI

Use the same camera name across collection, training, export, and board inference. For example, use `front` everywhere:

```bash
--robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}"
```

For board-side BPU inference, the matching options are:

```bash
--camera-index 0 \
--camera-name front
```

`--camera-name` must match files such as `front_mean.npy` / `front_std.npy` inside `bpu_output/`.

---

## 5. Data Collection (Data Collection)

**Recommended to run directly on RDK S600.**
Collecting high-quality demonstration data is key to training success. It is recommended to collect **50+** successful trajectories.

### 5.1 Run Collection Script

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --robot.id=s600_follower \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=s600_leader \
  --dataset.repo_id=my_id/so101_pick_place \
  --dataset.root=/path/to/datasets/so101_pick_place \
  --dataset.num_episodes=50 \
  --dataset.single_task="Pick and place the object" \
  --dataset.episode_time_s=40 \
  --dataset.reset_time_s=5 \
  --dataset.push_to_hub=false \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --display_data=true
```

If `--dataset.root` is omitted, data is stored under `~/.cache/huggingface/lerobot/<repo_id>` by default.

### 5.2 Key Parameters

| Parameter | Meaning | Recommendation/Note |
| :--- | :--- | :--- |
| `--robot.type` | Follower arm type | `so101_follower` |
| `--robot.port` | Follower serial port | Get it with `lerobot-find-port` |
| `--teleop.type` | Leader arm type | `so101_leader` |
| `--teleop.port` | Leader serial port | Get it with `lerobot-find-port` |
| `--robot.cameras` | Camera configuration | S600 USB cameras usually use `opencv` + `index_or_path` |
| `--dataset.repo_id` | Dataset ID | Format `user/dataset_name` |
| `--dataset.root` | Local save path | Recommended to set explicitly for later training |
| `--dataset.num_episodes` | Total episodes | `50+` recommended |
| `--dataset.single_task` | Task description | Keep it consistent with the actual collection task |
| `--dataset.episode_time_s` | Max duration per episode | `30-40`s for simple tasks |
| `--dataset.reset_time_s` | Reset time | `5`s recommended |
| `--dataset.push_to_hub` | Upload to Hub | Use `false` for local-only debugging |

### 5.3 Keyboard Controls

*   **Right Arrow (`->`)**: Stop current episode early and go to Reset.
*   **Left Arrow (`<-`)**: Discard current episode and re-record.
*   **ESC**: Stop collection task early and save data.

### 5.4 Data Verification

After collection, verify that the data is valid (images clear, motion synchronized).
## 6. Model Training (ACT Policy)

**This step must be run on the Development Machine (with GPU).**
Copy the dataset directory collected on RDK (e.g. `/path/to/datasets/so101_pick_place`) to the development machine.

### 6.1 Install Training Dependencies

```bash
conda activate lerobot
pip install -e ".[training]"
```

### 6.2 Start Training

LeRobot v0.5.2 recommends passing training parameters directly on the CLI instead of editing `lerobot/configs/train.py`.

```bash
lerobot-train \
  --dataset.repo_id=my_id/so101_pick_place \
  --dataset.root=/path/to/datasets/so101_pick_place \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_test \
  --job_name=act_so101_test \
  --steps=100000 \
  --batch_size=8 \
  --policy.device=cuda \
  --wandb.enable=true
```

**Parameter Details**:
*   `--dataset.repo_id`: Dataset ID, must match the collection step.
*   `--dataset.root`: Local dataset path.
*   `--policy.type=act`: Uses ACT; network structure adapts to robot/camera info in the dataset.
*   `--steps` / `--batch_size`: Training steps and batch size, set directly via CLI.
*   `--policy.device=cuda`: Use `cuda` on NVIDIA GPUs.
*   `--wandb.enable=true`: Enables W&B (requires `wandb login` first).

**Resume Training**:

If training is interrupted, you can resume by specifying the checkpoint's configuration file path. For example, to resume from the `last` checkpoint of the `act_so101_test` task:

```bash
lerobot-train \
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

For S600 / SO100 ACT, use `rdk_LeRobot_tools/bpu_export_config_s600_calfix.yaml` as a reference:

```yaml
dataset:
  repo_id: "local/so100_demo"
  root: "/path/to/datasets/so100_demo"
policy:
  type: "act"
  device: "cpu"
act_path: "/path/to/outputs/train/act_so100/checkpoints/008000/pretrained_model"
export_path: "/path/to/bpu_export_act_so100_s600_calfix"
cal_num: 100
onnx_sim: true
type: "nash-p"       # RDK S600
combine_jobs: 6
```

This config generates calibration data aligned with S600 runtime preprocessing: image tensors are converted from `0..255` to `0..1` before `(image - mean) / std`.

### 7.2 Export ONNX

```bash
# 1. Export ONNX (Development Machine)
cd rdk_LeRobot_tools
python export_bpu_actpolicy.py --config bpu_export_config_s600_calfix.yaml
```
*Success indicator: The directory specified by `export_path` contains `build_all.sh`, ONNX files, and calibration data.*

### 7.3 Compile BPU Model (OpenExplorer Docker Environment)

1.  **Install Docker**
    *   Follow official instructions to install and verify: [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)
    *   Verify:  
        ```bash
        sudo docker --version
        sudo docker run --rm hello-world
        ```

2.  **Get and Load Offline Image** (for S600, use the OE 3.7.0 S100/S600 CPU image)
    *   Download page: [https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview#docker-%E9%95%9C%E5%83%8F](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview#docker-%E9%95%9C%E5%83%8F)
    *   Load image:
        ```bash
        sudo docker load -i ai_toolchain_ubuntu_22_s100_s600_cpu_v3.7.0.tar
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
        - Recommended S600 image: `registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0`

4.  **Compile Model Inside Container**
    *   Enter the mounted directory and execute the build script:
        ```bash
        cd /workspace/bpu_export_act_so100_s600_calfix
        bash build_all.sh
        ```
    *   Compilation output is usually located under `bpu_output/` and each submodel directory inside `export_path` (confirm via script output).

5.  **Common Issues & Troubleshooting**
    *   **Permission Issues**: Permission errors when copying files back to the host; check file ownership or use `sudo chown -R`.
    *   **Insufficient Disk Space**: Compilation generates large temporary files; ensure the host has enough disk space.
    *   **Memory/IPC Errors**: Increase `--shm-size` (e.g., 15g) or appropriately increase container memory limits.
    *   **Uncertain Image Name**: Run `sudo docker images` to view the loaded image tags and IDs.
    *   If you need to keep container artifacts long-term, do not use `--rm` or write outputs to the host mounted directory.

**Example Full Workflow:**
```bash
# Run a one-shot compile command on the host
docker run --rm \
  -v /path/to/bpu_export_act_so100_s600_calfix:/workspace \
  -w /workspace \
  registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0 \
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
    |-- front_mean.npy      # camera name must match collection/export
    |-- front_std.npy
    |-- new_actions.npy
    `-- ...
```

After completion, copy the generated `bpu_output` folder to the RDK board for deployment.

---

## 8. Board Deployment & Inference (RDK S600)

### Prerequisites
1.  Installed LeRobot from `D-Robotics/lerobot` repository and `hbm-runtime`.
2.  Transferred the **`bpu_output`** folder (containing quantized `.hbm` models and calibration parameters) to the board.
3.  **Hardware Config**: Ensure robot port, camera index, and camera name match training/export settings. Calibration files are saved by `lerobot-calibrate` under `~/.cache/huggingface/lerobot/calibration/`.

### Run BPU Accelerated Inference

This is the final step to deploy the trained model to the RDK.

1.  **File Transfer**: Copy the `bpu_output` folder generated on the development machine to the RDK board.
2.  **Run Inference**:

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

    The current `bpu_control_robot.py` connects to **SO100Follower** by default, not `so101`. If you deploy on an SO-101 follower arm, first confirm the LeRobot robot type matches your hardware.

    ACT emits a 100-step action chunk in one inference. The script auto-detects `n_action_steps` from `new_actions.npy`. Do not pass `--n-action-steps 1` for debugging, because that changes ACT runtime semantics.

### Troubleshooting

*   **Robot Not Moving**: Check `ls /dev/ttyACM*`; confirm `--robot-port` is correct.
*   **Camera Error**: Confirm `--camera-index` and `--camera-name` match `bpu_output/*_mean.npy`.