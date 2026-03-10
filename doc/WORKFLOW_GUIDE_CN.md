[English](./WORKFLOW_GUIDE_EN.md) | 简体中文
# LeRobot + 地瓜机器人 RDK 全流程落地指南

本文档基于 [D-Robotics/lerobot](https://github.com/D-Robotics/lerobot) 仓库及本工具链，提供从零开始在 **SO-101 机械臂** 上实现 ACT 策略并部署到 **RDK S100/S100P** 的详细步骤。

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="imgs/so101-leader.webp" width="80%" />
        <br /><b>Leader Arm (主手)</b>
      </td>
      <td align="center">
        <img src="imgs/so101.webp" width="80%" />
        <br /><b>Follower Arm (从手)</b>
      </td>
    </tr>
  </table>
</div>

> **🚀 核心推荐：RDK S100/S100P 全流程方案**
> 
> **RDK S100/S100P 不仅仅是一个推理终端，它是全功能的边缘计算平台！**
> 除了模型训练（需要 GPU）外，您可以直接在 RDK 上完成以下所有工作：
> *   ✅ **硬件标定** (Calibration)
> *   ✅ **遥操作测试** (Teleoperation)
> *   ✅ **数据采集** (Data Collection)
> *   ✅ **BPU 模型推理** (Inference)
>
> 我们强烈推荐您利用 RDK 的便携性，直接连接机械臂进行数据采集和调试。

---

## 1. 环境搭建 (开发机 & RDK)

我们需要准备两套环境：
*   **开发机 (PC/服务器)**: 负责 **模型训练** 和 **模型导出编译** (GPU 必需)。
*   **RDK 板端**: 负责 **标定、采集、遥操** 和 **最终推理**。

### 1.1 开发机环境 (用于训练)

建议使用 Ubuntu 20.04/22.04 + NVIDIA GPU。

```bash
# 1. 克隆 D-Robotics 仓库
git clone https://github.com/D-Robotics/lerobot.git
cd lerobot
git clone https://github.com/D-Robotics/rdk_LeRobot_tools.git

# 2. 安装依赖
pip install -e .
pip install onnx onnxsim termcolor tqdm
```

### 1.2 RDK 板端环境 (用于采集与推理)

SSH 登录到 RDK S100/S100P：

```bash
# 1. 同样克隆 D-Robotics 的 LeRobot
git clone https://github.com/D-Robotics/lerobot.git
cd lerobot
pip install -e ".[feetech]"

# 2. 安装 BPU 运行时 (仅推理需要，但建议安装)
pip install hbm-runtime
```

---

## 2. 硬件配置与组装 (SO-101)

**提示：本章节操作可以在开发机上进行，也可以直接在 RDK S100 上连接屏幕或 SSH 进行！**

### 2.1 设置电机 ID (Set motor IDs)

在组装前，需要先设置每个电机的 ID。SO-101 主从手各需 6 个电机，ID 分别为 1-6。

**操作步骤：**
1.  每次只连接**一个**电机到转接板。
2.  运行以下命令设置 ID（例如设置为 1）：
    ```bash
    python lerobot/scripts/configure_motor.py \
      --port /dev/ttyUSB0 \
      --brand feetech \
      --model sts3215 \
      --baudrate 1000000 \
      --ID 1
    ```
3.  拔下当前电机，插上新电机，重复步骤将 ID 设置为 2, 3, 4, 5, 6。

**操作演示视频：**
<video controls width="100%" src="https://github.com/user-attachments/assets/b31c115f-e706-4dcd-b7f1-4535da62416d" type="video/mp4"></video>

### 2.2 机械臂组装步骤

请参考 [SO-ARM100 官方指南](https://github.com/TheRobotStudio/SO-ARM100) 进行组装。以下是关键关节组装演示：

| Leader-Arm Axis | Motor | Gear Ratio |
|-----------------|:-------:|:----------:|
| Base / Shoulder Yaw | 1 | 1 / 191 |
| Shoulder Pitch      | 2 | 1 / 345 |
| Elbow               | 3 | 1 / 191 |
| Wrist Roll          | 4 | 1 / 147 |
| Wrist Pitch         | 5 | 1 / 147 |
| Gripper             | 6 | 1 / 147 |

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

*   **Wiring (接线)**:
    <video controls width="100%" src="https://github.com/user-attachments/assets/4c2cacfd-9276-4ee4-8bf2-ba2492667b78" type="video/mp4"></video>

### 2.3 查找端口与修改配置 (RDK S100 推荐)

将组装好的机械臂连接到 RDK S100 的 USB 口。

```bash
python lerobot/scripts/find_motors_bus_port.py
```
记下输出的端口号，例如 `/dev/ttyUSB0` 和 `/dev/ttyUSB1`。

**修改配置文件**：
找到 `lerobot/common/robot_devices/robots/configs.py` 中的 `So101RobotConfig` 类，或者直接修改 YAML 配置 `lerobot/configs/robot/so101.yaml`。

```python
    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/ttyUSB0",  <-- 修改为主手端口
                motors={...},
            ),
        }
    )
    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/ttyUSB1",  <-- 修改为从手端口
                motors={...},
            ),
        }
    )
```

**操作演示视频：**
<video controls width="100%" src="https://github.com/user-attachments/assets/fc45d756-31bb-4a61-b973-a87d633d08a7" type="video/mp4"></video>

---

## 3. 校准 (Calibration)

**推荐在 RDK S100 上直接运行。**
校准是保证主从手同步的关键。**必须在机械臂处于零位（完全伸直）时运行**。

### 3.1 手动校准从手 (Follower)

按顺序将从手移动到以下位置：

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

### 3.2 手动校准主手 (Leader)

按顺序将主手移动到以下位置：

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

## 4. 摄像头配置 (Cameras)

**推荐在 RDK S100 上直接运行。**

### 4.1 查找摄像头索引

连接所有 USB 摄像头到 RDK，运行脚本：

```bash
python lerobot/common/robot_devices/cameras/opencv.py \
    --images-dir outputs/images_from_opencv_cameras
```
将生成的图片传回 PC 查看，或直接在板端确认 `camera_00/01` 对应的视角。

### 4.2 修改配置

在 `lerobot/common/robot_devices/robots/configs.py` 或 `so101.yaml` 中更新：

```python
        cameras={
            "laptop": OpenCVCameraConfig(
                camera_index=0,  <-- 确认索引
                fps=30,
                width=640,
                height=480,
            ),
            "phone": OpenCVCameraConfig(
                camera_index=1,  <-- 确认索引
                fps=30,
                width=640,
                height=480,
            ),
        },
```

---

## 5. 数据采集 (Data Collection)

**推荐在 RDK S100 上直接运行。**
收集高质量的演示数据是训练成功的关键。建议采集 **50 条** 以上的成功轨迹。

### 5.1 运行采集脚本

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

### 5.2 关键参数详解

| 参数 | 含义 | 推荐值/说明 |
| :--- | :--- | :--- |
| `--robot.type` | 机器人类型 | `so101` |
| `--fps` | 采集帧率 | `30` (ACT 模型标准帧率) |
| `--root` | 数据本地保存路径 | 建议包含任务名称，如 `data/task_name` |
| `--repo-id` | Hugging Face 仓库ID | 格式 `user/dataset_name`，用于上传分享 |
| `--warmup-time-s` | 预热时间 | `5`秒。开始录制前给您调整姿态的时间 |
| `--episode-time-s` | 单条数据最大时长 | 根据任务难度设定，简单抓取 `30-40`秒足够 |
| `--reset-time-s` | 复位时间 | `5`秒。每条录制结束后，给您将物体归位的时间 |
| `--num-episodes` | 计划采集总条数 | `50` 条起步，多多益善 |

### 5.3 键盘控制 (Keyboard Shortcuts)

在终端运行采集脚本时，可以使用键盘控制流程：

*   **右箭头 (`->`)**: 提前结束当前 episode 的录制，进入复位阶段 (Reset)。
*   **左箭头 (`<-`)**: 放弃当前 episode (不保存)，重新开始录制这一条。适用于操作失误的情况。
*   **ESC**: 提前结束整个采集任务，并开始保存数据。

### 5.4 数据集校验

采集完成后，务必检查数据是否有效（图像是否清晰，动作是否同步）。


## 6. 模型训练 (ACT Policy)

**此步骤必须在开发机 (带 GPU) 上运行。**
将采集好的 `data/so101_pick_place` 文件夹从 RDK 拷贝到开发机。

### 6.1 修改训练配置 (推荐)

我们推荐直接修改 `lerobot/configs/train.py` 文件，以设置训练的默认参数，也可以参考原版教程使用config.yaml等方式来实现。

以下是修改示例：

```python
# ... (文件顶部导入部分省略) ...

    # 训练核心参数
    seed: int | None = 1000
    # Number of workers for the dataloader.
    num_workers: int = 4
    batch_size: int = 8
    steps: int = 100_000
    eval_freq: int = 20_000
    log_freq: int = 200

# ... (文件其余部分省略) ...
```

### 6.2 启动训练与进阶

**标准启动命令**:

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

**参数详解**:
*   `--dataset.repo_id`: 指定训练使用的数据集 ID 指定了root后此处占位即可。
*   `--dataset.root`: 本地数据集路径 (例如 `data/so101_pick_place`)。
*   `--policy.type=act`: 指定使用 ACT 策略。该策略会自动加载 `configuration_act.py` 中的配置，并根据数据集中保存的机器人信息（如电机状态数量、相机数量）自动适配网络结构。
*   `--policy.device=cuda`: 指定训练设备。NVIDIA GPU 使用 `cuda`，Apple Silicon 可以使用 `mps`。
*   `--wandb.enable=true`: 开启 Weights and Biases 可视化训练曲线（需先运行 `wandb login`）。

**恢复训练 (Resume Training)**:

如果训练中断，可以通过指定 checkpoint 的配置文件路径来恢复训练。例如，从 `act_so101_test` 任务的最新 checkpoint (`last`) 恢复：

```bash
python lerobot/scripts/train.py \
  --config_path=outputs/train/act_so101_test/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

**监控训练**:
*   观察终端输出的 Loss 值，应呈下降趋势。
*   训练完成后，权重文件将保存在 `outputs/train/act_so101_test/checkpoints`。

---

## 7. 模型导出与 BPU 编译

**此步骤在开发机上进行。**

### 7.1 配置导出参数

编辑 `rdk_LeRobot_tools/bpu_export_config.yaml`：

```yaml
dataset:
  root: "data/so101_pick_place"
act_path: "outputs/train/act_so101/checkpoints/050000/pretrained_model"
type: "nash-e" # RDK S100/S100P
```

### 7.2 导出 ONNX

```bash
# 1. 导出 ONNX (开发机)
python export_bpu_actpolicy.py --config bpu_export_config.yaml
```
*成功标志：生成 `bpu_export_output` 目录，内含 `build_all.sh` 和校准数据。*

### 7.3 编译 BPU 模型 (OpenExplorer Docker 环境)

1.  **安装 Docker**
    *   按照官方说明安装并验证： [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)
    *   验证：  
        ```bash
        sudo docker --version
        sudo docker run --rm hello-world
        ```

2.  **获取并加载离线镜像**（推荐 CPU 镜像，根据 RDK 型号选择）
    *   镜像下载页：[https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview#docker-%E9%95%9C%E5%83%8F](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview#docker-%E9%95%9C%E5%83%8F)
    *   加载镜像：
        ```bash
        sudo docker load -i ai_toolchain_ubuntu_22_s100_xxx.tar
        ```

3.  **启动容器**（推荐参数）
    *   **说明**：将宿主机的工作目录挂载到容器内，增大共享内存避免内存/IPC 问题。
    *   **示例**（把 `/home/user/rdk_workspace` 映射到容器的 `/workspace`）：
        ```bash
        sudo docker run -it --rm \
         --network host \
         --shm-size=15g \
         -v /home/user/rdk_workspace:/workspace \
         --workdir /workspace \
         <docker-image-name> /bin/bash
        ```
    *   **常用替换项**：
        - `<docker-image-name>` 替换为加载后的镜像名（用 `docker images` 查看）

4.  **在容器内编译模型**
    *   进入挂载目录并执行编译脚本：
        ```bash
        cd /workspace/bpu_export_output
        bash build_all.sh
        ```
    *   编译输出通常位于 `bpu_export_output/` 下的子目录（根据脚本输出确认）。

5.  **常见问题与排查**
    *   **权限问题**：宿主机复制回文件时出现权限错误，检查文件属主或使用 `sudo chown -R`。
    *   **磁盘空间不足**：编译会产生较大临时文件，确保宿主机有足够磁盘空间。
    *   **内存/IPC 报错**：增加 `--shm-size`（例如 15g）或适当增加容器内存限制。
    *   **镜像名不确定**：运行 `sudo docker images` 查看加载的镜像标签与 ID。
    *   若需要长期保留容器产物，请不要使用 `--rm` 或把 outputs 写到宿主机挂载目录。

**示例完整流程：**
```bash
# 1. 加载镜像 (宿主机)
sudo docker load -i ai_toolchain_ubuntu_22_s100_xxx.tar

# 2. 启动容器并挂载当前工程目录 (宿主机)
sudo docker run -it --rm --network host --shm-size=15g \
  -v "$(pwd)":/workspace --workdir /workspace <docker-image-name> /bin/bash

# 3. 在容器内编译 (容器内)
cd /workspace/bpu_export_output
bash build_all.sh
```

预计产物为：

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

完成后，请将生成的 `bpu_output` 文件夹拷贝到 RDK 板端用于部署。

---

## 8. 板端部署与推理 (RDK S100/S100P)

### 前提条件
1.  已安装 `D-Robotics/lerobot` 仓库的 LeRobot 和 `hbm_runtime`。
2.  已将 **`bpu_output`** 文件夹（包含量化后的 `.hbm` 模型和校准参数）传输到板端。
3.  **硬件配置**: 请参考以上的数据采集和遥操作步骤，完成 `config` 文件的配置，确保**机械臂端口号**、**相机端口号**及**校准文件**配置正确。

### 运行 BPU 加速推理

这是将训练好的模型部署到 RDK 上的最终步骤。

1.  **文件传输**: 将开发机生成的 `bpu_output` 文件夹拷贝到 RDK 板子。
2.  **运行推理**：

    ```bash
    cd rdk_LeRobot_tools
    
    python bpu_control_robot.py \
      --bpu-act-path ../bpu_output \
      --fps 30 \
      --inference-time 60
    ```

### 故障排查

*   **机械臂不动**: 检查 `ls /dev/ttyUSB*`；检查 sudo 权限。
*   **相机报错**: 确认自动检测的相机 index 对应正确。