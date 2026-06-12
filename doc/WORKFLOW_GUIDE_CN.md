[English](./WORKFLOW_GUIDE_EN.md) | 简体中文
# LeRobot + 地瓜机器人 RDK 全流程落地指南

> 大约在一年前，我们在 **RDK S100** 上跑通了 Hugging Face [LeRobot](https://github.com/huggingface/lerobot) 的 ACT 策略部署——从遥操数采、模型训练，到 BPU 量化推理，端到端地走了一遍全流程，并把经验分享在了[社区论坛](https://forum.d-robotics.cc/t/topic/28858)上。
>
> 但过去这一年里，事情发生了不少变化：
>
> - **LeRobot 框架大幅升级**：从最初的 v0.1/v0.2 一路迭代到了 v0.5.2。API 几乎全部重写——数据集格式从 v2.1（一条 episode 一个文件）演进到了 v3.0（多 episode 合并打包），训练/采集/标定的命令行接口也换成了 `lerobot-record`、`lerobot-train`、`lerobot-calibrate` 等一套全新 CLI。
> - **地瓜机器人推出了 RDK S600**：更强的算力，搭配 OE 3.7.0 工具链和 `nash-p` 架构，成为端侧部署的新主力平台。
> - **老教程逐渐跟不上**：论坛里陆续有朋友反馈，按旧文档操作会遇到数据集格式不兼容、命令找不到、校准量化范围对不上等问题。
>
> 所以我们重新梳理了整条链路，基于 **LeRobot v0.5.2 + RDK S600 + SO-101 机械臂**，从头验证了全流程，并更新了导出脚本和工具链配置。这篇文档就是更新后的完整落地指南——无论你是第一次接触 LeRobot 的新朋友，还是从旧版教程迁移过来的老用户，都可以从这里开始。

本文档基于 [Hugging Face LeRobot](https://github.com/huggingface/lerobot) 仓库及本工具链，提供从零开始在 **SO-101 机械臂** 上实现 ACT 策略并部署到 **RDK S600** 的详细步骤。SO-101 机械臂装配、电机设置和校准流程也可参考官方 [SO-101 文档](https://huggingface.co/docs/lerobot/so101)。

**Pick and Place 演示：**

<div align="center">
  <img src="./assets/demo_pick_place.gif" width="480" alt="Pick and Place 演示" />
</div>

> 这个演示只是简单的 Pick and Place 展示，仅采集了 33 组训练数据。以下是 6 组训练数据（Episode 0/6/13/20/26/32）的并排可视化：

<div align="center">
  <img src="./assets/demo_episodes_grid.gif" width="640" alt="训练数据可视化 - 6 组 Episode 并排展示" />
</div>

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/SO101_Leader.webp" width="80%" />
        <br /><b>Leader Arm (主手)</b>
      </td>
      <td align="center">
        <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/SO101_Follower.webp" width="80%" />
        <br /><b>Follower Arm (从手)</b>
      </td>
    </tr>
  </table>
</div>

> **🚀 核心推荐：RDK S600 全流程方案**
> 
> **RDK S600 不仅仅是一个推理终端，它是全功能的边缘计算平台！**
> 除了模型训练（需要 GPU）外，您可以直接在 RDK 上完成以下所有工作：
> *   ✅ **硬件标定** (Calibration)
> *   ✅ **遥操作测试** (Teleoperation)
> *   ✅ **数据采集** (Data Collection)
> *   ✅ **BPU 模型推理** (Inference)
>
> 我们强烈推荐您利用 RDK 的便携性，直接连接机械臂进行数据采集和调试。

---

> **版本说明**：
> *   **LeRobot**: 本分支按 LeRobot v0.5.2 验证。
> *   **Python 关键依赖**: `datasets 4.8.5`, `torch 2.7.1+cu126`, `onnxruntime 1.26.0`, `onnx 1.21.0`, `numpy 2.2.6`。
> *   **硬件**: 本文档面向 **RDK S600 + SO-101/SO100 单臂 ACT**。
> *   **BPU 编译目标**: S600 使用 `nash-p`，推荐 OE 3.7.0 S100/S600 工具链。

---

## 1. 环境搭建 (开发机 & RDK)

**请使用 [huggingface/lerobot](https://github.com/huggingface/lerobot) 仓库，不要使用已过时的 `D-Robotics/lerobot` fork。本分支按 LeRobot v0.5.2 验证。**

我们需要准备两套环境：
*   **开发机 (PC/服务器)**: 负责 **模型训练** 和 **模型导出编译** (GPU 必需)。
*   **RDK 板端**: 负责 **标定、采集、遥操** 和 **最终推理**。

### 1.1 开发机环境 (用于训练)

建议使用 Ubuntu 20.04/22.04 + NVIDIA GPU。

```bash
# 1. 克隆 Hugging Face LeRobot 仓库
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git clone https://github.com/D-Robotics/rdk_LeRobot_tools.git
cd rdk_LeRobot_tools && git checkout s600 && cd ..

# 2. 安装依赖
conda activate lerobot
pip install -e ".[feetech]"
pip install onnx onnxsim termcolor tqdm safetensors
```

### 1.2 RDK 板端环境 (用于采集与推理)

SSH 登录到 RDK S600：

```bash
# 1. 同样克隆 Hugging Face LeRobot 和本工具仓库
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git clone https://github.com/D-Robotics/rdk_LeRobot_tools.git
cd rdk_LeRobot_tools && git checkout s600 && cd ..
pip install -e ".[feetech]"

# 2. 安装 BPU 运行时 (仅推理需要，但建议安装)
pip install hbm-runtime
```

---

## 2. 硬件配置与组装 (SO-101)

**提示：本章节操作可以在开发机上进行，也可以直接在 RDK S600 上连接屏幕或 SSH 进行！**

### 2.1 设置电机 ID (Set motor IDs)

在组装前，需要先设置每个电机的 ID。SO-101 主从手各需 6 个电机，ID 分别为 1-6。

**操作步骤：**
1.  每次只连接**一个**电机到转接板。
2.  使用 LeRobot v0.5.2 的电机设置命令，按提示逐个连接电机并设置 ID。Follower 示例：
    ```bash
    lerobot-setup-motors \
      --robot.type=so101_follower \
      --robot.port=/dev/ttyACM0
    ```
3.  Leader 示例：
    ```bash
    lerobot-setup-motors \
      --teleop.type=so101_leader \
      --teleop.port=/dev/ttyACM1
    ```
4.  根据命令行提示，每次只连接指定电机，依次完成 1-6 号电机的 ID 和波特率设置。

**电机设置演示视频：**
<video controls width="100%" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/setup_motors_so101_2.mp4" type="video/mp4"></video>

### 2.2 机械臂组装步骤

请参考 [SO-ARM100 官方指南](https://github.com/TheRobotStudio/SO-ARM100) 和 Hugging Face [SO-101 文档](https://huggingface.co/docs/lerobot/so101) 进行组装。以下是关键关节组装演示：

| Leader-Arm Axis | Motor | Gear Ratio |
|-----------------|:-------:|:----------:|
| Base / Shoulder Yaw | 1 | 1 / 191 |
| Shoulder Pitch      | 2 | 1 / 345 |
| Elbow               | 3 | 1 / 191 |
| Wrist Roll          | 4 | 1 / 147 |
| Wrist Pitch         | 5 | 1 / 147 |
| Gripper             | 6 | 1 / 147 |

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

### 2.3 查找端口 (RDK S600 推荐)

将组装好的机械臂连接到 RDK S600 的 USB 口。LeRobot v0.5.2 推荐使用：

```bash
lerobot-find-port
```
按提示拔插 USB 后记录端口号，例如 `/dev/ttyACM0`、`/dev/ttyACM1`。Linux 上如果串口权限不足，可临时执行：

```bash
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```

后续运行采集或推理脚本时，通过命令行参数传入端口；本工具的 `bpu_control_robot.py` 默认使用 `--robot-port /dev/ttyACM0`。

---

## 3. 校准 (Calibration)

**推荐在 RDK S600 上直接运行。**
校准是保证主从手同步和模型迁移有效的关键。LeRobot v0.5.2 使用 `lerobot-calibrate` 命令，流程与 Hugging Face 官方 [SO-101 文档](https://huggingface.co/docs/lerobot/so101) 一致：

1. 先把机械臂摆到各关节运动范围的中位。
2. 按回车后，再依次把每个关节完整转动一遍，记录其运动范围。

**校准演示视频：**

<video controls width="100%" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/calibrate_so101_2.mp4" type="video/mp4"></video>

### 3.1 校准从手 (Follower)

```bash
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=s600_follower
```

### 3.2 校准主手 (Leader)

```bash
lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=s600_leader
```

---

## 4. 摄像头配置 (Cameras)

**推荐在 RDK S600 上直接运行。**
LeRobot v0.5.2 不再通过 `configs.py` / `so101.yaml` 改相机，而是在 `lerobot-record` / `bpu_control_robot.py` 的命令行里直接传 `--robot.cameras` 或 `--camera-index`。

### 4.1 查找摄像头索引

连接所有 USB 摄像头到 RDK，运行：

```bash
lerobot-find-cameras
```

记下每个摄像头对应的 `index_or_path`，例如 `0`、`1`。

### 4.2 在命令行里配置相机

采集和推理时，把相机名和索引写进命令行。相机名要和后续训练、导出、板端推理保持一致，例如统一使用 `front`：

```bash
--robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}"
```

板端 BPU 推理时，对应参数是：

```bash
--camera-index 0 \
--camera-name front
```

`--camera-name` 必须和 `bpu_output/` 里的 `front_mean.npy` / `front_std.npy` 文件名一致。

---

## 5. 数据采集 (Data Collection)

**推荐在 RDK S600 上直接运行。**
收集高质量的演示数据是训练成功的关键。建议采集 **50 条** 以上的成功轨迹。

### 5.1 运行采集脚本

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

如果不指定 `--dataset.root`，数据默认会写到 `~/.cache/huggingface/lerobot/<repo_id>`。

### 5.2 关键参数详解

| 参数 | 含义 | 推荐值/说明 |
| :--- | :--- | :--- |
| `--robot.type` | 从手机械臂类型 | `so101_follower` |
| `--robot.port` | 从手串口 | 通过 `lerobot-find-port` 获取 |
| `--teleop.type` | 主手机械臂类型 | `so101_leader` |
| `--teleop.port` | 主手串口 | 通过 `lerobot-find-port` 获取 |
| `--robot.cameras` | 相机配置 | S600 USB 摄像头一般使用 `opencv` + `index_or_path` |
| `--dataset.repo_id` | 数据集 ID | 格式 `user/dataset_name` |
| `--dataset.root` | 本地保存路径 | 建议显式指定，便于后续拷贝到开发机训练 |
| `--dataset.num_episodes` | 计划采集总条数 | `50` 条起步，多多益善 |
| `--dataset.single_task` | 当前数据集任务描述 | 要与实际采集任务一致 |
| `--dataset.episode_time_s` | 单条数据最大时长 | 简单任务建议 `30-40` 秒 |
| `--dataset.reset_time_s` | 复位时间 | 建议 `5` 秒 |
| `--dataset.push_to_hub` | 是否上传 Hub | 本地调试建议 `false` |

### 5.3 键盘控制 (Keyboard Shortcuts)

在终端运行采集脚本时，可以使用键盘控制流程：

*   **右箭头 (`->`)**: 提前结束当前 episode 的录制，进入复位阶段 (Reset)。
*   **左箭头 (`<-`)**: 放弃当前 episode (不保存)，重新开始录制这一条。适用于操作失误的情况。
*   **ESC**: 提前结束整个采集任务，并开始保存数据。

### 5.4 数据集校验

采集完成后，务必检查数据是否有效（图像是否清晰，动作是否同步）。


## 6. 模型训练 (ACT Policy)

**此步骤必须在开发机 (带 GPU) 上运行。**
将 RDK 上采集好的数据集目录（例如 `/path/to/datasets/so101_pick_place`）拷贝到开发机。

### 6.1 安装训练依赖

```bash
conda activate lerobot
pip install -e ".[training]"
```

### 6.2 启动训练

LeRobot v0.5.2 推荐直接通过命令行传参，不再修改 `lerobot/configs/train.py`。

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

**参数详解**:
*   `--dataset.repo_id`: 数据集 ID，需与采集时一致。
*   `--dataset.root`: 本地数据集路径。
*   `--policy.type=act`: 使用 ACT 策略，网络结构会按数据集里的机器人和相机信息自动适配。
*   `--steps` / `--batch_size`: 训练步数和 batch size，直接通过 CLI 指定。
*   `--policy.device=cuda`: NVIDIA GPU 使用 `cuda`。
*   `--wandb.enable=true`: 开启 W&B（需先 `wandb login`）。

**恢复训练 (Resume Training)**:

如果训练中断，可以通过指定 checkpoint 的配置文件路径来恢复训练。例如，从 `act_so101_test` 任务的最新 checkpoint (`last`) 恢复：

```bash
lerobot-train \
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

S600 / SO100 ACT 可直接参考 `rdk_LeRobot_tools/bpu_export_config_s600_calfix.yaml`：

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

该配置会生成与 S600 运行时一致的校准数据：图像先从 `0..255` 缩放到 `0..1`，再执行 `(image - mean) / std`。

### 7.2 导出 ONNX 及编译配置

```bash
# 在开发机上运行
cd rdk_LeRobot_tools
python export_bpu_actpolicy.py --config bpu_export_config_s600_calfix.yaml
```

运行该脚本后，它会按顺序执行以下 6 个步骤：

**① 加载模型与数据集，自动检测相机**

脚本从 `act_path` 加载 PyTorch ACT checkpoint，从 `dataset.root` 读取数据集。然后从数据集中取一个 batch，扫描所有以 `observation.images.` 开头的字段，自动推断出相机名称（例如 `front`、`laptop`）。

**② 导出前后处理归一化参数**

从 checkpoint 目录中的 processor safetensors 文件读取训练时保存的统计量：
*   `{camera_name}_mean.npy` / `{camera_name}_std.npy`：图像归一化的均值和标准差。
*   `action_mean.npy` / `action_std.npy`：state 输入的归一化参数（来自 preprocessor）。
*   `action_mean_unnormalize.npy` / `action_std_unnormalize.npy`：action 输出的反归一化参数（来自 postprocessor）。

这些 `.npy` 文件会在板端推理时由 `bpu_control_robot.py` 加载，用于 BPU 外部的手动归一化/反归一化。

**③ 导出 VisionEncoder ONNX**

将 ACT 的 `backbone`（ResNet）和 `encoder_img_feat_input_proj`（特征投影层）提取出来，封装为 `BPU_ACTPolicy_VisionEncoder` 子模型。输入是一张归一化后的图像，输出是视觉特征图 `[1, 512, 15, 20]`。导出为 ONNX 后，若 `onnx_sim: true` 则自动调用 onnxsim 简化图结构。

**④ 导出 TransformerLayers ONNX**

将 ACT 的 encoder + decoder + action_head 部分封装为 `BPU_ACTPolicy_TransformerLayers` 子模型。它有两个（或多个）输入：
*   `states`：归一化后的 6 维关节状态 `[1, 6]`
*   `{camera_name}_features`：VisionEncoder 输出的视觉特征 `[1, 512, 15, 20]`

输出是 `Actions [1, 100, 6]`（ACT 的 100 步 action chunk）。同时保存一份 `new_actions.npy` 到 `bpu_output/`，用于精度验证。

**⑤ 生成 OE 编译配置和构建脚本**

为两个子模型分别生成：
*   `config_BPU_ACTPolicy_VisionEncoder.yaml` / `config_BPU_ACTPolicy_TransformerLayers.yaml`：OE `hb_compile` 使用的编译配置，包含 ONNX 路径、校准数据目录、`march: nash-p`、`norm_type: no_preprocess` 等。
*   `build_BPU_ACTPolicy_VisionEncoder.sh` / `build_BPU_ACTPolicy_TransformerLayers.sh`：各自的编译脚本。
*   `build_all.sh`：一键编译两个子模型的总入口脚本。

**⑥ 生成量化校准数据**

遍历训练数据集（最多 `cal_num` 个样本），对每个样本：
*   图像先做 `0..255 → /255.0 → (image - mean) / std`，保存为 VisionEncoder 的校准数据。
*   将归一化后的图像送入 VisionEncoder 前向推理，得到视觉特征，保存为 Transformer 的 `{camera_name}/` 校准数据。
*   归一化后的 state 保存为 Transformer 的 `state/` 校准数据。

*成功标志：`export_path` 指定的目录中生成了如下结构：*

```
export_path/
├── BPU_ACTPolicy_VisionEncoder/
│   ├── BPU_ACTPolicy_VisionEncoder.onnx
│   ├── config_BPU_ACTPolicy_VisionEncoder.yaml
│   ├── calibration_data_BPU_ACTPolicy_VisionEncoder/
│   └── build_BPU_ACTPolicy_VisionEncoder.sh
├── BPU_ACTPolicy_TransformerLayers/
│   ├── BPU_ACTPolicy_TransformerLayers.onnx
│   ├── config_BPU_ACTPolicy_TransformerLayers.yaml
│   ├── calibration_data_BPU_ACTPolicy_TransformerLayers/
│   │   ├── state/
│   │   └── front/
│   └── build_BPU_ACTPolicy_TransformerLayers.sh
├── bpu_output/
│   ├── action_mean.npy / action_std.npy
│   ├── action_mean_unnormalize.npy / action_std_unnormalize.npy
│   ├── front_mean.npy / front_std.npy
│   └── new_actions.npy
└── build_all.sh
```

### 7.3 编译 BPU 模型 (OpenExplorer Docker 环境)

1.  **安装 Docker**
    *   按照官方说明安装并验证： [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)
    *   验证：  
        ```bash
        sudo docker --version
        sudo docker run --rm hello-world
        ```

2.  **获取并加载离线镜像**（S600 推荐 OE 3.7.0 S100/S600 CPU 镜像）
    *   工具链版本发布汇总帖（持续更新）：[https://forum.d-robotics.cc/t/topic/35229](https://forum.d-robotics.cc/t/topic/35229)
    *   下载离线镜像包：
        ```bash
        wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe/3.7.0/ai_toolchain_ubuntu_22_s100_s600_cpu_v3.7.0.tar
        ```
    *   加载镜像：
        ```bash
        sudo docker load -i ai_toolchain_ubuntu_22_s100_s600_cpu_v3.7.0.tar
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
        - S600 推荐镜像名：`registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0`

4.  **在容器内编译模型**
    *   进入挂载目录并执行编译脚本：
        ```bash
        cd /workspace/bpu_export_act_so100_s600_calfix
        bash build_all.sh
        ```
    *   编译输出通常位于 `export_path` 下的 `bpu_output/` 和各子模型目录中（根据脚本输出确认）。

5.  **常见问题与排查**
    *   **权限问题**：宿主机复制回文件时出现权限错误，检查文件属主或使用 `sudo chown -R`。
    *   **磁盘空间不足**：编译会产生较大临时文件，确保宿主机有足够磁盘空间。
    *   **内存/IPC 报错**：增加 `--shm-size`（例如 15g）或适当增加容器内存限制。
    *   **镜像名不确定**：运行 `sudo docker images` 查看加载的镜像标签与 ID。
    *   若需要长期保留容器产物，请不要使用 `--rm` 或把 outputs 写到宿主机挂载目录。

**示例完整流程：**
```bash
# 直接在宿主机运行一次性编译命令
docker run --rm \
  -v /path/to/bpu_export_act_so100_s600_calfix:/workspace \
  -w /workspace \
  registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0 \
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
    |-- front_mean.npy      # 相机名与采集/导出时一致
    |-- front_std.npy
    |-- new_actions.npy
    `-- ...
```

完成后，请将生成的 `bpu_output` 文件夹拷贝到 RDK 板端用于部署。

---

## 8. 板端部署与推理 (RDK S600)

### 前提条件
1.  已安装 [huggingface/lerobot](https://github.com/huggingface/lerobot) 和 `hbm-runtime`。
2.  已将 **`bpu_output`** 文件夹（包含量化后的 `.hbm` 模型和校准参数）传输到板端。
3.  **硬件配置**: 确保机械臂端口、相机索引、相机名称与训练/导出时一致；校准文件由 `lerobot-calibrate` 自动保存到 `~/.cache/huggingface/lerobot/calibration/`。

### 运行 BPU 加速推理

这是将训练好的模型部署到 RDK 上的最终步骤。

1.  **文件传输**: 将开发机生成的 `bpu_output` 文件夹拷贝到 RDK 板子。
2.  **运行推理**：

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

    当前 `bpu_control_robot.py` 默认连接 **SO100Follower**，不是 `so101`。如果你实际部署的是 SO-101 从手，需要先确认 LeRobot 机器人类型与机械臂一致。

    ACT 一次推理会输出 100 步 action chunk，脚本会自动从 `new_actions.npy` 推断 `n_action_steps`。不要为了调试传 `--n-action-steps 1`，否则会改变 ACT 的运行语义。

### 故障排查

*   **机械臂不动**: 检查 `ls /dev/ttyACM*`；确认 `--robot-port` 正确。
*   **相机报错**: 确认 `--camera-index` 和 `--camera-name` 与 `bpu_output/*_mean.npy` 一致。

### BPU 推理性能基准

在 RDK S600 上对 ACT 模型各模块进行纯 BPU 性能测试（20 次 warmup + 200 次正式采样）：

| 模块 | 平均推理时间 | 帧率 |
| :--- | :--- | :--- |
| VisionEncoder | 3.92 ms | 255.0 inf/s |
| TransformerLayers | 2.29 ms | 436.4 inf/s |
| **完整 ACT** | **6.20 ms** | **161.2 inf/s** |

ACT 一次输出 100 步 action chunk，因此在 30 fps 控制频率下，每 3.33 秒仅需一次 BPU 推理（6.20 ms），其余时间 BPU 处于空闲状态。