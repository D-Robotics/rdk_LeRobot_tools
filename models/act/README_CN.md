[English](./README.md) | 简体中文

# RDK S600 上的 ACT Policy

本文说明 ACT 模型导出、BPU 编译、量化校准，以及 SO100 在 RDK S600 上的板端推理流程。

**Pick and Place 演示：**

<div align="center">
  <img src="../../doc/assets/demo_pick_place.gif" width="480" alt="Pick and Place 演示" />
</div>

> 需要说明的是，这个演示只是简单的 Pick and Place 展示，仅采集了 33 组训练数据。以下是 6 组训练数据（Episode 0/6/13/20/26/32）的并排可视化：

<div align="center">
  <img src="../../doc/assets/demo_episodes_grid.gif" width="640" alt="训练数据可视化 - 6 组 Episode 并排展示" />
</div>

本目录中的 ACT 正式实现用于把基于 [Hugging Face LeRobot](https://github.com/huggingface/lerobot) 训练的策略导出并部署到地瓜机器人 RDK S600 BPU。除非特别说明，后续命令均从仓库根目录执行。

全流程文档可以参考：👉 *[全流程文档](../../doc/WORKFLOW_GUIDE_CN.md)*

## ACT 目录结构

- `models/act/export_bpu_actpolicy.py`：**ACT 模型导出的正式实现**。根目录 `export_bpu_actpolicy.py` 作为向后兼容入口保留。
- `bpu_export_config_s600_calfix.yaml`: **S600 推荐配置**。LeRobot v0.5.2 / SO100 ACT / `nash-p`。
- `bpu_export_config.yaml`: 其他平台通用模板（默认 `nash-e`，S600 不要用）。
- `models/act/bpu_control_robot.py`：**ACT 板端运行的正式实现**。根目录 `bpu_control_robot.py` 作为向后兼容入口保留。

## 1. 环境准备

### 1.1 开发机 (用于模型转换)

请使用 Hugging Face 官方 LeRobot 仓库搭建环境：
👉 **[https://github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)**

不要使用 D-Robotics 维护的 `D-Robotics/lerobot` 仓库，该 fork 版本较旧。

本分支验证时使用的是 **LeRobot v0.5.2**。实际验证环境中的关键版本为：

```text
datasets 4.8.5
torch 2.7.1+cu126
onnxruntime 1.26.0
onnx 1.21.0
numpy 2.2.6
```

推荐在独立的 conda 环境中安装：

```bash
conda activate lerobot
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git clone https://github.com/D-Robotics/rdk_LeRobot_tools.git
cd rdk_LeRobot_tools && git checkout s600 && cd ..
pip install -e ".[feetech]"
pip install onnx onnxsim termcolor tqdm safetensors
```

*注意：模型编译（ONNX -> HBM）需要在地瓜机器人提供的 Docker 工具链环境（OpenExplorer）中进行。*

### 1.2 RDK 板端 (用于模型部署)

板端请同样使用 Hugging Face 官方 `huggingface/lerobot` 仓库：

1. **安装 LeRobot 和本工具仓库**:
  ```bash
    git clone https://github.com/huggingface/lerobot.git
    cd lerobot
    git clone https://github.com/D-Robotics/rdk_LeRobot_tools.git
    cd rdk_LeRobot_tools && git checkout s600 && cd ..
    pip install -e ".[feetech]"
    # 本分支验证环境使用 datasets 4.8.5。
    # 不要使用 D-Robotics/lerobot 仓库。
  ```
2. **安装 BPU 推理库**:
  ```bash
    pip install hbm-runtime
  ```

## 2. 模型导出与编译 (在开发机上执行)

此过程分为两步：首先由 `export_bpu_actpolicy.py` 从 PyTorch checkpoint 导出 ONNX 和编译配置，然后由 OE 工具链把 ONNX 编译成 `.hbm`。

### 导出链路说明

仓库里其实有**两套配置文件**，作用不同：


| 配置文件                                 | 谁使用                       | 作用                                          |
| ------------------------------------ | ------------------------- | ------------------------------------------- |
| `bpu_export_config_s600_calfix.yaml` | `export_bpu_actpolicy.py` | 指定 checkpoint、数据集、导出目录、`nash-p`、`cal_num` 等 |
| `config_BPU_ACTPolicy_*.yaml`        | OE 工具链 `hb_compile`       | 指定 ONNX 路径、校准数据、量化/编译参数                     |


也就是说：**你手写的是“导出阶段配置”；工具链吃的是导出脚本自动生成的 `config_*.yaml`。**

OE 工具链本身只接受 ONNX，不会直接读 PyTorch checkpoint。因此标准流程是：

```text
bpu_export_config_s600_calfix.yaml
        ↓
export_bpu_actpolicy.py
        ↓
export_path/
├── BPU_ACTPolicy_VisionEncoder/
│   ├── BPU_ACTPolicy_VisionEncoder.onnx
│   ├── config_BPU_ACTPolicy_VisionEncoder.yaml
│   └── calibration_data_BPU_ACTPolicy_VisionEncoder/
├── BPU_ACTPolicy_TransformerLayers/
│   ├── BPU_ACTPolicy_TransformerLayers.onnx
│   ├── config_BPU_ACTPolicy_TransformerLayers.yaml
│   └── calibration_data_BPU_ACTPolicy_TransformerLayers/
├── bpu_output/                  # 运行时归一化参数 + 最终 .hbm
└── build_all.sh
        ↓
OE Docker: hb_compile
        ↓
bpu_output/*.hbm
```

#### 为什么 ACT 要拆成两个 ONNX？

完整 ACT 链路是：

```text
图像 → VisionEncoder → front_features
state + front_features → Transformer → Actions [1, 100, 6]
```

为了适配 BPU 工具链，导出脚本会把 ACT 拆成两个子模型：

1. **VisionEncoder**：`images` → `Vision_Features`
2. **TransformerLayers**：`states` + `front_features` → `Actions`

板端再由 `bpu_control_robot.py` 串起来运行。

#### `export_bpu_actpolicy.py` 会做什么？

运行 `python export_bpu_actpolicy.py --config bpu_export_config_s600_calfix.yaml` 后，脚本会依次执行以下 6 个步骤：

**① 加载模型与数据集，自动检测相机**

从 `act_path` 加载 PyTorch ACT checkpoint，从 `dataset.root` 读取数据集。取一个 batch，扫描所有 `observation.images.`* 字段，自动推断出相机名称（如 `front`）。

**② 导出前后处理归一化参数**

从 checkpoint 中的 processor safetensors 文件读取训练时保存的统计量，保存到 `bpu_output/*.npy`：

- `{camera_name}_mean.npy` / `{camera_name}_std.npy`：图像归一化均值和标准差。
- `action_mean.npy` / `action_std.npy`：state 输入的归一化参数。
- `action_mean_unnormalize.npy` / `action_std_unnormalize.npy`：action 输出的反归一化参数。

这些 `.npy` 在板端由 `bpu_control_robot.py` 加载，用于 BPU 外部的手动归一化/反归一化。

**③ 导出 VisionEncoder ONNX**

将 ACT 的 `backbone`（ResNet）和 `encoder_img_feat_input_proj` 封装为 `BPU_ACTPolicy_VisionEncoder`。输入归一化后的图像，输出视觉特征图 `[1, 512, 15, 20]`。若 `onnx_sim: true` 则调用 onnxsim 简化图结构。

**④ 导出 TransformerLayers ONNX**

将 encoder + decoder + action_head 封装为 `BPU_ACTPolicy_TransformerLayers`。输入 `states` 和 VisionEncoder 输出的 `{camera_name}_features`，输出 `Actions [1, 100, 6]`。同时保存 `new_actions.npy` 用于精度验证。

**⑤ 生成 OE 编译配置和构建脚本**

为两个子模型分别生成 `config_BPU_ACTPolicy_*.yaml`、`build_*.sh` 和一键编译入口 `build_all.sh`。配置中包含 `march: nash-p`、`norm_type: no_preprocess` 等参数。

**⑥ 生成量化校准数据**

遍历训练数据集（最多 `cal_num` 个样本）：

- 图像 `0..255 → /255.0 → (image - mean) / std` → VisionEncoder 校准数据。
- 图像送入 VisionEncoder 前向推理 → 视觉特征 → Transformer `{camera_name}/` 校准数据。
- 归一化后的 state → Transformer `state/` 校准数据。

图像校准数据会先判断输入是否为 `0..255`，必要时 `/255.0`，再执行 `(image - mean) / std`，与板端 `uint8 -> /255.0 -> normalize` 保持一致。

#### `config_BPU_ACTPolicy_VisionEncoder.yaml` 含义

```yaml
model_parameters:
  onnx_model: BPU_ACTPolicy_VisionEncoder.onnx   # 要编译的 ONNX
  march: nash-p                                  # S600 目标架构
calibration_parameters:
  cal_data_dir: calibration_data_BPU_ACTPolicy_VisionEncoder
  cal_data_type: float32
input_parameters:
  norm_type: no_preprocess                       # 归一化在 BPU 外完成
compiler_parameters:
  jobs: 6                                        # 对应 combine_jobs
```

这里的 `cal_data_dir` 存放的是**已经归一化好的图像 tensor**（例如 `front_0000000000.npy`），供量化时统计数值范围。

#### `config_BPU_ACTPolicy_TransformerLayers.yaml` 含义

```yaml
input_parameters:
  input_name: states;front_features;
  norm_type: no_preprocess;no_preprocess;
calibration_parameters:
  cal_data_dir: calibration_data_.../state;calibration_data_.../front;
```

Transformer 有两个输入：


| 输入名              | 校准数据内容                                  |
| ---------------- | --------------------------------------- |
| `states`         | 归一化后的关节状态 `[1, 6]`                      |
| `front_features` | VisionEncoder 输出的特征图 `[1, 512, 15, 20]` |


注意：Transformer 的校准数据**不是原始图像**，而是 Vision 子模型跑出来的 feature。

#### 板端运行时的数据流

```text
相机 uint8
  → /255.0
  → (image - front_mean) / front_std
  → VisionEncoder.hbm
  → front_features
  → (state - action_mean) / action_std
  → TransformerLayers.hbm
  → Actions [100, 6]
  → 反归一化
  → 发给机械臂
```

因为归一化都在 BPU 外完成，所以两个 `config_*.yaml` 里都写 `norm_type: no_preprocess`。

### 第一步：导出 ONNX 及配置

1. **修改配置文件**:
  S600 / SO100 ACT 请直接编辑 `bpu_export_config_s600_calfix.yaml`：
  - `dataset.root`: 训练时使用的数据集根目录。
  - `act_path`: 训练好的 ACT checkpoint 路径。
  - `export_path`: 导出输出目录。
  - `type`: S600 固定为 `nash-p`。
  - `cal_num`: 建议 `100`。
2. **运行导出脚本**:
  ```bash
    cd rdk_LeRobot_tools
    python export_bpu_actpolicy.py --config bpu_export_config_s600_calfix.yaml
  ```
    运行成功后，会在 `export_path` 指定目录下生成 ONNX、校准数据和 `build_all.sh`。
    **重要提示：** 导出脚本会从 LeRobot v0.5.2 checkpoint 的 processor safetensors 中读取图像、state 和 action 的归一化参数。图像校准数据会先确保输入是 `0..1` float，再做 `(image - mean) / std`，从而和板端运行时的 `uint8 -> /255.0 -> normalize` 保持一致。

### 第二步：编译 BPU 模型

进入工具链 Docker 环境，运行上一步生成的编译脚本：

```bash
cd /path/to/bpu_export_act_so100_s600_calfix
bash build_all.sh
```

S600 推荐使用 OE 3.7.0 S100/S600 Docker 工具链。工具链版本发布汇总（持续更新）：[https://forum.d-robotics.cc/t/topic/35229](https://forum.d-robotics.cc/t/topic/35229)

```bash
# 下载离线镜像包
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe/3.7.0/ai_toolchain_ubuntu_22_s100_s600_cpu_v3.7.0.tar
# 加载镜像
sudo docker load -i ai_toolchain_ubuntu_22_s100_s600_cpu_v3.7.0.tar
```

运行编译：

```bash
docker run --rm \
  -v /path/to/bpu_export_act_so100_s600_calfix:/workspace \
  -w /workspace \
  registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0 \
  bash build_all.sh
```

编译完成后，导出目录下会生成 `**bpu_output**` 文件夹。这个文件夹包含了：

- `.hbm` / `.bin`: 编译好的 BPU 模型文件（可在 BPU 上运行）。
- `.npy`: 运行时所需的归一化参数。
- `new_actions.npy`: 转换前的模型推理结果（用于精度验证）。

**请将 `bpu_output` 文件夹拷贝到 RDK 板端。**

预计产物为：

```
bpu_output/
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

## 3. 板端推理 (在 RDK 上执行)

板端推理的核心是使用 `**bpu_control_robot.py**` 脚本。

### 前提条件

1. 已安装 [huggingface/lerobot](https://github.com/huggingface/lerobot) 和 `hbm-runtime`。
2. 已将 `**bpu_output**` 文件夹（包含量化后的 `.hbm` 模型和校准参数）传输到板端。
3. **硬件配置**: 通过命令行传入机械臂端口、相机索引和相机名称；校准文件由 `lerobot-calibrate` 保存。

### 运行步骤

1. 连接机器人。当前脚本默认使用 **SO100Follower**。
2. 运行控制脚本：
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

### 常见参数

- `--bpu-act-path`: BPU 模型文件夹路径 (必须包含 `.hbm` 和 `.npy` 文件)。
- `--robot-port`: 从手串口，默认 `/dev/ttyACM0`。
- `--camera-index` / `--camera-name`: 相机索引和名称，需与 `bpu_output/*_mean.npy` 一致。
- `--fps`: 控制循环频率 (默认 30Hz)。
- `--inference-time`: 自动运行的持续时间 (秒)。

### BPU 推理性能基准

在 RDK S600 上对 ACT 模型各模块进行纯 BPU 性能测试（20 次 warmup + 200 次正式采样）：


| 模块                | 平均推理时间      | 帧率              |
| ----------------- | ----------- | --------------- |
| VisionEncoder     | 3.92 ms     | 255.0 inf/s     |
| TransformerLayers | 2.29 ms     | 436.4 inf/s     |
| **完整 ACT**        | **6.20 ms** | **161.2 inf/s** |


ACT 一次输出 100 步 action chunk，因此在 30 fps 控制频率下，每 3.33 秒仅需一次 BPU 推理（6.20 ms），其余时间 BPU 处于空闲状态。

## 4. 数据集格式说明 (v3.0 vs v2.1)

本分支基于 LeRobot v0.5.2，数据集格式为 `**codebase_version: v3.0`**；`stable` 分支兼容的是 **v2.1** 格式。两者的主要区别：


| 维度                 | v2.1 (stable 分支)                                  | v3.0 (本分支 s600)                                           |
| ------------------ | ------------------------------------------------- | --------------------------------------------------------- |
| **组织方式**           | 一 episode 一文件                                     | 多 episode 合并进大文件                                          |
| **数据文件**           | `data/chunk-000/episode_000000.parquet`           | `data/chunk-000/file-000.parquet`                         |
| **视频文件**           | `videos/chunk-000/camera/episode_000000.mp4`      | `videos/camera/chunk-000/file-000.mp4`                    |
| **元数据**            | JSON Lines (`meta/episodes.jsonl`, `tasks.jsonl`) | Parquet (`meta/episodes/*.parquet`, `meta/tasks.parquet`) |
| **episode 统计**     | 独立 `episodes_stats.jsonl`                         | 合并进 `meta/episodes/*.parquet`                             |
| `**next.done` 字段** | 有                                                 | 已移除，靠元数据推断 episode 边界                                     |


核心数据字段（`observation.state`、`action`、`timestamp`、`episode_index` 等）语义不变，变化的是存储方式。v3.0 将多条 episode 合并为少量大文件，通过 `meta/episodes/` 中的偏移量索引 episode，适合大规模数据集（百万级 episode），初始化更快、文件系统压力更小。

如果你的训练数据是 v2.1 格式，需要先用官方脚本转换：

```bash
python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id=<your-repo-id> --root=<dataset-path>
```

## 注意事项

- **模型兼容性**: 板端运行必须使用经过 OE 工具链量化并编译的 `.hbm` / `.bin` 模型，不能直接运行 ONNX 或 PyTorch 模型。
- **机器人配置**: `bpu_control_robot.py` 当前硬编码为 `SO100Follower`。如果部署 SO-101，需要先改代码中的机器人类型。
- **S600 校准一致性**: 对于 LeRobot v0.5.2 的 ACT 模型，图像 calibration 必须与运行时预处理一致，即 `uint8 -> /255.0 -> (image - mean) / std`。如果直接用 `0..255` 图像做 ImageNet normalize，会导致 Vision/Transformer 量化范围错误，BPU 输出动作可能严重偏离 PyTorch/ONNX。
- **ACT chunk**: ACT 一次输出 100 步 action chunk，板端不要为了规避问题传 `--n-action-steps 1`。
