[English](./README.md) | 简体中文
# RDK LeRobot Tools

**此 `s600` 分支用于 LeRobot v0.5.2 的 ACT 模型导出与 RDK S600 BPU 部署验证。**

**注意：S600 请使用 `nash-p` / OE 3.7.0 S100/S600 工具链。**

本仓库提供了一套工具，用于将基于 [LeRobot](https://github.com/D-Robotics/lerobot) 框架训练的 ACT 策略模型导出并部署到地瓜机器人 RDK S600 上，利用 BPU 进行高效推理。

全流程文档可以参考：👉 *[全流程文档](./doc/WORKFLOW_GUIDE_CN.md)*

## 目录结构

*   `damo/`: 适配 DAMO 开发者矩阵-乐云具身智能开发平台的工具包。
*   `export_bpu_actpolicy.py`: **模型导出脚本**（在开发机/训练服务器上运行）。用于将 PyTorch 权重转换为 ONNX 并生成 BPU 编译所需的配置文件和脚本。
*   `bpu_export_config.yaml`: 模型导出配置文件。
*   `bpu_export_config_s600_calfix.yaml`: LeRobot v0.5.2 / SO100 ACT / S600 (`nash-p`) 的示例导出配置。
*   `bpu_control_robot.py`: **板端部署脚本**（在 RDK 板端运行）。加载编译好的 BPU 模型并控制机器人。

## 1. 环境准备

### 1.1 开发机 (用于模型转换)

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
cd /path/to/lerobot
pip install -e ".[feetech]"
```

也可以使用 D-Robotics 提供的 LeRobot 仓库搭建开发环境，以确保最佳兼容性：
👉 **https://github.com/D-Robotics/lerobot**
```bash
    git clone https://github.com/D-Robotics/lerobot.git
    cd lerobot
    git clone https://github.com/D-Robotics/rdk_LeRobot_tools.git
    pip install -e ".[feetech]"
```

需安装以下 Python 包用于 ONNX 导出和处理：

```bash
pip install onnx onnxsim termcolor tqdm safetensors
```

*注意：模型编译（ONNX -> HBM）需要在地瓜机器人提供的 Docker 工具链环境（OpenExplorer）中进行。*

### 1.2 RDK 板端 (用于模型部署)

板端运行环境对稳定性要求较高，请务必 Clone `D-Robotics/lerobot` 仓库的指定版本：

1.  **安装 LeRobot (D-Robotics Fork 版)**:
    ```bash
    git clone https://github.com/D-Robotics/lerobot.git
    cd lerobot
    pip install -e ".[feetech]"
    # 本分支验证环境使用 datasets 4.8.5。
    # 若使用其他 LeRobot 版本，请以对应 checkpoint 和数据集能被当前环境正确加载为准。
    ```

2.  **安装 BPU 推理库**:
    ```bash
    pip install hbm-runtime
    ```

## 2. 模型导出与编译 (在开发机上执行)

此过程分为两步：首先导出 ONNX 和配置，然后使用地平线工具链编译为 BPU 模型。

### 第一步：导出 ONNX 及配置

1.  **修改配置文件**:
    编辑 `bpu_export_config.yaml`，根据实际情况修改以下关键字段：
    *   `dataset.root`: 训练时使用的数据集根目录。
    *   `act_path`: 训练好的 ACT 模型检查点路径 (包含 `config.json` 和 `model.safetensors`)。
    *   `type`: BPU 平台类型。脚本会自动根据此类型调整编译参数。
        *   `nash-e` / `nash-m` / `nash-p`: 适用于 Nash 架构；S600 使用 `nash-p`。
        *   `bayes` / `bayes-e`: 适用于 RDK X5 等 Bayes 架构。

    S600 / SO100 ACT 可以参考 `bpu_export_config_s600_calfix.yaml`。该配置使用：
    *   `type: "nash-p"`
    *   `cal_num: 100`
    *   `export_path: ".../bpu_export_act_so100_s600_calfix"`

2.  **运行导出脚本**:
    ```bash
    python export_bpu_actpolicy.py --config bpu_export_config.yaml
    ```
    运行成功后，会在 `bpu_export_output` (或配置指定的目录) 下生成 ONNX 模型、校准数据和编译脚本 (`build_all.sh`)。

    S600 示例：
    ```bash
    python export_bpu_actpolicy.py --config bpu_export_config_s600_calfix.yaml
    ```

    **重要提示：** 导出脚本会从 LeRobot v0.5.2 checkpoint 的 processor safetensors 中读取图像、state 和 action 的归一化参数。图像校准数据会先确保输入是 `0..1` float，再做 `(image - mean) / std`，从而和板端运行时的 `uint8 -> /255.0 -> normalize` 保持一致。

### 第二步：编译 BPU 模型

进入工具链 Docker 环境，运行上一步生成的编译脚本：

```bash
cd bpu_export_output
bash build_all.sh
```

S600 推荐使用 OE 3.7.0 S100/S600 Docker 工具链，例如：

```bash
docker run --rm \
  -v /path/to/bpu_export_act_so100_s600_calfix:/workspace \
  -w /workspace \
  registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0 \
  bash build_all.sh
```

编译完成后，`bpu_export_output` 目录下会生成一个 **`bpu_output`** 文件夹。这个文件夹包含了：
*   `.hbm` / `.bin`: 编译好的 BPU 模型文件（可在 BPU 上运行）。
*   `.npy`: 运行时所需的归一化参数。
*   `new_actions.npy`: 转换前的模型推理结果（用于精度验证）。

**请将 `bpu_output` 文件夹拷贝到 RDK 板端。**

预计产物为：

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

## 3. 板端推理 (在 RDK 上执行)

板端推理的核心是使用 **`bpu_control_robot.py`** 脚本。

### 前提条件
1.  已安装 `D-Robotics/lerobot` 仓库的 LeRobot 和 `hbm_runtime`。
2.  已将 **`bpu_output`** 文件夹（包含量化后的 `.hbm` 模型和校准参数）传输到板端。
3.  **硬件配置**: 请参考官方的数据采集和遥操作步骤，完成 `config` 文件的配置，确保**机械臂端口号**、**相机端口号**及**校准文件**配置正确。

### 运行步骤

1.  连接机器人（默认配置为 `so101`）。
2.  运行控制脚本，指定模型路径：

    ```bash
    # 假设 bpu_output 在当前目录下
    python bpu_control_robot.py --bpu-act-path ./bpu_output
    ```

### 常见参数

*   `--bpu-act-path`: BPU 模型文件夹路径 (必须包含 `.hbm` 和 `.npy` 文件)。
*   `--fps`: 控制循环频率 (默认 30Hz)。
*   `--inference-time`: 自动运行的持续时间 (秒)。

## 4. DAMO 平台模型适配

如果您使用的是 **DAMO 开发者矩阵-乐云具身智能开发平台** 采集的数据和训练的模型，在进行 BPU 模型导出前，**必须** 对数据集进行格式适配。

**操作步骤：**

1.  **备份数据**：该操作会直接修改源文件，请务必先备份您的数据集文件夹。
2.  **编辑脚本**：打开 `damo/replace.py`，修改 `folder_path` 为数据集路径。
3.  **运行转换**：`python damo/replace.py`

## 注意事项

*   **模型兼容性**: 板端运行必须使用经过 OE 工具链量化并编译的 `.hbm` / `.bin` 模型，不能直接运行 ONNX 或 PyTorch 模型。
*   **机器人配置**: `bpu_control_robot.py` 默认连接 `so101` 机器人。如需更改，请修改代码中的 `make_robot("so101")`。
*   **S600 校准一致性**: 对于 LeRobot v0.5.2 的 ACT 模型，图像 calibration 必须与运行时预处理一致，即 `uint8 -> /255.0 -> (image - mean) / std`。如果直接用 `0..255` 图像做 ImageNet normalize，会导致 Vision/Transformer 量化范围错误，BPU 输出动作可能严重偏离 PyTorch/ONNX。
*   **ACT chunk**: ACT 一次输出 100 步 action chunk，板端不要为了规避问题传 `--n-action-steps 1`。