[English](./README.md) | 简体中文
# RDK LeRobot Tools

**此版本为 STABLE (稳定) 版本，主要适配较旧版本 LeRobot（兼容 v2.1 数据集）。新版本 LeRobot 请切换到对应的分支。**

**注意：本工具目前仅在 RDK S100 上验证了 ACT 模型的部署效果，其他硬件平台或模型架构的效果无法保证。**

本仓库提供了一套工具，用于将基于 [LeRobot](https://github.com/D-Robotics/lerobot) 框架训练的 ACT 策略模型导出并部署到地瓜机器人 RDK S100 上，利用 BPU 进行高效推理。

全流程文档可以参考：👉 *[全流程文档](./doc/WORKFLOW_GUIDE_CN.md)*

## 目录结构

*   `damo/`: 适配 DAMO 开发者矩阵-乐云具身智能开发平台的工具包。
*   `export_bpu_actpolicy.py`: **模型导出脚本**（在开发机/训练服务器上运行）。用于将 PyTorch 权重转换为 ONNX 并生成 BPU 编译所需的配置文件和脚本。
*   `bpu_export_config.yaml`: 模型导出配置文件。
*   `bpu_control_robot.py`: **板端部署脚本**（在 RDK 板端运行）。加载编译好的 BPU 模型并控制机器人。

## 1. 环境准备

### 1.1 开发机 (用于模型转换)

**严格推荐**使用 D-Robotics 提供的 LeRobot 仓库搭建开发环境，以确保最佳兼容性：
👉 **https://github.com/D-Robotics/lerobot**
```bash
    git clone https://github.com/D-Robotics/lerobot.git
    cd lerobot
    git clone https://github.com/D-Robotics/rdk_LeRobot_tools.git
    pip install -e ".[feetech]"
```
此版本兼容 v2.1 数据集，本仓库的导出工具只要能加载 v2.1 数据集的历史版本均可工作。

或者clone官方仓库后切换到对应的旧版本分支：
```bash
    git clone https://github.com/huggingface/lerobot.git
    cd lerobot
    git checkout 8cfab3882480bdde38e42d93a9752de5ed42cae2  # 切换到 v2.1 版本对应的 commit
    git clone https://github.com/D-Robotics/rdk_LeRobot_tools.git
    pip install -e ".[feetech]"
```

**特别注意：** LeRobot 较新版本对旧代码可能存在兼容性问题，导致报错。D-Robotics fork 的 LeRobot 仓库已将 `datasets` 库版本锁定。如果您是直接 Clone 官方 LeRobot 仓库并切换到旧版，可能需要手动将 `datasets` 库降级到 `datasets==2.19.0` 以避免兼容性问题。

需安装以下 Python 包用于 ONNX 导出和处理：

```bash
pip install onnx onnxsim termcolor tqdm
```

*注意：模型编译（ONNX -> HBM）需要在地瓜机器人提供的 Docker 工具链环境（OpenExplorer）中进行。*

### 1.2 RDK 板端 (用于模型部署)

板端运行环境对稳定性要求较高，请务必 Clone `D-Robotics/lerobot` 仓库的指定版本：

1.  **安装 LeRobot (D-Robotics Fork 版)**:
    ```bash
    git clone https://github.com/D-Robotics/lerobot.git
    cd lerobot
    pip install -e ".[feetech]"
    # D-Robotics fork 版本已锁定 datasets 依赖，无需手动操作。
    # 若您使用其他 LeRobot 仓库版本并遇到兼容性问题，可能需手动安装 datasets==2.19.0
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
        *   `nash-e` / `nash-m` / `nash-p`: 适用于 RDK S100 等 Nash 架构。
        *   `bayes` / `bayes-e`: 适用于 RDK X5 等 Bayes 架构。

2.  **运行导出脚本**:
    ```bash
    python export_bpu_actpolicy.py --config bpu_export_config.yaml
    ```
    运行成功后，会在 `bpu_export_output` (或配置指定的目录) 下生成 ONNX 模型、校准数据和编译脚本 (`build_all.sh`)。

    **重要提示：** 针对较新版本的 LeRobot (v2.1)，为避免导出时缺少关键字键 (`policy.type` 等)，请务必放开 `bpu_export_config.yaml` 中对应 `policy` 和 `dataset` 部分的注释。具体请参考 `bpu_export_config.yaml` 模板文件中的说明。

### 第二步：编译 BPU 模型

进入工具链 Docker 环境，运行上一步生成的编译脚本：

```bash
cd bpu_export_output
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