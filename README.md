# RDK LeRobot Tools

本仓库提供了一套工具，用于将基于 [LeRobot](https://github.com/huggingface/lerobot) 框架训练的 ACT 策略模型导出并部署到地平线 RDK 系列开发板（如 RDK X5, RDK X5E 等）上，利用 BPU 进行高效推理。

## 目录结构

*   `damo/`: 适配 DAMO 开发者矩阵-乐云具身智能开发平台的工具包。
*   `export_bpu_actpolicy.py`: **模型导出脚本**（在开发机/训练服务器上运行）。用于将 PyTorch 权重转换为 ONNX 并生成 BPU 编译所需的配置文件和脚本。
*   `bpu_export_config.yaml`: 模型导出配置文件。
*   `bpu_control_robot.py`: **板端部署脚本**（在 RDK 板端运行）。加载编译好的 BPU 模型并控制机器人。

## 1. 环境准备

### 1.1 开发机 (用于模型转换)

需安装 `lerobot` 及其依赖，并补充安装以下 Python 包（用于 ONNX 导出和简化）：

```bash
pip install onnx onnxsim termcolor
```

*注意：模型编译（ONNX -> HBM）通常需要在地平线提供的 Docker 工具链环境中进行。*

### 1.2 RDK 板端 (用于模型部署)

1.  **安装 LeRobot**:
    为了保证最佳兼容性，必须使用以下经过验证的 commit 版本：

    ```bash
    git clone https://github.com/huggingface/lerobot.git
    cd lerobot
    git checkout 8cfab3882480bdde38e42d93a9752de5ed42cae2
    pip install -e .
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
    *   `type`: BPU 平台类型 (例如 `nash-e` 对应 RDK X5)。

    ```yaml
    dataset:
      root: "/path/to/your/dataset"
    
    act_path: "/path/to/your/pretrained_model"
    
    type: "nash-e" # 可选: nash-e, nash-m, nash-p, bayes-e, bayes
    ```

2.  **运行导出脚本**:

    ```bash
    python export_bpu_actpolicy.py --config bpu_export_config.yaml
    ```

    运行成功后，会在 `bpu_export_output` (或配置指定的目录) 下生成 ONNX 模型、校准数据和编译脚本 (`build_all.sh`)。

### 第二步：编译 BPU 模型

进入地平线工具链 Docker 环境（或确保已安装 `hb_compile`/`hb_mapper` 工具），运行上一步生成的编译脚本：

```bash
cd bpu_export_output
bash build_all.sh
```

编译完成后，`bpu_export_output` 目录下会生成一个 **`bpu_output`** 文件夹。这个文件夹包含了最终部署所需的所有文件（`.hbm` 模型文件和 `.npy` 归一化参数）。

**请将 `bpu_output` 文件夹拷贝到 RDK 板端。**

## 3. 板端推理 (在 RDK 上执行)

1.  确保 `bpu_output` 文件夹已传输到 RDK 板端。
2.  连接机器人（默认配置为 `so101`，如需更改请修改脚本）。
3.  运行控制脚本：

    ```bash
    # 假设 bpu_output 在当前目录下
    python bpu_control_robot.py --bpu-act-path ./bpu_output
    ```

### 常见参数

*   `--bpu-act-path`: BPU 模型文件夹路径 (包含 `.hbm` 和 `.npy` 文件)。
*   `--fps`: 控制循环频率 (默认 30Hz)。
*   `--inference-time`: 自动运行的持续时间 (秒)。

## 4. DAMO 平台模型适配

如果您使用的是 **DAMO 开发者矩阵-乐云具身智能开发平台** 训练的模型，在进行 BPU 模型导出前，**必须** 对数据集进行格式适配。

这是因为 DAMO 平台产生的数据集字段名称（如 `action`, `observation.state`）与 LeRobot 标准格式不完全一致（需要转换为 `action.joint`, `observation.state.joint`），直接使用会导致量化失败。

**操作步骤：**

1.  **备份数据**：该操作会直接修改源文件，请务必先备份您的数据集文件夹。
2.  **编辑脚本**：
    打开 `damo/replace.py` 文件，修改底部的 `folder_path` 变量为您下载的数据集路径：

    ```python
    if __name__ == "__main__":
        # --------------------------
        # 配置区域：修改这里的路径
        # --------------------------
        folder_path = "/path/to/your/damo/dataset"  # <--- 修改这里
        
        process_folder(folder_path)
    ```

3.  **运行转换**：

    ```bash
    python damo/replace.py
    ```

    脚本会自动递归扫描目录下的 `.parquet`, `.json`, `.jsonl` 文件并完成关键字替换。

## 注意事项

*   **数据兼容性**: 如果使用 DAMO 平台训练的数据，请确保数据格式已被正确适配（如有必要使用 `damo/` 目录下的工具）。
*   **机器人配置**: `bpu_control_robot.py` 默认连接 `so101` 机器人。如果您使用的是其他类型的机器人，请在运行前修改代码中的 `make_robot("so101")` 为您的机器人型号。
