English| [简体中文](./README_CN.md)

# RDK LeRobot Tools

**This `s100` branch is intended for deploying LeRobot v0.5.2 ACT policies on RDK S100 BPU.**

**Note: For S100, use `nash-e` and the OE 3.7.0 S100/S600 toolchain.**

**Pick and Place Demo:**

<div align="center">
  <img src="./doc/assets/demo_pick_place.gif" width="480" alt="Pick and Place Demo" />
</div>

> Note that this demo is just a simple Pick and Place demonstration with only 33 episodes of training data collected. Below is a side-by-side visualization of 6 training episodes (Episode 0/6/13/20/26/32):

<div align="center">
  <img src="./doc/assets/demo_episodes_grid.gif" width="640" alt="Training data visualization - 6 episodes side by side" />
</div>

This repository provides a set of tools to export ACT policy models trained with [Hugging Face LeRobot](https://github.com/huggingface/lerobot) and deploy them to D-Robotics RDK S100, utilizing the BPU for efficient inference.

For the full workflow documentation, see: 👉 *[Full Workflow Guide](./doc/WORKFLOW_GUIDE_EN.md)*

## Directory Structure

- `export_bpu_actpolicy.py`: **Model Export Script** (runs on the development machine/training server). Used to convert PyTorch weights to ONNX and generate configuration files and scripts required for BPU compilation.
- `bpu_export_config.yaml`: **Generic config template** (defaults to `nash-e` for S100). Copy and edit for your dataset/checkpoint.
- `bpu_control_robot.py`: **On-Board Deployment Script** (runs on the RDK board). Loads the compiled BPU model and controls the robot.
- `bpu_runtime/`: **C++ BPU inference extension** (pybind11). Replaces the `hbm-runtime` Python package, enabling deployment under Python 3.12 + LeRobot v0.5.2 without the Python 3.10 ABI constraint of the system `hbm_runtime.so`.

## 1. Environment Preparation

### 1.1 Development Machine (For Model Conversion)

Use the official Hugging Face LeRobot repository:
👉 **[https://github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)**

Do **not** use the outdated `D-Robotics/lerobot` fork.

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
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git clone https://github.com/D-Robotics/rdk_LeRobot_tools.git
cd rdk_LeRobot_tools && git checkout s100 && cd ..
pip install -e ".[feetech]"
pip install onnx onnxsim termcolor tqdm safetensors
```

*Note: Model compilation (ONNX -> HBM) needs to be performed in the Docker toolchain environment (OpenExplorer) provided by D-Robotics.*

### 1.2 RDK S100 Board (For Model Deployment)

The on-board deployment uses **Python 3.12** + **LeRobot v0.5.2** + a **C++ pybind11 BPU extension** (bundled in this repo under `bpu_runtime/`).

This replaces the `hbm-runtime` PyPI package, whose pre-built `.so` is compiled for Python 3.10 and cannot be imported under Python 3.12. The C++ extension links directly against the system BPU libraries (`libdnn.so`, `libhbucp.so`) and is compiled for whatever Python version your venv uses.

#### Step 1: Install uv and create Python 3.12 venv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd ~
git clone https://github.com/huggingface/lerobot.git
cd lerobot
uv venv --python 3.12 .venv
source .venv/bin/activate
```

#### Step 2: Install LeRobot and tools

```bash
git clone https://github.com/D-Robotics/rdk_LeRobot_tools.git
cd rdk_LeRobot_tools && git checkout s100 && cd ..
uv pip install -e ".[feetech]"
uv pip install onnx onnxsim termcolor tqdm safetensors numpy
```

#### Step 3: Build the C++ BPU runtime extension

The `bpu_runtime/` directory contains a pybind11 module that wraps the BPU C/C++ inference API (`hbDNN` / `hbUCP`). It exposes a `BPUACTRuntime` class that is a drop-in replacement for `hbm_runtime.HB_HBMRuntime`.

```bash
cd rdk_LeRobot_tools/bpu_runtime
uv pip install pybind11
mkdir build && cd build
cmake -DPython3_EXECUTABLE=$(which python) \
      -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())") \
      ..
make -j$(nproc)
```

After a successful build, you will have `bpu_act_runtime.cpython-312-aarch64-linux-gnu.so` in `bpu_runtime/build/`.

The `bpu_control_robot.py` script will automatically find and import this `.so` — no manual `PYTHONPATH` needed. It tries `hbm_runtime` first (the original PyPI package), and falls back to the C++ extension if `hbm_runtime` is unavailable.

#### Step 4: Verify the build

```bash
cd rdk_LeRobot_tools
python -c "
from bpu_act_runtime import BPUACTRuntime
import numpy as np
rt = BPUACTRuntime(['bpu_output/BPU_ACTPolicy_VisionEncoder.hbm',
                     'bpu_output/BPU_ACTPolicy_TransformerLayers.hbm'])
out = rt.run({'images': np.zeros((1,3,480,640),dtype=np.float32)}, model_name='VisionEncoder')
print('Vision output:', {k: v.shape for k,v in out.items()})
"
```

## 2. Model Export and Compilation (Executed on Development Machine)

This process has two stages: `export_bpu_actpolicy.py` exports ONNX and compile configs from a PyTorch checkpoint, then the OE toolchain compiles those ONNX files into `.hbm`.

### Export Pipeline Overview

There are **two different config files** in this workflow:


| Config File                  | Used By                   | Purpose                                                     |
| ---------------------------- | ------------------------- | ----------------------------------------------------------- |
| `bpu_export_config.yaml`     | `export_bpu_actpolicy.py` | checkpoint, dataset, export path, `nash-e`, `cal_num`, etc. |
| `config_BPU_ACTPolicy_*.yaml` | OE `hb_compile`           | ONNX path, calibration data, quantization/compile settings  |


In short: **you edit the export-stage YAML; the toolchain consumes the auto-generated `config_*.yaml` files.**

The OE toolchain only accepts ONNX. It does not read PyTorch checkpoints directly. The standard flow is:

```text
bpu_export_config.yaml
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
├── bpu_output/                  # runtime normalize params + final .hbm
└── build_all.sh
        ↓
OE Docker: hb_compile
        ↓
bpu_output/*.hbm
```

#### Why Split ACT into Two ONNX Models?

The full ACT path is:

```text
image -> VisionEncoder -> front_features
state + front_features -> Transformer -> Actions [1, 100, 6]
```

For BPU deployment, the export script splits ACT into two submodels:

1. **VisionEncoder**: `images` -> `Vision_Features`
2. **TransformerLayers**: `states` + `front_features` -> `Actions`

`bpu_control_robot.py` chains them together on the board.

#### What Does `export_bpu_actpolicy.py` Do?

When you run `python export_bpu_actpolicy.py --config bpu_export_config.yaml`, the script executes the following 6 steps:

**① Load model and dataset, auto-detect cameras**

Loads the PyTorch ACT checkpoint from `act_path` and reads the dataset from `dataset.root`. Fetches one batch, scans all `observation.images.`* fields, and auto-detects camera names (e.g., `front`).

**② Export pre/post-processing normalization parameters**

Reads statistics saved during training from processor safetensors files in the checkpoint, saves to `bpu_output/*.npy`:

- `{camera_name}_mean.npy` / `{camera_name}_std.npy`: Image normalization mean and std.
- `action_mean.npy` / `action_std.npy`: State input normalization parameters.
- `action_mean_unnormalize.npy` / `action_std_unnormalize.npy`: Action output denormalization parameters.

These `.npy` files are loaded by `bpu_control_robot.py` at board-side inference time for manual normalization/denormalization outside the BPU.

**③ Export VisionEncoder ONNX**

Wraps the ACT `backbone` (ResNet) and `encoder_img_feat_input_proj` as `BPU_ACTPolicy_VisionEncoder`. Input: normalized image. Output: visual feature map `[1, 512, 15, 20]`. If `onnx_sim: true`, onnxsim simplifies the graph.

**④ Export TransformerLayers ONNX**

Wraps encoder + decoder + action_head as `BPU_ACTPolicy_TransformerLayers`. Inputs: `states` and VisionEncoder output `{camera_name}_features`. Output: `Actions [1, 100, 6]`. Also saves `new_actions.npy` for precision verification.

**⑤ Generate OE compile configuration and build scripts**

Generates `config_BPU_ACTPolicy_*.yaml`, `build_*.sh`, and the one-click `build_all.sh` for both submodels. Configs include `march: nash-e`, `norm_type: no_preprocess`, etc.

**⑥ Generate quantization calibration data**

Iterates over the training dataset (up to `cal_num` samples):

- Image `0..255 → /255.0 → (image - mean) / std` → VisionEncoder calibration data.
- Image through VisionEncoder forward pass → visual features → Transformer `{camera_name}/` calibration data.
- Normalized state → Transformer `state/` calibration data.

Image calibration tensors are converted from `0..255` to `0..1` when needed, then normalized with `(image - mean) / std`, matching board runtime `uint8 -> /255.0 -> normalize`.

#### `config_BPU_ACTPolicy_VisionEncoder.yaml`

```yaml
model_parameters:
  onnx_model: BPU_ACTPolicy_VisionEncoder.onnx
  march: nash-e
calibration_parameters:
  cal_data_dir: calibration_data_BPU_ACTPolicy_VisionEncoder
  cal_data_type: float32
input_parameters:
  norm_type: no_preprocess
compiler_parameters:
  jobs: 6
```

`cal_data_dir` stores **already normalized image tensors** (for example `front_0000000000.npy`) used to estimate quantization ranges.

#### `config_BPU_ACTPolicy_TransformerLayers.yaml`

```yaml
input_parameters:
  input_name: states;front_features;
  norm_type: no_preprocess;no_preprocess;
calibration_parameters:
  cal_data_dir: calibration_data_.../state;calibration_data_.../front;
```

Transformer has two inputs:


| Input            | Calibration Content                     |
| ---------------- | --------------------------------------- |
| `states`         | normalized joint state `[1, 6]`         |
| `front_features` | VisionEncoder output `[1, 512, 15, 20]` |


Transformer calibration is **not** based on raw images. It uses Vision features.

#### Board Runtime Data Flow

```text
camera uint8
  -> /255.0
  -> (image - front_mean) / front_std
  -> VisionEncoder.hbm
  -> front_features
  -> (state - action_mean) / action_std
  -> TransformerLayers.hbm
  -> Actions [100, 6]
  -> unnormalize
  -> send to robot
```

Because normalization happens outside the BPU, both `config_*.yaml` files use `norm_type: no_preprocess`.

### Step 1: Export ONNX and Configuration

1. **Modify Configuration File**:
  For S100 / SO100 ACT, edit `bpu_export_config.yaml`:
  - `dataset.root`: Dataset root used during training.
  - `act_path`: Trained ACT checkpoint path.
  - `export_path`: Export output directory.
  - `type`: Must be `nash-e` for S100.
  - `cal_num`: Recommended `100`.
2. **Run Export Script**:
  ```bash
    cd rdk_LeRobot_tools
    python export_bpu_actpolicy.py --config bpu_export_config.yaml
  ```
    After successful execution, ONNX files, calibration data, and `build_all.sh` are generated under `export_path`.
    **Important Note:** The export script reads image, state, and action normalization statistics from the LeRobot v0.5.2 checkpoint processor safetensors. Image calibration tensors are first ensured to be `0..1` float inputs, then normalized with `(image - mean) / std`, matching the board runtime path `uint8 -> /255.0 -> normalize`.

### Step 2: Compile BPU Model

Enter the toolchain Docker environment and run the compilation script generated in the previous step:

```bash
cd /path/to/bpu_export_act_so100_s100_calfix
bash build_all.sh
```

For S100, use the OE 3.7.0 S100/S600 Docker toolchain. Toolchain release summary (continuously updated): [https://forum.d-robotics.cc/t/topic/35229](https://forum.d-robotics.cc/t/topic/35229)

```bash
# Download offline image package
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe/3.7.0/ai_toolchain_ubuntu_22_s100_s600_cpu_v3.7.0.tar
# Load image
sudo docker load -i ai_toolchain_ubuntu_22_s100_s600_cpu_v3.7.0.tar
```

Run compilation:

```bash
docker run --rm \
  -v /path/to/bpu_export_act_so100_s100_calfix:/workspace \
  -w /workspace \
  registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0 \
  bash build_all.sh
```

After compilation is complete, a `**bpu_output**` folder is generated under the export directory. This folder contains:

- `.hbm` / `.bin`: The compiled BPU model files (executable on the BPU).
- `.npy`: Normalization parameters required for runtime.
- `new_actions.npy`: Model inference results before conversion (used for precision verification).

**Please copy the `bpu_output` folder to the RDK board.**

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

## 3. On-Board Inference (Executed on RDK S100)

The core of on-board inference is using the `**bpu_control_robot.py**` script.

### Prerequisites

1. Installed [huggingface/lerobot](https://github.com/huggingface/lerobot) v0.5.2 in a Python 3.12 venv.
2. Built the C++ BPU runtime extension (`bpu_runtime/build/bpu_act_runtime.*.so`).
3. Transferred the `**bpu_output**` folder (containing the quantized `.hbm` model and calibration parameters) to the board.
4. **Hardware Configuration**: Pass robot port, camera index, and camera name via CLI. Calibration files are saved by `lerobot-calibrate`.

### Run Steps

1. Connect the robot. The current script uses **SO100Follower** by default.
2. Run the control script:
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

- `--bpu-act-path`: BPU model folder path (must contain `.hbm` and `.npy` files).
- `--robot-port`: Follower serial port, default `/dev/ttyACM0`.
- `--camera-index` / `--camera-name`: Camera index and name, must match `bpu_output/*_mean.npy`.
- `--fps`: Control loop frequency (default 30Hz).
- `--inference-time`: Duration of automatic operation (seconds).

### BPU Inference Performance Benchmark

Pure BPU performance benchmark on RDK S100 for each ACT module (20 warmup + 200 official samples):


| Module            | Avg. Inference Time | Frame Rate      |
| ----------------- | ------------------- | --------------- |
| VisionEncoder     | 4.14 ms             | 241.7 inf/s     |
| TransformerLayers | 3.30 ms             | 302.8 inf/s     |
| **Complete ACT**  | **7.54 ms**         | **132.6 inf/s** |


ACT outputs a 100-step action chunk in one inference. At 30 fps control frequency, only one BPU inference (7.54 ms) is needed every 3.33 seconds, leaving the BPU idle for the rest of the time.

## 4. C++ BPU Runtime (`bpu_runtime/`)

### Why a C++ Extension?

The `hbm-runtime` PyPI package (`hbm_runtime.HB_HBMRuntime`) is compiled as a pybind11/C extension linked against Python 3.10. LeRobot v0.5.2 requires Python >= 3.12 (`requires-python = ">=3.12"` in `pyproject.toml`). This makes it impossible to `import hbm_runtime` under Python 3.12.

The `bpu_runtime/` directory contains a self-contained C++ pybind11 extension that wraps the same BPU C API (`hbDNN` / `hbUCP`) and exposes a `BPUACTRuntime` class with an identical interface. It links against the system BPU libraries (`libdnn.so`, `libhbucp.so`) which are Python-version-independent, so it compiles cleanly for any Python version.

### How It Works

`bpu_control_robot.py` tries `hbm_runtime` first. If unavailable, it imports the C++ extension and wraps it in a thin adapter class that matches the `HB_HBMRuntime` interface:

```python
try:
    from hbm_runtime import HB_HBMRuntime
except ImportError:
    from bpu_act_runtime import BPUACTRuntime as _BPUACTRuntime

    class HB_HBMRuntime:
        def __init__(self, model_paths):
            self._rt = _BPUACTRuntime(model_paths)
        def run(self, inputs, model_name=""):
            output = self._rt.run(inputs, model_name=model_name)
            return {model_name: output}
```


### Build

```bash
cd bpu_runtime
uv pip install pybind11    # or: pip install pybind11
mkdir build && cd build
cmake -DPython3_EXECUTABLE=$(which python) \
      -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())") \
      ..
make -j$(nproc)
```

The output `.so` (e.g., `bpu_act_runtime.cpython-312-aarch64-linux-gnu.so`) is placed in `bpu_runtime/build/` and is automatically discovered by `bpu_control_robot.py`.

### Files

```
bpu_runtime/
├── CMakeLists.txt              # CMake build config
├── inc/
│   └── bpu_pybind.hpp          # BPU inference wrapper (BPUSubModel + BPUACTRuntime)
├── src/
│   └── bpu_act_runtime.cc      # pybind11 module entry point
└── build/                      # compiled .so (generated by cmake/make)
    └── bpu_act_runtime.cpython-312-aarch64-linux-gnu.so
```

## 5. Dataset Format Note (v3.0 vs v2.1)

This branch uses LeRobot v0.5.2 with `**codebase_version: v3.0`** dataset format. The `stable` branch is compatible with **v2.1** format. Key differences:


| Aspect                | v2.1 (stable branch)                              | v3.0 (this s100 branch)                                   |
| --------------------- | ------------------------------------------------- | --------------------------------------------------------- |
| **Organization**      | One file per episode                              | Multiple episodes packed into larger files                |
| **Data files**        | `data/chunk-000/episode_000000.parquet`           | `data/chunk-000/file-000.parquet`                         |
| **Video files**       | `videos/chunk-000/camera/episode_000000.mp4`      | `videos/camera/chunk-000/file-000.mp4`                    |
| **Metadata**          | JSON Lines (`meta/episodes.jsonl`, `tasks.jsonl`) | Parquet (`meta/episodes/*.parquet`, `meta/tasks.parquet`) |
| **Episode stats**     | Separate `episodes_stats.jsonl`                   | Merged into `meta/episodes/*.parquet`                     |
| `**next.done` field** | Present                                           | Removed; episode boundaries inferred from metadata        |


Core data fields (`observation.state`, `action`, `timestamp`, `episode_index`, etc.) are semantically unchanged — only the storage layout differs. v3.0 consolidates many episodes into fewer, larger files and uses offset-based indexing in `meta/episodes/`, which is designed for datasets with millions of episodes and provides faster initialization and less file-system pressure.

If your training data is in v2.1 format, convert it first using the official script:

```bash
python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id=<your-repo-id> --root=<dataset-path>
```

## Notes

- **Model Compatibility**: On-board execution must use `.hbm` / `.bin` models quantized and compiled by the OE toolchain, and cannot directly run ONNX or PyTorch models.
- **Robot Configuration**: `bpu_control_robot.py` is currently hardcoded to `SO100Follower`. For SO-101 deployment, update the robot type in code first.
- **BPU march**: This branch targets S100 (`nash-e`). For S600 deployment, use `nash-p` and the corresponding toolchain image.
