[English](./README.md) | 简体中文

# RDK S600 上的 Pi0 Policy

本文说明 Pi0 的后训练、链路感知量化、Standalone HBM 部署、离线验证，以及 SO100 双相机严格同步推理。

Pi0 的实现统一放在 `models/pi0/` 下。当前已经在真实 SO100 follower 和两路 USB 相机上验证以下严格同步链路：

```text
front + side + 6 维关节状态 + 固定任务 Prompt
  ↓
SigLIP HBM × 3 个物理槽位
  ├─ front：真实图像
  ├─ side：真实图像
  └─ 第三槽：masked empty image
  ↓
PaliGemma HBM → 36 个 KV Tensor
  ↓
Action Expert HBM × 10 次 Flow Matching
  ↓
[50, 6] SO100 绝对关节目标
  ↓
30 Hz SO100Follower.send_action()
```

正式启动器固定使用 `prefetch_steps=0`：每个动作块都读取最新 state、front 和 side，完整推理并执行 50 步之后，才采集下一组观测。PaliGemma KV 只在同一次请求的 10 次 Expert 去噪中复用，不跨图像或动作块复用。

## 已验证基线

- LeRobot：v0.5.2。
- 机械臂：SO100 follower，6 维 degree 动作。
- 相机：front + side 两路真实相机，外加一个被 mask 的 Pi0 空槽。
- 训练：100 个 episode、15,000 steps、BF16、batch size 1；SigLIP、PaliGemma、Expert 全部参与训练。
- Runtime：D-Robotics LLM S600 SDK 1.0.2，`nash-p`，运行时不依赖 `libxlm.so`。
- 连续真机验证：严格同步运行 64 个 chunk，每个 chunk 都更新 front + side。
- 同一固定真实输入的 BF16/HBM 对比：MAE `0.4405°`、RMSE `0.5903°`、最大误差 `1.6888°`、relative L2 `0.852%`、cosine `0.999981858`。

这组数值只用于确认固定输入下整条链路没有明显漂移，不等于任务成功率报告。

## 目录结构

```text
models/pi0/
├─ pi0_full_pipeline.py         # 相机、SO100 生命周期、TCP server、30 Hz 控制
├─ pi0_standalone_offline.py    # 不访问机械臂串口的离线整链 smoke test
├─ pi0_torch_server.py          # 服务器 BF16 参考实现
├─ run_live_sync.sh             # 已验证的严格同步真机启动器
├─ validate_pi0_config.py       # deployment artifact/hash 校验
├─ configs/                     # deployment 配置、stats、量化 manifest
├─ native/                      # standalone C++ runtime 与 SDK 兼容层
└─ tools/                       # 训练、校准与三段量化
```

仓库不会提交 HBM、数据集、真实校准图片/KV、token、机械臂标定、诊断 dump 或编译产物。

## 板端目录约定

```text
/root/rdk_LeRobot_tools
/home/sunrise/lerobot
/root/D-Robotics_LLM_S600_1.0.2_SDK/oellm_runtime
/root/pi0_models/versions
```

最终 deployment 配置：

```text
models/pi0/configs/deployments/pi0_full_v5_2cam_positionfp16_siglip_fixed16_paligemma_hbmkv_expert_20260801.json
```

对应 manifest 保存模型来源、文件大小和 SHA256。不要只替换其中一份 HBM；更换上游模型后必须重新生成或重新验证后续链路。

## 编译与无动作验证

编译 standalone runtime：

```bash
cd /root/rdk_LeRobot_tools
git checkout s600
bash models/pi0/native/build_standalone_pi0.sh
```

进行静态检查并校验最终 deployment bundle：

```bash
cd /root/rdk_LeRobot_tools/models/pi0
bash -n run_live_sync.sh run_pi0_standalone_config.sh native/build_standalone_pi0.sh
/home/sunrise/lerobot/.venv/bin/python -m py_compile \
  pi0_full_pipeline.py pi0_standalone_offline.py pi0_torch_server.py validate_pi0_config.py
/home/sunrise/lerobot/.venv/bin/python validate_pi0_config.py \
  configs/deployments/pi0_full_v5_2cam_positionfp16_siglip_fixed16_paligemma_hbmkv_expert_20260801.json
```

离线双图 smoke test 不会打开机械臂串口：

```bash
/home/sunrise/lerobot/.venv/bin/python -u pi0_standalone_offline.py \
  --front /path/to/front.jpg \
  --side /path/to/side.jpg \
  --state 0 0 0 0 0 0 \
  --output-dir diagnostics/offline_smoke_$(date +%Y%m%d_%H%M%S)
```

预期输出 shape 为 `[50, 6]`。

## 连续真机推理

启动前确认 `/dev/ttyACM0` 是 follower、`/dev/video0` 与 `/dev/video2` 是目标相机，并清空机械臂工作区：

```bash
cd /root/rdk_LeRobot_tools/models/pi0
./run_live_sync.sh
```

按 `Ctrl+C` 停止。Python controller 负责管理进程组，并在正常退出时关闭力矩。启动器显式使用 `--force-model-actions`，不会增加 1°/20° 相对目标限制或 chunk blend，因此必须有人现场监护。

## 训练与链路感知量化入口

| 阶段 | 入口 | 关键约束 |
| --- | --- | --- |
| 全参数 Post-training | `models/pi0/tools/train_pi0_full_v5_2cam_100ep.sh` | 增加第二路真实相机后，三段模型都需要参与训练。 |
| 真实校准集 | `models/pi0/tools/prepare_pi0_calibration_v5_2cam.py` | 使用覆盖任务过程的 front + side 样本和匹配的 checkpoint stats。 |
| SigLIP | `models/pi0/tools/quantize_siglip_real_calib.py` | position embedding 保留高精度。 |
| PaliGemma | `models/pi0/tools/quantize_paligemma_real_calib.py` | 使用真实 SigLIP HBM 输出校准。 |
| Expert | `models/pi0/tools/quantize_expert_real_calib.py` | 使用最终 PaliGemma HBM 的真实 36 组 KV 校准。 |

正确顺序是 `SigLIP HBM → 导出真实视觉特征 → PaliGemma HBM → 导出真实 KV → Expert HBM`。如果三个模型都使用浮点中间量独立校准，即使每份 HBM 都能编译，整条闭环仍可能出现明显误差。
