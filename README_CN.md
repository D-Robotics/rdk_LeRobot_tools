[English](./README.md) | 简体中文

# RDK LeRobot Tools

RDK LeRobot Tools 用于把基于 [Hugging Face LeRobot](https://github.com/huggingface/lerobot) 训练的策略部署到地瓜机器人 RDK 设备。仓库集中维护从训练模型到真实硬件所需的模型导出、量化、编译、板端推理和机械臂控制代码。

当前 `s600` 分支维护两套面向 RDK S600 的 Policy：

| Policy | 仓库提供的能力 | 使用文档 |
| --- | --- | --- |
| ACT | ONNX 导出、BPU 编译、量化校准和 SO100 板端推理 | [ACT 使用说明](./models/act/README_CN.md) |
| Pi0 | 后训练工具、三段链路感知量化、Standalone HBM Runtime 和 SO100 双相机严格同步推理 | [Pi0 使用说明](./models/pi0/README_CN.md) |

## 仓库结构

```text
models/
├─ act/       # ACT 实现与使用文档
└─ pi0/       # Pi0 实现与使用文档
doc/          # 公共流程文档和素材
```

根目录的 `bpu_control_robot.py` 和 `export_bpu_actpolicy.py` 仅作为旧 ACT 命令的兼容入口保留。新增的模型代码和详细文档应放在对应的 `models/<policy>/` 目录中。

仓库不会提交训练 checkpoint、HBM/ONNX 模型、数据集、量化校准样本、机械臂标定、诊断 dump、token 或编译产物。各模型目录中的使用文档会说明对应的本地运行资产和目录要求。
