#!/usr/bin/env python

# Copyright (c) 2025，WuChao&&MaChao D-Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 注意: 此程序在开发机的训练环境中运行
# Attention: This program runs on developer machine training environment.

import logging
import os
import shutil
import sys
import numpy as np
import torch
import torch.nn as nn
import argparse
import onnx
import yaml
from copy import deepcopy
from termcolor import colored
from onnxsim import simplify
from pprint import pformat
from tqdm import tqdm # Import tqdm

try:
    from lerobot.common.policies.act.modeling_act import ACTPolicy
    from lerobot.common.datasets.factory import make_dataset
    from lerobot.common.utils.utils import get_safe_torch_device, init_logging
except ImportError:
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.datasets.factory import make_dataset
    from lerobot.utils.utils import get_safe_torch_device, init_logging

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig

BPU_VisionEncoder = "BPU_ACTPolicy_VisionEncoder"
BPU_TransformerLayers = "BPU_ACTPolicy_TransformerLayers"

REPOSITORY = "REPOSITORY"
TAG = "TAG"

# 定义一个辅助类，用于强制 yaml 转储时使用流式风格 (即 {...})
class FlowStyleDict(dict): pass

def flow_dict_representer(dumper, data):
    return dumper.represent_mapping('tag:yaml.org,2002:map', data, flow_style=True)

# 注册自定义代表器，确保使用 FlowStyleDict 的字典会显示为 {key: value}
yaml.add_representer(FlowStyleDict, flow_dict_representer)

def load_config_and_inject_args():
    """从配置文件加载参数并注入到sys.argv中，同时返回BPU参数"""
    # 先检查是否有--config参数
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument('--config', type=str, help='Path to BPU export config YAML file')
    temp_args, _ = temp_parser.parse_known_args()

    if temp_args.config:
        logging.info(f"Loading config from: {temp_args.config}")
        with open(temp_args.config, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        # 将配置文件中的LeRobot参数转换为命令行参数格式
        injected_args = []

        # 处理dataset参数
        if 'dataset' in config_dict:
            dataset_cfg = config_dict['dataset']
            if 'repo_id' in dataset_cfg:
                injected_args.extend(['--dataset.repo_id', str(dataset_cfg['repo_id'])])
            if 'root' in dataset_cfg:
                injected_args.extend(['--dataset.root', str(dataset_cfg['root'])])

        # 处理policy参数
        if 'policy' in config_dict:
            policy_cfg = config_dict['policy']
            if 'type' in policy_cfg:
                injected_args.extend(['--policy.type', str(policy_cfg['type'])])
            if 'device' in policy_cfg:
                injected_args.extend(['--policy.device', str(policy_cfg['device'])])
            if 'repo_id' in policy_cfg:
                injected_args.extend(['--policy.repo_id', str(policy_cfg['repo_id'])])

        # 处理wandb参数
        if 'wandb' in config_dict:
            wandb_cfg = config_dict['wandb']
            if 'enable' in wandb_cfg:
                injected_args.extend(['--wandb.enable', str(wandb_cfg['enable']).lower()])

        # 将注入的参数添加到sys.argv中（但不重复已存在的参数）
        existing_args = set()
        for arg in sys.argv[1:]:
            if arg.startswith('--'):
                existing_args.add(arg.split('=')[0])

        for i in range(0, len(injected_args), 2):
            if injected_args[i] not in existing_args:
                sys.argv.extend([injected_args[i], injected_args[i+1]])

        # 移除所有BPU相关参数（包括--config），避免draccus解析时报错
        bpu_params = ['config', 'act-path', 'export-path', 'cal-num', 'onnx-sim', 'type', 'combine-jobs']

        # 过滤掉BPU参数及其值
        filtered_argv = [sys.argv[0]]  # 保留脚本名称
        i = 1
        while i < len(sys.argv):
            arg = sys.argv[i]
            # 检查是否是BPU参数
            is_bpu_param = False
            for param in bpu_params:
                if arg == f'--{param}' or arg.startswith(f'--{param}='):
                    is_bpu_param = True
                    # 如果是 --param value 格式，跳过下一个参数（值）
                    if '=' not in arg and i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
                        i += 1  # 跳过值
                    break

            if not is_bpu_param:
                filtered_argv.append(arg)

            i += 1

        sys.argv = filtered_argv

        logging.info(f"Cleaned sys.argv for draccus: {sys.argv}")
        return config_dict

    return None

# 全局变量存储配置文件内容
_global_config = None

class PlatformConfig:
    """Helper class to manage platform-specific configurations (Nash vs Bayes)"""
    def __init__(self, platform_type):
        self.type = platform_type
        self.is_bayes = "bayes" in platform_type
        
        if self.is_bayes:
            self.opset_version = 11
            self.optimize_level = 'O3'
            self.optimization_params = "set_all_nodes_int16;set_Softmax_input_int16;set_Softmax_output_int16;"
            self.compile_cmd_prefix = "hb_mapper makertbin --model-type onnx --config"
            self.model_ext = ".bin"
            self.cal_data_ext = ".nchw"
            self.bin_copy_cmd = "cp bpu_model_output/{model_prefix}.bin ../{bpu_output_name}"
        else: # Default to nash
            self.opset_version = 19
            self.optimize_level = 'O2'
            self.optimization_params = "set_all_nodes_int16"
            self.compile_cmd_prefix = "hb_compile --config"
            self.model_ext = ".hbm"
            self.cal_data_ext = ".npy"
            self.bin_copy_cmd = "cp bpu_model_output/{model_prefix}.hbm ../{bpu_output_name}"

    def save_calibration(self, path, tensor_data):
        data_np = tensor_data.detach().cpu().numpy()
        # Add extension if not present in name pattern (nash code added .npy in loop, bayes added .nchw)
        # But wait, logic in main loop constructs full path with name. 
        # Nash name: "%.10d.npy"%i
        # Bayes name: "%.10d.nchw"%i
        if self.is_bayes:
            data_np.tofile(path)
        else:
            np.save(path, data_np)

    def get_cal_data_name(self, index):
        if self.is_bayes:
            return "%.10d.nchw" % index
        else:
            return "%.10d.npy" % index

@parser.wrap()
def main(cfg: TrainPipelineConfig):
    logging.info(pformat(cfg.to_dict()))

    # BPU导出参数 - 默认值
    class BPUOptions:
        act_path = 'outputs/train/act_so100_test/checkpoints/001000/pretrained_model'
        export_path = 'bpu_export_output'
        cal_num = 400
        onnx_sim = True
        type = "nash-e"
        combine_jobs = 6

    opt = BPUOptions()

    # 如果有全局配置文件，从配置文件加载BPU参数
    global _global_config
    if _global_config:
        opt.act_path = _global_config.get('act_path', opt.act_path)
        opt.export_path = _global_config.get('export_path', opt.export_path)
        opt.cal_num = _global_config.get('cal_num', opt.cal_num)
        opt.onnx_sim = _global_config.get('onnx_sim', opt.onnx_sim)
        opt.type = _global_config.get('type', opt.type)
        opt.combine_jobs = _global_config.get('combine_jobs', opt.combine_jobs)
        logging.info("BPU parameters loaded from config file")

    # 命令行参数覆盖
    bpu_parser = argparse.ArgumentParser()
    bpu_parser.add_argument('--config', type=str, help='Path to BPU export config YAML file')
    bpu_parser.add_argument('--act-path', type=str, help='Path to LeRobot ACT Policy model.')
    bpu_parser.add_argument('--export-path', type=str, help='Path to save LeRobot ACT Policy model.')
    bpu_parser.add_argument('--cal-num', type=int, help='Num of images to generate')
    bpu_parser.add_argument('--onnx-sim', type=bool, help='Simplify onnx or not.')
    bpu_parser.add_argument('--type', type=str, help='Optional: nash-e, nash-m, nash-p, bayes-e, bayes')
    bpu_parser.add_argument('--combine-jobs', type=int, help='combine jobs for OpenExplore.')

    cli_opt, _ = bpu_parser.parse_known_args()

    if cli_opt.act_path: opt.act_path = cli_opt.act_path
    if cli_opt.export_path: opt.export_path = cli_opt.export_path
    if cli_opt.cal_num is not None: opt.cal_num = cli_opt.cal_num
    if cli_opt.onnx_sim is not None: opt.onnx_sim = cli_opt.onnx_sim
    if cli_opt.type: opt.type = cli_opt.type
    if cli_opt.combine_jobs is not None: opt.combine_jobs = cli_opt.combine_jobs

    if "bernoulli2" in opt.type:
        print("Need to do. But Bernoulli2 is very similiar with Bayes.")
        exit()

    # 初始化平台配置
    platform = PlatformConfig(opt.type)

    logging.info("="*80)
    logging.info(colored("BPU Export Configuration:", 'light_cyan'))
    logging.info(f"  ACT Model Path:      {opt.act_path}")
    logging.info(f"  Export Path:         {opt.export_path}")
    logging.info(f"  Calibration Samples: {opt.cal_num}")
    logging.info(f"  ONNX Simplify:       {opt.onnx_sim}")
    logging.info(f"  BPU Type:            {opt.type} (Bayes: {platform.is_bayes})")
    logging.info(f"  Compiler Jobs:       {opt.combine_jobs}")
    logging.info(f"  Dataset Root:        {cfg.dataset.root}")
    logging.info("="*80)

    # 路径设置
    if os.path.exists(opt.export_path): 
        shutil.rmtree(opt.export_path)
    
    visionEncoder_ws = os.path.join(opt.export_path, BPU_VisionEncoder)
    transformersLayers_ws = os.path.join(opt.export_path, BPU_TransformerLayers)
    
    # ONNX路径
    onnx_name_Vision = BPU_VisionEncoder + ".onnx"
    onnx_path_Vision = os.path.join(visionEncoder_ws, onnx_name_Vision)
    onnx_name_Transformer = BPU_TransformerLayers + ".onnx"
    onnx_path_Transformer = os.path.join(transformersLayers_ws, onnx_name_Transformer)
    
    # 校准数据路径
    cal_data_name_Vision = "calibration_data_" + BPU_VisionEncoder
    cal_data_path_Vision = os.path.join(visionEncoder_ws, cal_data_name_Vision)
    cal_data_name_Transformer = "calibration_data_" + BPU_TransformerLayers
    cal_data_path_Transformer = os.path.join(transformersLayers_ws, cal_data_name_Transformer)
    state_cal_data_path_Transformer = os.path.join(cal_data_path_Transformer, "state")
    
    # 配置文件路径
    config_yaml_path_Vision = os.path.join(visionEncoder_ws, f"config_{BPU_VisionEncoder}.yaml")
    config_yaml_path_Transformer = os.path.join(transformersLayers_ws, f"config_{BPU_TransformerLayers}.yaml")
    
    # 脚本路径
    bash_path_Vision = os.path.join(visionEncoder_ws, f"build_{BPU_VisionEncoder}.sh")
    bash_path_Transformer = os.path.join(transformersLayers_ws, f"build_{BPU_TransformerLayers}.sh")
    
    # 最终输出路径
    bpu_output_name = "bpu_output"
    bpu_output_path = os.path.join(opt.export_path, bpu_output_name)
    bash_build_all_path = os.path.join(opt.export_path, "build_all.sh") 

    # 创建目录
    for p in [visionEncoder_ws, transformersLayers_ws, cal_data_path_Vision, 
              cal_data_path_Transformer, state_cal_data_path_Transformer, bpu_output_path]:
        os.makedirs(p, exist_ok=True)
        logging.info(colored(f"mkdir: {p} Success.", 'green'))

    # 加载模型
    policy = ACTPolicy.from_pretrained(opt.act_path).cpu().eval()
    logging.info(colored(f"Load ACT Policy Model: {opt.act_path} Success.", 'light_red'))
    
    # CUDA Configs
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # 数据集
    dataset = make_dataset(cfg)
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=0, batch_size=1, shuffle=True,
        sampler=None, pin_memory=device.type != "cpu", drop_last=False,
    )
    logging.info(colored(f"Load ACT Policy Dataset: \n{dataset} Success.", 'light_red'))

    # 动态获取相机
    batch = next(iter(dataloader))
    image_keys = [key for key in batch.keys() if key.startswith('observation.images.')]
    camera_names = [key.split('.')[-1] for key in image_keys]
    kvs = image_keys + ['observation.state']
    batch = dict(filter(lambda item: item[0] in kvs, batch.items()))
    
    logging.info(f"Detected cameras: {camera_names}")
    
    # 1. 导出前后处理参数 (通用)
    for camera_name in camera_names:
        buffer_name = f"buffer_observation_images_{camera_name}"
        if hasattr(policy.normalize_inputs, buffer_name):
            buffer = getattr(policy.normalize_inputs, buffer_name)
            camera_std = buffer.std.data.detach().cpu().numpy()
            camera_mean = buffer.mean.data.detach().cpu().numpy()
            np.save(os.path.join(bpu_output_path, f"{camera_name}_std.npy"), camera_std)
            np.save(os.path.join(bpu_output_path, f"{camera_name}_mean.npy"), camera_mean)

    action_std = policy.normalize_inputs.buffer_observation_state.std.data.detach().cpu().numpy()
    action_mean = policy.normalize_inputs.buffer_observation_state.mean.data.detach().cpu().numpy()
    action_std_unnormalize = policy.unnormalize_outputs.buffer_action.std.data.detach().cpu().numpy()
    action_mean_unnormalize = policy.unnormalize_outputs.buffer_action.mean.data.detach().cpu().numpy()

    np.save(os.path.join(bpu_output_path, "action_std.npy"), action_std)
    np.save(os.path.join(bpu_output_path, "action_mean.npy"), action_mean)
    np.save(os.path.join(bpu_output_path, "action_std_unnormalize.npy"), action_std_unnormalize)
    np.save(os.path.join(bpu_output_path, "action_mean_unnormalize.npy"), action_mean_unnormalize)

    # 2. 导出 VisionEncoder ONNX
    batch = policy.normalize_inputs(batch)
    m_VisionEncoder = BPU_ACTPolicy_VisionEncoder(policy)
    m_VisionEncoder.eval()

    # 生成样例输入进行 tracing
    # 注意: 这里只用第一个相机的输入来trace整个VisionEncoder类，因为它们共享相同的结构
    # 但实际导出时，我们需要确保动态输入的兼容性，这里VisionEncoder只接受单一图像张量
    input_tensor = batch[f'observation.images.{camera_names[0]}']
    
    logging.info(f"Using ONNX opset version: {platform.opset_version} for type: {opt.type}")
    
    torch.onnx.export(
        m_VisionEncoder,
        input_tensor,
        onnx_path_Vision,
        export_params=True,
        opset_version=platform.opset_version,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['Vision_Features'],
        dynamic_axes=None
    )
    onnx_sim(onnx_path_Vision, opt.onnx_sim)
    logging.info(colored(f"Export {onnx_path_Vision} Success.", 'green'))

    # 生成用于Transformer导出的视觉特征 (所有相机)
    vision_features = []
    for camera_name in camera_names:
        img_tensor = batch[f'observation.images.{camera_name}']
        feat = m_VisionEncoder(img_tensor)
        vision_features.append(feat)

    # 3. 导出 TransformerLayers ONNX
    m_TransformerLayers = BPU_ACTPolicy_TransformerLayers(policy, camera_names)
    m_TransformerLayers.eval()

    state = batch["observation.state"]
    input_names = ['states'] + [f'{camera_name}_features' for camera_name in camera_names]
    logging.info(f"Transformer input names: {input_names}")

    # 计算并保存 new_actions.npy 用于调试和验证
    actions = m_TransformerLayers(state, *vision_features)
    np.save(os.path.join(bpu_output_path, "new_actions.npy"), actions.detach().cpu().numpy())
    logging.info(colored(f"Saved new_actions.npy to {bpu_output_path}/new_actions.npy for debug.", 'green'))
    
    torch.onnx.export(
        m_TransformerLayers,
        (state, *vision_features),
        onnx_path_Transformer,
        export_params=True,
        opset_version=platform.opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=['Actions'],
        dynamic_axes=None
    )
    onnx_sim(onnx_path_Transformer, opt.onnx_sim)
    logging.info(colored(f"Export {onnx_path_Transformer} Success.", 'green'))

    # 4. 生成编译配置 YAML
    
    # 通用参数构造
    def generate_yaml_content(model_name, onnx_model, input_name_str, input_type_str, 
                              nchw_str, norm_type_str, cal_data_dir_str, cal_data_type_str, is_vision=True):
        debug_val = True if is_vision else False 
        
        # 使用字典构建配置，结构清晰且易于维护
        config = {
            'model_parameters': {
                'onnx_model': onnx_model,
                'march': opt.type,
                'layer_out_dump': False,
                'working_dir': 'bpu_model_output',
                'output_model_file_prefix': model_name
            },
            'input_parameters': {
                'input_name': input_name_str,
                'input_type_rt': input_type_str,
                'input_layout_rt': nchw_str,
                'input_type_train': input_type_str,
                'input_layout_train': nchw_str,
                'norm_type': norm_type_str
            },
            'calibration_parameters': {
                'cal_data_dir': cal_data_dir_str,
                'cal_data_type': cal_data_type_str,
                'calibration_type': 'default',
                'optimization': platform.optimization_params
            },
            'compiler_parameters': {
                'jobs': opt.combine_jobs,
                'compile_mode': 'latency',
                'debug': debug_val,
                'optimize_level': platform.optimize_level
            }
        }

        # Nash 平台需要 extra_params
        # 为了让 YAML 输出中 extra_params 显示为内联样式 {key: val}，我们先将其作为字符串放入，
        # 然后在生成 YAML 后去除字符串的引号。
        if not platform.is_bayes:
            config['compiler_parameters']['extra_params'] = "{input_no_padding: true, output_no_padding: true}"
        
        # 生成 YAML
        yaml_str = yaml.dump(config, sort_keys=False, default_flow_style=False, width=float("inf"))
        
        # 去除 extra_params 值周围的单引号，使其变回 YAML 字典结构
        return yaml_str.replace("'dict_start", "{").replace("dict_end'", "}") \
                       .replace("'{input_no_padding: true, output_no_padding: true}'", "{input_no_padding: true, output_no_padding: true}")

    # Vision YAML
    yaml_content_vision = generate_yaml_content(
        model_name=BPU_VisionEncoder,
        onnx_model=onnx_name_Vision,
        input_name_str="",
        input_type_str="featuremap",
        nchw_str="NCHW",
        norm_type_str="no_preprocess",
        cal_data_dir_str=cal_data_name_Vision,
        cal_data_type_str="float32",
        is_vision=True
    )
    with open(config_yaml_path_Vision, "w", encoding="utf-8") as f: f.write(yaml_content_vision)
    
    # Transformer YAML (Dynamic Inputs)
    input_name_list = ['states'] + [f'{camera_name}_features' for camera_name in camera_names]
    input_name_str = ';'.join(input_name_list) + ';'
    
    input_type_list = ['featuremap'] * len(input_name_list)
    input_type_str = ';'.join(input_type_list) + ';'
    
    cal_data_dirs = [os.path.join(cal_data_name_Transformer, "state")]
    cal_data_dirs.extend([os.path.join(cal_data_name_Transformer, c) for c in camera_names])
    cal_data_dir_str = ';'.join(cal_data_dirs) + ';'
    
    cal_data_type_str = ';'.join(['float32'] * len(input_name_list)) + ';'
    
    # NCHW and Norm strings are repeated
    nchw_str = ';'.join(['NCHW'] * len(input_name_list)) + ';'
    norm_type_str = ';'.join(['no_preprocess'] * len(input_name_list)) + ';'
    
    yaml_content_transformer = generate_yaml_content(
        model_name=BPU_TransformerLayers,
        onnx_model=onnx_name_Transformer,
        input_name_str=input_name_str,
        input_type_str=input_type_str,
        nchw_str=nchw_str,
        norm_type_str=norm_type_str,
        cal_data_dir_str=cal_data_dir_str,
        cal_data_type_str=cal_data_type_str,
        is_vision=False
    )
    with open(config_yaml_path_Transformer, "w", encoding="utf-8") as f: f.write(yaml_content_transformer)

    logging.info("YAML configs generated.")

    # 5. 生成 Bash 脚本
    def generate_bash(config_name, model_name):
        return f'''
#!/bin/bash
set -e -v
cd $(dirname $0) || exit
{platform.compile_cmd_prefix} {config_name}
chmod 777 ./*
{platform.bin_copy_cmd.format(model_prefix=model_name, bpu_output_name=bpu_output_name)}
'''
    
    with open(bash_path_Vision, "w", encoding="utf-8") as f: 
        f.write(generate_bash(os.path.basename(config_yaml_path_Vision), BPU_VisionEncoder))
        
    with open(bash_path_Transformer, "w", encoding="utf-8") as f: 
        f.write(generate_bash(os.path.basename(config_yaml_path_Transformer), BPU_TransformerLayers))
        
    with open(bash_build_all_path, "w", encoding="utf-8") as f:
        f.write(f'''
#!/bin/bash
cd {BPU_VisionEncoder} && bash {os.path.basename(bash_path_Vision)} && cd ..
cd {BPU_TransformerLayers} && bash {os.path.basename(bash_path_Transformer)} && cd ..
echo "End of build all."
''')
    logging.info("Bash scripts generated.")

    # 6. 生成校准数据
    # 创建 Transformer 校准数据子目录
    input_names_TransformerLayers = camera_names + ["state"]
    for input_name in input_names_TransformerLayers:
        p = os.path.join(cal_data_path_Transformer, input_name)
        os.makedirs(p, exist_ok=True)

    logging.info("Generating calibration data...")
    
    # Use tqdm to wrap the dataloader for a progress bar
    for i, batch in tqdm(enumerate(dataloader), total=min(opt.cal_num, len(dataloader)), desc="Calibrating"):
        if i >= opt.cal_num: break
        
        file_name = platform.get_cal_data_name(i)
        batch = policy.normalize_inputs(batch)
        
        camera_inputs = {}
        for camera_name in camera_names:
            camera_inputs[camera_name] = batch[f'observation.images.{camera_name}']
        
        state_input = batch["observation.state"]
        
        # VisionEncoder Cal Data (Save every 4th sample, consistent with original)
        if i % 4 == 0:
            for camera_name in camera_names:
                p = os.path.join(cal_data_path_Vision, f"{camera_name}_" + file_name)
                platform.save_calibration(p, camera_inputs[camera_name])

        # TransformerLayers Cal Data (Input is Vision Features + State)
        for camera_name in camera_names:
            vision_feature = m_VisionEncoder(camera_inputs[camera_name])
            p = os.path.join(cal_data_path_Transformer, camera_name, file_name)
            platform.save_calibration(p, vision_feature)

        p = os.path.join(state_cal_data_path_Transformer, file_name)
        platform.save_calibration(p, state_input)

    # 结束打印
    print()
    print(colored("="*80, 'light_green'))
    print(colored(f"Export Success.", 'light_red'))
    os.system(f"tree {opt.export_path} -L 2 -h")
    print()
    print(colored("="*80, 'light_green'))
    print(colored("Reference Command: ", 'light_red'))
    print(f"[Docker] Run Command: [sudo] docker run [--gpus all] -it -v {os.path.join(os.getcwd(), opt.export_path)}:/open_explorer {REPOSITORY}:{TAG}")
    print(f"[BPU] Run Command: bash build_all.sh")
    print()
    print(colored("="*80, 'light_green'), "\n")


def onnx_sim(onnx_path, do_sim):   
    if do_sim:
        model_onnx = onnx.load(onnx_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        model_onnx, check = simplify(
            model_onnx,
            dynamic_input_shape=False,
            input_shapes=None)
        assert check, 'assert check failed'
        onnx.save(model_onnx, onnx_path)    

class BPU_ACTPolicy_VisionEncoder(nn.Module):
    def __init__(self, act_policy):
        super().__init__()
        self.backbone = deepcopy(act_policy.model.backbone)
        self.encoder_img_feat_input_proj = deepcopy(act_policy.model.encoder_img_feat_input_proj)
    def forward(self, images):
        cam_features = self.backbone(images)["feature_map"]
        cam_features = self.encoder_img_feat_input_proj(cam_features)
        return cam_features

class BPU_ACTPolicy_TransformerLayers(nn.Module):
    def __init__(self, act_policy, camera_names):
        super().__init__()
        self.model = deepcopy(act_policy.model)
        self.camera_names = camera_names

    def forward(self, states, *vision_features):
        latent_sample = torch.zeros([1, self.model.config.latent_dim], dtype=torch.float32)

        encoder_in_tokens = [self.model.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = self.model.encoder_1d_feature_pos_embed.weight.unsqueeze(1).unbind(dim=0)
        encoder_in_tokens.append(self.model.encoder_robot_state_input_proj(states))

        all_cam_features = []
        all_cam_pos_embeds = []

        # 动态处理所有相机的视觉特征
        for vision_feature in vision_features:
            cam_pos_embed = self.model.encoder_cam_feat_pos_embed(vision_feature)
            all_cam_features.append(vision_feature)
            all_cam_pos_embeds.append(cam_pos_embed)

        tokens = []
        for token in encoder_in_tokens:
            tokens.append(token.view(1,1,self.model.config.dim_model))
        all_cam_features = torch.cat(all_cam_features, axis=-1).permute(2, 3, 0, 1).view(-1,1,self.model.config.dim_model)
        tokens.append(all_cam_features)
        encoder_in_tokens = torch.cat(tokens, axis=0)

        pos_embeds = []
        for pos_embed in encoder_in_pos_embed:
            pos_embeds.append(pos_embed.view(1,1,self.model.config.dim_model))
        all_cam_pos_embeds = torch.cat(all_cam_pos_embeds, axis=-1).permute(2, 3, 0, 1).view(-1,1,self.model.config.dim_model)
        pos_embeds.append(all_cam_pos_embeds)
        encoder_in_pos_embed = torch.cat(pos_embeds, axis=0)

        encoder_out = self.model.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        decoder_in = torch.zeros(
            (self.model.config.chunk_size, 1, self.model.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.model.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.model.decoder_pos_embed.weight.unsqueeze(1),
        )

        decoder_out = decoder_out.transpose(0, 1)
        actions = self.model.action_head(decoder_out)
        return actions

if __name__ == "__main__":
    init_logging()
    _global_config = load_config_and_inject_args()
    main()
