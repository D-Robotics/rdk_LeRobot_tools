import pandas as pd
import json
import os
import glob
from typing import Dict, List, Any, Iterator

def replace_keys_in_dict(data: Any) -> Any:
    """
    递归替换字典中的键名
    - action → action.joint
    - observation.state → observation.state.joint
    """
    if isinstance(data, dict):
        new_dict: Dict[str, Any] = {}
        for key, value in data.items():
            new_key = {
                'action': 'action.joint',
                'observation.state': 'observation.state.joint'
            }.get(key, key)
            new_dict[new_key] = replace_keys_in_dict(value)
        return new_dict
    elif isinstance(data, list):
        return [replace_keys_in_dict(item) for item in data]
    else:
        return data

def process_parquet(file_path: str) -> bool:
    """直接处理 Parquet 文件并覆盖原文件"""
    try:
        print(f"处理 Parquet: {os.path.basename(file_path)}")
        df = pd.read_parquet(file_path, engine='auto')
        
        # 构建列名映射
        column_mapping = {}
        for old_col, new_col in [
            ('action', 'action.joint'),
            ('observation.state', 'observation.state.joint')
        ]:
            if old_col in df.columns:
                column_mapping[old_col] = new_col
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
            print(f"  └─ 替换列名: {list(column_mapping.items())}")
        else:
            print(f"  └─ 无需要替换的列")
        
        # 直接覆盖原文件
        df.to_parquet(file_path, index=False, engine='auto')
        print(f"  └─ 已覆盖原文件\n")
        return True
    except Exception as e:
        print(f"  └─ 错误: {str(e)}\n")
        return False

def process_json(file_path: str) -> bool:
    """直接处理 JSON 文件并覆盖原文件"""
    try:
        print(f"处理 JSON: {os.path.basename(file_path)}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        modified_data = replace_keys_in_dict(data)
        
        # 直接覆盖原文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(modified_data, f, indent=2, ensure_ascii=False, sort_keys=False)
        
        print(f"  └─ 已覆盖原文件\n")
        return True
    except json.JSONDecodeError as e:
        print(f"  └─ 错误: JSON 格式无效 - {str(e)}\n")
        return False
    except Exception as e:
        print(f"  └─ 错误: {str(e)}\n")
        return False

def process_jsonl(file_path: str) -> bool:
    """直接处理 JSONL 文件并覆盖原文件"""
    try:
        print(f"处理 JSONL: {os.path.basename(file_path)}")
        
        modified_lines: List[str] = []
        line_count = 0
        error_lines = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    modified_lines.append('')  # 保留空行
                    continue
                
                try:
                    data = json.loads(line)
                    modified_data = replace_keys_in_dict(data)
                    modified_line = json.dumps(modified_data, ensure_ascii=False)
                    modified_lines.append(modified_line)
                    line_count += 1
                except json.JSONDecodeError as e:
                    modified_lines.append(line)  # 保留错误行，不修改
                    print(f"  ├─ 警告: 第 {line_num} 行 JSON 格式无效 - {str(e)}（已保留原行）")
                    error_lines += 1
        
        # 直接覆盖原文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(modified_lines) + '\n')
        
        print(f"  └─ 处理完成: 有效行 {line_count} | 错误行 {error_lines}")
        print(f"  └─ 已覆盖原文件\n")
        return True if line_count > 0 or error_lines == 0 else False
    except Exception as e:
        print(f"  └─ 错误: {str(e)}\n")
        return False

def process_folder(folder_path: str) -> None:
    """递归处理文件夹中的所有 JSON、JSONL 和 Parquet 文件，直接覆盖原文件"""
    folder_path = os.path.abspath(folder_path)
    if not os.path.isdir(folder_path):
        print(f"❌ 错误：文件夹 '{folder_path}' 不存在或不是目录")
        return
    
    # 查找所有目标文件（递归）
    parquet_files = glob.glob(os.path.join(folder_path, '**', '*.parquet'), recursive=True)
    json_files = glob.glob(os.path.join(folder_path, '**', '*.json'), recursive=True)
    jsonl_files = glob.glob(os.path.join(folder_path, '**', '*.jsonl'), recursive=True)
    
    total_files = len(parquet_files) + len(json_files) + len(jsonl_files)
    if total_files == 0:
        print(f"ℹ️ 在 '{folder_path}' 中未找到 JSON、JSONL 或 Parquet 文件")
        return
    
    # 显示统计信息
    print("=" * 80)
    print(f"📂 处理根目录: {folder_path}")
    print(f"⚠️  警告：处理后将直接覆盖原文件，请确保已备份！")
    print(f"📊 找到文件：{len(parquet_files)} 个 Parquet | {len(json_files)} 个 JSON | {len(jsonl_files)} 个 JSONL")
    print("=" * 80 + "\n")
    
    success_count = 0
    
    # 处理 Parquet
    if parquet_files:
        print("🔧 开始处理 Parquet 文件：")
        print("-" * 40)
        for file_path in parquet_files:
            if process_parquet(file_path):
                success_count += 1
    
    # 处理 JSON
    if json_files:
        print("🔧 开始处理 JSON 文件：")
        print("-" * 40)
        for file_path in json_files:
            if process_json(file_path):
                success_count += 1
    
    # 处理 JSONL
    if jsonl_files:
        print("🔧 开始处理 JSONL 文件：")
        print("-" * 40)
        for file_path in jsonl_files:
            if process_jsonl(file_path):
                success_count += 1
    
    # 汇总结果
    print("=" * 80)
    print(f"✅ 处理完成！")
    print(f"📈 结果：成功 {success_count}/{total_files} 个文件")
    print(f"⚠️  所有修改已直接应用到原文件")
    print("=" * 80)

if __name__ == "__main__":
    # --------------------------
    # 配置区域：修改这里的路径
    # --------------------------
    folder_path = "./data"  # 替换为你的文件夹路径
    
    process_folder(folder_path)