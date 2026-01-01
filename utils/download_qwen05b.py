#!/usr/bin/env python3
"""
手动下载Qwen 0.5B模型
使用方法：
1. 从其他途径下载模型文件（百度网盘、阿里云盘等）
2. 将文件放到 ./models/cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/xxx/ 目录
3. 确保包含以下文件：
   - config.json
   - tokenizer.json
   - tokenizer_config.json
   - merges.txt (如果使用BPE tokenizer)
   - vocab.txt (如果使用BPE tokenizer)
   - model.safetensors 或 pytorch_model.bin
"""

import os
import json

def create_model_structure():
    """创建模型目录结构"""
    base_dir = "./models/cache/models--Qwen--Qwen2.5-0.5B-Instruct"
    snapshots_dir = os.path.join(base_dir, "snapshots")
    
    # 创建必要的目录
    os.makedirs(snapshots_dir, exist_ok=True)
    
    # 创建说明文件
    readme = """
# Qwen2.5-0.5B-Instruct 模型目录

请将下载的模型文件放到这个目录下。

必需文件：
1. config.json - 模型配置
2. tokenizer.json - tokenizer文件
3. tokenizer_config.json - tokenizer配置
4. special_tokens_map.json - 特殊token映射
5. merges.txt - BPE合并规则（如果使用BPE）
6. vocab.txt - 词汇表（如果使用BPE）
7. model.safetensors 或 pytorch_model.bin - 模型权重

推荐从以下途径获取模型：
1. HuggingFace: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
2. 百度网盘/阿里云盘（国内加速）
3. 通过有网络的机器下载后传输
"""
    
    with open(os.path.join(base_dir, "README.md"), 'w', encoding='utf-8') as f:
        f.write(readme)
    
    print(f"✓ 目录结构已创建: {base_dir}")
    print(readme)

if __name__ == "__main__":
    create_model_structure()
