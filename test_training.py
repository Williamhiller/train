#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练测试脚本，用于快速验证训练流程

使用方法：
    python test_training.py
"""

import os
import json
import tempfile
from utils.model_utils import format_training_prompt

def create_test_data(output_dir: str):
    """
    创建测试数据
    
    Args:
        output_dir: 输出目录
    """
    # 创建小型测试数据集
    test_data = [
        {
            "input": "预测以下足球比赛的结果：\n主队：巴塞罗那\n客队：皇家马德里\n主队赔率：2.1\n平局赔率：3.4\n客队赔率：3.2",
            "output": "结果: 平局\n比分: 2-2"
        },
        {
            "input": "预测以下足球比赛的结果：\n主队：曼城\n客队：利物浦\n主队赔率：1.85\n平局赔率：3.6\n客队赔率：4.2",
            "output": "结果: 主队胜\n比分: 3-1"
        }
    ]
    
    # 保存训练和验证数据
    train_file = os.path.join(output_dir, 'train.json')
    val_file = os.path.join(output_dir, 'validation.json')
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(test_data[:1], f, ensure_ascii=False, indent=2)
    
    return train_file, val_file

def create_test_config(train_file: str, val_file: str, output_dir: str):
    """
    创建测试配置文件
    
    Args:
        train_file: 训练文件路径
        val_file: 验证文件路径
        output_dir: 输出目录
    """
    config = {
        "model": {
            "name": "meta-llama/Llama-3.2-1B",
            "max_length": 512,
            "quantization": True,
            "quant_bits": 4
        },
        "training": {
            "output_dir": os.path.join(output_dir, "test_model"),
            "num_epochs": 1,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "logging_steps": 1,
            "save_steps": 1,
            "evaluation_strategy": "steps",
            "eval_steps": 1,
            "seed": 42,
            "fp16": False,
            "push_to_hub": False
        },
        "data": {
            "train_file": train_file,
            "validation_file": val_file,
            "test_file": val_file,
            "sample_rate": 1.0
        },
        "inference": {
            "model_path": os.path.join(output_dir, "test_model"),
            "max_new_tokens": 50,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        },
        "peft": {
            "use_peft": True,
            "peft_type": "lora",
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "v_proj"]
        },
        "logging": {
            "log_level": "INFO",
            "log_file": os.path.join(output_dir, "test_training.log")
        }
    }
    
    config_file = os.path.join(output_dir, 'test_config.yaml')
    
    import yaml
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    return config_file

def main():
    """
    主函数
    """
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"创建临时测试环境: {temp_dir}")
        
        # 创建测试数据
        train_file, val_file = create_test_data(temp_dir)
        print(f"创建测试数据: {train_file}, {val_file}")
        
        # 创建测试配置
        config_file = create_test_config(train_file, val_file, temp_dir)
        print(f"创建测试配置: {config_file}")
        
        print("\n测试训练流程准备就绪！")
        print("您可以使用以下命令运行测试训练:")
        print(f"python train.py --config {config_file}")
        print("\n注意：这只是一个流程测试，由于模型和数据都很简化，")
        print("实际预测效果不会很好。完成流程测试后，您应该使用")
        print("真实的足球比赛数据进行完整训练。")

if __name__ == "__main__":
    main()