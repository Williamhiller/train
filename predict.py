#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
足球比赛结果预测脚本

使用方法：
    python predict.py --model_path models/fine_tuned_model --input_file examples/sample_input.json
"""

import argparse
import os
import sys
import json
import torch
from utils.model_utils import (
    load_fine_tuned_model,
    get_model_generation_params
)
from utils.prediction_utils import (
    prepare_prediction_input,
    parse_prediction_output,
    load_prediction_input,
    save_prediction_result,
    validate_prediction_input
)

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='足球比赛结果预测')
    parser.add_argument('--model_path', type=str, 
                        required=True,
                        help='模型路径')
    parser.add_argument('--input_file', type=str, 
                        required=True,
                        help='输入文件路径')
    parser.add_argument('--output_file', type=str, 
                        default='prediction_result.json',
                        help='输出文件路径')
    parser.add_argument('--base_model', type=str, 
                        default='meta-llama/Llama-3.2-1B',
                        help='基础模型名称（如果使用PEFT模型）')
    parser.add_argument('--quantization', action='store_true',
                        default=True,
                        help='是否使用模型量化')
    parser.add_argument('--max_new_tokens', type=int,
                        default=100,
                        help='生成的最大token数量')
    parser.add_argument('--temperature', type=float,
                        default=0.7,
                        help='生成温度')
    parser.add_argument('--top_p', type=float,
                        default=0.9,
                        help='top_p采样参数')
    return parser.parse_args()

def generate_prediction(model, tokenizer, prompt: str, generation_params: dict) -> str:
    """
    使用模型生成预测结果
    
    Args:
        model: 加载的模型
        tokenizer: 分词器
        prompt: 输入提示
        generation_params: 生成参数
        
    Returns:
        生成的预测文本
    """
    # 确保模型在正确的设备上
    device = next(model.parameters()).device
    
    # 编码输入
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        padding=True,
        truncation=True
    ).to(device)
    
    # 生成预测
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_params
        )
    
    # 解码输出
    # 只获取生成的部分（不包括输入）
    input_length = inputs['input_ids'].shape[1]
    generated_text = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    )
    
    return generated_text.strip()

def predict_single_match(
    model,
    tokenizer,
    input_data: dict,
    generation_params: dict
) -> dict:
    """
    预测单个比赛结果
    
    Args:
        model: 模型
        tokenizer: 分词器
        input_data: 输入数据
        generation_params: 生成参数
        
    Returns:
        预测结果
    """
    print(f"\n预测比赛: {input_data.get('home_team')} vs {input_data.get('away_team')}")
    
    # 准备输入提示
    prompt = prepare_prediction_input(input_data)
    print("输入提示:")
    print(prompt)
    print("\n生成预测中...")
    
    # 生成预测
    generated_text = generate_prediction(model, tokenizer, prompt, generation_params)
    print("\n原始输出:")
    print(generated_text)
    
    # 解析预测结果
    result = parse_prediction_output(generated_text)
    
    # 添加比赛信息
    result['home_team'] = input_data.get('home_team')
    result['away_team'] = input_data.get('away_team')
    result['league'] = input_data.get('league', '未知联赛')
    
    # 格式化结果显示
    print("\n预测结果:")
    print(f"- 预测: {result['prediction']} (W=主队胜, D=平局, L=客队胜)")
    print(f"- 置信度: {result['confidence']:.2f}")
    print(f"- 概率分布: {result['detailed_probabilities']}")
    if result['score']:
        print(f"- 预测比分: {result['score']}")
    
    return result

def main():
    """
    主函数
    """
    args = parse_args()
    
    # 检查模型路径是否存在
    if not os.path.exists(args.model_path):
        print(f"错误：模型路径 {args.model_path} 不存在")
        print("请先运行 train.py 训练模型")
        sys.exit(1)
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"错误：输入文件 {args.input_file} 不存在")
        sys.exit(1)
    
    # 加载输入数据
    input_data = load_prediction_input(args.input_file)
    
    # 验证输入数据
    if not validate_prediction_input(input_data):
        print("输入数据验证失败，请检查输入文件格式")
        sys.exit(1)
    
    # 设置生成参数
    generation_params = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": True,
        "pad_token_id": None,  # 将在加载模型后设置
        "eos_token_id": None   # 将在加载模型后设置
    }
    
    print(f"\n加载模型: {args.model_path}")
    
    # 加载模型
    try:
        model, tokenizer = load_fine_tuned_model(
            model_path=args.model_path,
            base_model_name=args.base_model,
            quantization=args.quantization
        )
        
        # 更新生成参数中的token id
        generation_params["pad_token_id"] = tokenizer.pad_token_id
        generation_params["eos_token_id"] = tokenizer.eos_token_id
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("尝试以CPU模式加载...")
        # 尝试以CPU模式加载
        try:
            model, tokenizer = load_fine_tuned_model(
                model_path=args.model_path,
                base_model_name=args.base_model,
                quantization=False,
                device="cpu"
            )
            generation_params["pad_token_id"] = tokenizer.pad_token_id
            generation_params["eos_token_id"] = tokenizer.eos_token_id
        except Exception as e2:
            print(f"CPU模式加载也失败: {e2}")
            sys.exit(1)
    
    # 执行预测
    print("\n开始预测...")
    result = predict_single_match(model, tokenizer, input_data, generation_params)
    
    # 保存预测结果
    save_prediction_result(result, args.output_file)
    
    print(f"\n预测完成！结果已保存至: {args.output_file}")

if __name__ == "__main__":
    main()