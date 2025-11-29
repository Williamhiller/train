#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
足球比赛预测系统接口

提供多种方式进行足球比赛预测的统一接口
支持文件输入、API调用、交互式预测等多种模式
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from utils.prediction_utils import (
    prepare_prediction_input,
    parse_prediction_output,
    validate_prediction_input
)
from utils.model_utils import load_fine_tuned_model

def format_prediction_result(result):
    """
    格式化预测结果为易读形式
    
    Args:
        result: 预测结果字典
        
    Returns:
        格式化的预测结果字符串
    """
    lines = []
    lines.append("\n========= 预测结果 =========")
    lines.append(f"比赛: {result.get('home_team', 'N/A')} vs {result.get('away_team', 'N/A')}")
    lines.append(f"联赛: {result.get('league', 'N/A')}")
    lines.append(f"预测结果: {result.get('prediction', '未知')} (W=主胜, D=平局, L=客胜)")
    lines.append(f"置信度: {result.get('confidence', 0):.2f}")
    lines.append(f"概率分布: {result.get('detailed_probabilities', '{}')}")
    if result.get('score'):
        lines.append(f"预测比分: {result.get('score')}")
    if result.get('analysis'):
        lines.append(f"\n分析理由: {result.get('analysis')}")
    lines.append("==========================\n")
    return '\n'.join(lines)

def interactive_prediction(model, tokenizer, generation_params):
    """
    交互式预测模式
    
    Args:
        model: 加载的模型
        tokenizer: 分词器
        generation_params: 生成参数
    """
    print("\n====== 交互式足球比赛预测 ======")
    print("输入 'exit' 退出，'help' 查看帮助")
    
    while True:
        try:
            # 获取基本信息
            home_team = input("主队名称: ").strip()
            if home_team.lower() in ['exit', 'quit', 'q']:
                break
            if home_team.lower() == 'help':
                print_help()
                continue
                
            away_team = input("客队名称: ").strip()
            league = input("联赛名称: ").strip()
            
            # 获取赔率信息
            try:
                home_odds = float(input("主队胜率赔率: "))
                draw_odds = float(input("平局赔率: "))
                away_odds = float(input("客队胜率赔率: "))
            except ValueError:
                print("赔率格式错误，使用默认值")
                home_odds = 3.0
                draw_odds = 3.5
                away_odds = 2.5
            
            # 获取排名信息
            try:
                home_rank = int(input("主队排名: "))
                away_rank = int(input("客队排名: "))
            except ValueError:
                print("排名格式错误，使用默认值")
                home_rank = 10
                away_rank = 10
            
            # 构建预测数据
            prediction_data = {
                "home_team": home_team,
                "away_team": away_team,
                "league": league,
                "odds": {
                    "home_win": home_odds,
                    "draw": draw_odds,
                    "away_win": away_odds
                },
                "home_team_rank": home_rank,
                "away_team_rank": away_rank
            }
            
            # 执行预测
            result = predict_single_match(model, tokenizer, prediction_data, generation_params)
            
            # 显示结果
            print(format_prediction_result(result))
            
            # 保存结果选项
            save_result = input("是否保存结果? (y/n): ").strip().lower()
            if save_result == 'y':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"prediction_{home_team}_{away_team}_{timestamp}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"结果已保存至: {filename}")
                
        except KeyboardInterrupt:
            print("\n程序已中断")
            break
        except Exception as e:
            print(f"发生错误: {e}")
    
    print("\n感谢使用足球比赛预测系统！")

def batch_prediction(model, tokenizer, input_file, output_file, generation_params):
    """
    批量预测模式
    
    Args:
        model: 加载的模型
        tokenizer: 分词器
        input_file: 输入文件路径
        output_file: 输出文件路径
        generation_params: 生成参数
    """
    print(f"\n开始批量预测，输入文件: {input_file}")
    
    # 加载输入数据
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 确保是列表格式
        if isinstance(data, dict):
            # 单个预测的情况
            predictions = [data]
        elif isinstance(data, list):
            # 批量预测的情况
            predictions = data
        else:
            raise ValueError("输入文件必须包含JSON对象或数组")
            
    except Exception as e:
        print(f"加载输入文件失败: {e}")
        return False
    
    # 执行批量预测
    results = []
    total = len(predictions)
    
    for i, prediction_data in enumerate(predictions):
        print(f"\n[{i+1}/{total}] 预测: {prediction_data.get('home_team', 'N/A')} vs {prediction_data.get('away_team', 'N/A')}")
        
        try:
            # 验证输入数据
            if not validate_prediction_input(prediction_data):
                print(f"  警告: 数据验证失败，跳过此预测")
                continue
                
            # 执行预测
            result = predict_single_match(model, tokenizer, prediction_data, generation_params)
            results.append(result)
            
            # 显示简要结果
            print(f"  预测: {result.get('prediction', '未知')} (置信度: {result.get('confidence', 0):.2f})")
            
            # 为避免过度请求，添加小延迟
            if i < total - 1:
                time.sleep(0.5)
                
        except Exception as e:
            print(f"  预测失败: {e}")
    
    # 保存结果
    if results:
        try:
            # 构建最终结果对象
            final_result = {
                "timestamp": datetime.now().isoformat(),
                "total_predictions": total,
                "successful_predictions": len(results),
                "predictions": results
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            
            print(f"\n批量预测完成！")
            print(f"成功预测: {len(results)}/{total}")
            print(f"结果已保存至: {output_file}")
            return True
            
        except Exception as e:
            print(f"保存结果失败: {e}")
            return False
    else:
        print("没有成功的预测结果")
        return False

def print_help():
    """
    显示帮助信息
    """
    print("\n=== 交互式足球比赛预测帮助 ===")
    print("输入比赛相关信息进行预测")
    print("必填字段:")
    print("  - 主队名称和客队名称")
    print("  - 联赛名称")
    print("  - 赔率信息 (主队胜、平局、客队胜)")
    print("  - 排名信息")
    print("\n提示:")
    print("  - 输入'exit'或'q'退出程序")
    print("  - 输入'help'显示此帮助")
    print("  - 赔率和排名请输入数字\n")

def predict_single_match(model, tokenizer, input_data, generation_params):
    """
    预测单个比赛结果（从predict.py导入的功能）
    
    Args:
        model: 模型
        tokenizer: 分词器
        input_data: 输入数据
        generation_params: 生成参数
        
    Returns:
        预测结果
    """
    import torch
    
    # 准备输入提示
    prompt = prepare_prediction_input(input_data)
    
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
    input_length = inputs['input_ids'].shape[1]
    generated_text = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    ).strip()
    
    # 解析预测结果
    result = parse_prediction_output(generated_text)
    
    # 添加比赛信息
    result['home_team'] = input_data.get('home_team')
    result['away_team'] = input_data.get('away_team')
    result['league'] = input_data.get('league', '未知联赛')
    result['timestamp'] = datetime.now().isoformat()
    
    return result

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='足球比赛预测系统接口')
    
    # 模式选择
    parser.add_argument('--mode', type=str,
                        choices=['interactive', 'batch', 'api'],
                        default='interactive',
                        help='预测模式: interactive (交互式), batch (批量), api (API服务)')
    
    # 通用参数
    parser.add_argument('--model_path', type=str,
                        default='models/fine_tuned_model',
                        help='模型路径')
    parser.add_argument('--base_model', type=str,
                        default='meta-llama/Llama-3.2-1B',
                        help='基础模型名称')
    parser.add_argument('--quantization', action='store_true',
                        default=True,
                        help='是否使用模型量化')
    
    # 批量模式参数
    parser.add_argument('--input_file', type=str,
                        help='批量预测的输入文件路径')
    parser.add_argument('--output_file', type=str,
                        default='batch_predictions.json',
                        help='批量预测的输出文件路径')
    
    # 生成参数
    parser.add_argument('--max_new_tokens', type=int,
                        default=100,
                        help='生成的最大token数量')
    parser.add_argument('--temperature', type=float,
                        default=0.7,
                        help='生成温度')
    parser.add_argument('--top_p', type=float,
                        default=0.9,
                        help='top_p采样参数')
    
    args = parser.parse_args()
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        print(f"错误：模型路径 {args.model_path} 不存在")
        print("请先运行 train.py 训练模型，或者指定正确的模型路径")
        sys.exit(1)
    
    # 批量模式需要输入文件
    if args.mode == 'batch' and not args.input_file:
        print("错误：批量模式需要指定输入文件 --input_file")
        parser.print_help()
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
        
        # 更新生成参数
        generation_params["pad_token_id"] = tokenizer.pad_token_id
        generation_params["eos_token_id"] = tokenizer.eos_token_id
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("尝试以CPU模式加载...")
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
    
    # 根据模式执行预测
    print(f"\n预测模式: {args.mode}")
    
    if args.mode == 'interactive':
        interactive_prediction(model, tokenizer, generation_params)
    elif args.mode == 'batch':
        batch_prediction(model, tokenizer, args.input_file, args.output_file, generation_params)
    elif args.mode == 'api':
        print("API服务模式正在开发中...")
        # 这里可以实现一个简单的REST API服务
        # 例如使用FastAPI或Flask

if __name__ == "__main__":
    main()