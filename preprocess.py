#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
足球比赛数据预处理脚本

使用方法：
    python preprocess.py --input_file examples/sample_training_data.json --output_dir data
"""

import argparse
import os
import sys
from utils.data_processor import preprocess_and_split_data

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='足球比赛数据预处理')
    parser.add_argument('--input_file', type=str, 
                        default='examples/sample_training_data.json',
                        help='输入的原始数据文件路径')
    parser.add_argument('--output_dir', type=str, 
                        default='data',
                        help='输出目录')
    parser.add_argument('--test_ratio', type=float, 
                        default=0.1,
                        help='测试集比例')
    parser.add_argument('--val_ratio', type=float, 
                        default=0.1,
                        help='验证集比例')
    return parser.parse_args()

def main():
    """
    主函数
    """
    args = parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"错误：输入文件 {args.input_file} 不存在")
        sys.exit(1)
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置输出文件路径
    train_output = os.path.join(args.output_dir, 'train.json')
    val_output = os.path.join(args.output_dir, 'validation.json')
    test_output = os.path.join(args.output_dir, 'test.json')
    
    print(f"开始预处理数据...")
    print(f"输入文件: {args.input_file}")
    print(f"输出目录: {args.output_dir}")
    
    # 预处理并分割数据
    preprocess_and_split_data(
        file_path=args.input_file,
        train_output=train_output,
        val_output=val_output,
        test_output=test_output,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio
    )
    
    print("数据预处理完成！")
    print(f"训练数据: {train_output}")
    print(f"验证数据: {val_output}")
    print(f"测试数据: {test_output}")

if __name__ == "__main__":
    main()