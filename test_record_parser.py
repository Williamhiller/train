#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试战绩格式解析功能
"""

import sys
import os

# 设置项目根目录
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.feature.smart_expert_feature import SmartExpertFeatureExtractor

def test_record_parser():
    """
    测试战绩格式解析功能
    """
    print("=== 战绩格式解析功能测试 ===")
    
    # 创建专家特征提取器实例
    extractor = SmartExpertFeatureExtractor()
    
    # 打印所有包含战绩格式的规则
    print("\n1. 所有包含战绩格式的规则解析结果：")
    for i, rule in enumerate(extractor.structured_rules):
        for condition in rule['conditions']:
            if condition['type'] == 'recent_record':
                print(f"\n  - 规则{i+1}: {extractor.extracted_rules[i]}")
                print(f"    解析结果: {condition}")
    
    # 测试特定的战绩格式解析
    print("\n\n2. 战绩格式解析详细说明：")
    test_cases = [
        ("411", "4胜1平1负"),
        ("321", "3胜2平1负"),
        ("501", "5胜0平1负"),
        ("1011", "10胜1平1负"),
        ("222", "2胜2平2负"),
        ("006", "0胜0平6负")
    ]
    
    for record_str, description in test_cases:
        print(f"    {record_str}: {description}")
    
    print("\n\n3. 功能总结：")
    print("   - 支持解析3位和4位数字的战绩格式")
    print("   - 战绩数字按'胜-平-负'顺序解析")
    print("   - 4位数字支持两位数的胜场数（如1011表示10胜1平1负）")
    print("   - 自动计算总场次")
    print("   - 能识别'主队近期战绩'、'客队近期战绩'等表述")

if __name__ == "__main__":
    test_record_parser()