#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更精确地收集未映射的球队ID
"""

import json
import os

# 读取训练数据
with open('/Users/Williamhiler/Documents/my-project/train/colab_training/match/match_train_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 从预处理脚本中读取当前的球队映射
with open('/Users/Williamhiler/Documents/my-project/train/temp/preprocess_match_data.py', 'r', encoding='utf-8') as f:
    script_content = f.read()

# 提取当前的球队映射
try:
    # 找到team_mapping字典的起始和结束位置
    start_pos = script_content.find('team_mapping = {')
    end_pos = script_content.find('}', start_pos) + 1
    mapping_str = script_content[start_pos:end_pos]
    
    # 执行这段代码来获取当前映射
    exec(mapping_str)
    current_team_mapping = team_mapping
    print(f"当前已映射 {len(current_team_mapping)} 个球队ID")
except Exception as e:
    print(f"无法读取当前球队映射：{e}")
    current_team_mapping = {}

# 收集未映射的球队ID
missing_team_ids = set()

for sample in data:
    text = sample['text']
    
    # 查找所有可能的球队ID上下文
    lines = text.split('\n')
    for line in lines:
        # 检查是否包含球队相关的关键词
        if any(keyword in line for keyword in ['对阵：', '主队近期战绩：', '客队近期战绩：', '历史交锋：', 'vs', 'VS']):
            # 提取所有数字
            words = line.split()
            for word in words:
                if word.isdigit():
                    team_id = word
                    # 检查是否已映射
                    if team_id not in current_team_mapping:
                        missing_team_ids.add(team_id)

print("\n未映射的球队ID：")
for team_id in sorted(missing_team_ids):
    print(f"  '{team_id}': '',")

print(f"\n共发现 {len(missing_team_ids)} 个未映射的球队ID")
