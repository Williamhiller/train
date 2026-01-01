#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查更新后的数据中是否还有未映射的球队ID
"""

import json
import re

# 读取更新后的训练数据
with open('/Users/Williamhiler/Documents/my-project/train/colab_training/match/match_train_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 从预处理脚本中读取当前的球队映射
with open('/Users/Williamhiler/Documents/my-project/train/temp/preprocess_match_data.py', 'r', encoding='utf-8') as f:
    script_content = f.read()

# 提取当前的球队映射
try:
    start_pos = script_content.find('team_mapping = {')
    end_pos = script_content.find('}', start_pos) + 1
    mapping_str = script_content[start_pos:end_pos]
    exec(mapping_str)
    current_team_mapping = team_mapping
    print(f"当前已映射 {len(current_team_mapping)} 个球队ID")
except Exception as e:
    print(f"无法读取当前球队映射：{e}")
    current_team_mapping = {}

# 检查数据中是否还有未映射的球队ID
print("\n检查数据中的球队ID...")

# 统计未映射的球队ID
unmapped_team_ids = set()

for sample in data[:10]:  # 只检查前10个样本
    text = sample['text']
    lines = text.split('\n')
    for line in lines:
        if any(keyword in line for keyword in ['对阵：', '主队近期战绩：', '客队近期战绩：', '历史交锋：']):
            # 提取所有数字
            words = line.split()
            for word in words:
                if word.isdigit():
                    if word not in current_team_mapping:
                        unmapped_team_ids.add(word)
                        print(f"  发现未映射的球队ID：{word} 在 '{line[:50]}...'")

if unmapped_team_ids:
    print(f"\n共发现 {len(unmapped_team_ids)} 个未映射的球队ID")
    print("未映射的球队ID：", sorted(unmapped_team_ids))
else:
    print("\n✓ 所有球队ID都已映射！")
