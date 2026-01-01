#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
收集所有未映射的球队ID
"""

import json
import os
import re

# 读取训练数据
with open('/Users/Williamhiler/Documents/my-project/train/colab_training/match/match_train_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取所有可能的球队ID
team_ids = set()

# 正则表达式匹配球队ID（数字）
team_id_pattern = re.compile(r'\b(\d+)\b')

for sample in data:
    text = sample['text']
    # 查找所有数字
    matches = team_id_pattern.findall(text)
    for match in matches:
        # 只保留可能是球队ID的数字（2-5位）
        if 2 <= len(match) <= 5:
            team_ids.add(match)

print("发现的球队ID：")
for team_id in sorted(team_ids):
    print(f"  '{team_id}': '',")

print(f"\n共发现 {len(team_ids)} 个球队ID")
