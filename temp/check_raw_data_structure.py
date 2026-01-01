#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看原始数据结构，了解球队ID的真正位置
"""

import json

# 读取一个原始数据文件
file_path = "/Users/Williamhiler/Documents/my-project/train/examples/英超_2015-2016_aggregated.json"

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 取第一个比赛数据
first_match_id, first_match_info = list(data.items())[0]

print(f"比赛ID：{first_match_id}")
print("\n比赛基本信息：")
print(f"  比赛时间：{first_match_info.get('matchTime')}")
print(f"  主队ID：{first_match_info.get('homeTeamId')}")
print(f"  客队ID：{first_match_info.get('awayTeamId')}")
print(f"  结果：{first_match_info.get('result')}")
print(f"  比分：{first_match_info.get('homeScore')}-{first_match_info.get('awayScore')}")

# 查看历史数据结构
print("\n主队历史数据结构：")
home_data = first_match_info.get('details', {}).get('history', {}).get('homeData', [])
for i, match in enumerate(home_data[:3]):
    print(f"  历史比赛 {i+1}：{match}")

print("\n客队历史数据结构：")
away_data = first_match_info.get('details', {}).get('history', {}).get('awayData', [])
for i, match in enumerate(away_data[:3]):
    print(f"  历史比赛 {i+1}：{match}")

print("\n历史交锋数据结构：")
history_data = first_match_info.get('details', {}).get('history', {}).get('historyData', [])
for i, match in enumerate(history_data[:3]):
    print(f"  交锋 {i+1}：{match}")
