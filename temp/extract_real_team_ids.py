#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从原始比赛数据中提取所有真实的球队ID
"""

import json
import os
import glob

# 获取所有聚合数据文件
input_dir = "/Users/Williamhiler/Documents/my-project/train/examples"
file_pattern = os.path.join(input_dir, "*_aggregated.json")
files = glob.glob(file_pattern)

# 收集所有球队ID
all_team_ids = set()

for file_path in files:
    print(f"处理文件：{os.path.basename(file_path)}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for match_id, match_info in data.items():
        # 提取主客队ID
        home_team_id = str(match_info.get("homeTeamId", ""))
        away_team_id = str(match_info.get("awayTeamId", ""))
        
        if home_team_id:
            all_team_ids.add(home_team_id)
        if away_team_id:
            all_team_ids.add(away_team_id)
        
        # 提取历史数据中的球队ID
        details = match_info.get("details", {})
        history = details.get("history", {})
        
        # 主队历史数据
        home_data = history.get("homeData", [])
        for match in home_data:
            if isinstance(match, list) and len(match) >= 2:
                team_id = str(match[1])  # 对手球队ID
                all_team_ids.add(team_id)
        
        # 客队历史数据
        away_data = history.get("awayData", [])
        for match in away_data:
            if isinstance(match, list) and len(match) >= 2:
                team_id = str(match[1])  # 对手球队ID
                all_team_ids.add(team_id)
        
        # 历史交锋数据
        history_data = history.get("historyData", [])
        for match in history_data:
            if isinstance(match, list) and len(match) >= 2:
                team_id1 = str(match[0])
                team_id2 = str(match[1])
                all_team_ids.add(team_id1)
                all_team_ids.add(team_id2)

print(f"\n从 {len(files)} 个文件中提取到 {len(all_team_ids)} 个真实球队ID：")
for team_id in sorted(all_team_ids):
    print(f"  '{team_id}': '',")

# 保存到文件
output_file = "/Users/Williamhiler/Documents/my-project/train/temp/all_team_ids.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    for team_id in sorted(all_team_ids):
        f.write(f"'{team_id}': '',\n")

print(f"\n已保存到文件：{output_file}")
