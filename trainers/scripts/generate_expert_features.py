import sys
import os
import json

# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成所有赛季的专家特征，并保存到train-data/expert目录下
"""

import sys
import os
import json

# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.feature.smart_expert_feature import SmartExpertFeatureExtractor

def generate_expert_features():
    """
    生成所有赛季的专家特征
    """
    # 初始化专家特征提取器
    extractor = SmartExpertFeatureExtractor()
    
    # 获取所有赛季
    seasons = [
        '2015-2016',
        '2016-2017',
        '2017-2018',
        '2018-2019',
        '2019-2020',
        '2020-2021',
        '2021-2022',
        '2022-2023',
        '2023-2024',
        '2024-2025'
    ]
    
    for season in seasons:
        print(f"\n=== 正在处理赛季 {season} ===")
        
        # 读取赔率特征
        odds_features_path = f"/Users/Williamhiler/Documents/my-project/train/train-data/odds/{season}_odds_features.json"
        if not os.path.exists(odds_features_path):
            print(f"未找到赛季 {season} 的赔率特征文件")
            continue
        
        with open(odds_features_path, 'r', encoding='utf-8') as f:
            odds_features_json = json.load(f)
        
        # 赔率特征结构是 {'season': 'xxxx-xxxx', 'matches': [{'match_id': 'xxxx', ...}, ...]}
        odds_matches = odds_features_json.get('matches', [])
        
        # 创建赔率特征字典，以match_id为键
        odds_features = {}
        for match in odds_matches:
            match_id = match.get('match_id')
            if match_id:
                odds_features[match_id] = match
        
        # 读取球队状态特征
        team_state_features_path = f"/Users/Williamhiler/Documents/my-project/train/train-data/team_state/{season}_team_state_features.json"
        if not os.path.exists(team_state_features_path):
            print(f"未找到赛季 {season} 的球队状态特征文件")
            continue
        
        with open(team_state_features_path, 'r', encoding='utf-8') as f:
            team_state_features = json.load(f)
        
        # 生成专家特征
        expert_features_data = {}
        
        # 遍历所有比赛
        for match_id, team_state_data in team_state_features.items():
            if match_id not in odds_features:
                print(f"比赛 {match_id} 没有赔率特征，跳过")
                continue
            
            # 获取赔率特征
            odds_data = odds_features[match_id]
            
            # 构造比赛基本数据
            match_data = {
                'matchId': match_id,
                'homeTeamId': team_state_data['home_team']['id'],
                'awayTeamId': team_state_data['away_team']['id']
            }
            
            try:
                # 提取专家特征
                expert_features = extractor.extract_expert_features(match_data, odds_data, team_state_data)
                expert_features_data[match_id] = expert_features
                print(f"已提取比赛 {match_id} 的专家特征")
            except Exception as e:
                print(f"提取比赛 {match_id} 的专家特征失败: {e}")
                continue
        
        # 保存专家特征
        if expert_features_data:
            extractor.save_expert_features(season, expert_features_data)
            print(f"赛季 {season} 共提取 {len(expert_features_data)} 场比赛的专家特征")
        else:
            print(f"赛季 {season} 未提取到专家特征")
    
    print("\n所有赛季的专家特征提取完成！")

if __name__ == "__main__":
    generate_expert_features()