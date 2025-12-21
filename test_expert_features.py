#!/usr/bin/env python3
"""
测试专家特征提取器的脚本
用于验证修复后的专家特征提取器是否能正确提取特征
"""

import os
import sys
import json
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.feature.expert_feature import ExpertFeatureExtractor

def load_sample_data():
    """加载样本数据用于测试"""
    # 加载赔率数据
    odds_file = '/Users/Williamhiler/Documents/my-project/train/train-data/odds/2017-2018_odds_features.json'
    with open(odds_file, 'r', encoding='utf-8') as f:
        odds_data = json.load(f)
    
    # 加载球队状态数据
    team_state_file = '/Users/Williamhiler/Documents/my-project/train/train-data/team_state/2017-2018_team_state_features.json'
    with open(team_state_file, 'r', encoding='utf-8') as f:
        team_state_data = json.load(f)
    
    # 加载专家特征数据
    expert_file = '/Users/Williamhiler/Documents/my-project/train/train-data/expert/2017-2018_expert_features.json'
    with open(expert_file, 'r', encoding='utf-8') as f:
        expert_features = json.load(f)
    
    return odds_data, team_state_data, expert_features

def test_expert_feature_extractor():
    """测试专家特征提取器"""
    print("开始测试专家特征提取器...")
    
    # 加载数据
    odds_data, team_state_data, expert_features = load_sample_data()
    
    # 创建提取器实例
    extractor = ExpertFeatureExtractor()
    
    # 随机选择5场比赛进行测试
    match_ids = list(team_state_data.keys())[:5]
    
    for match_id in match_ids:
        print(f"\n测试比赛ID: {match_id}")
        
        # 查找对应的赔率数据
        match_odds = None
        for match in odds_data['matches']:
            if match['match_id'] == match_id:
                match_odds = match
                break
        
        if not match_odds:
            print(f"  未找到对应的赔率数据")
            continue
        
        # 获取球队状态数据
        state_data = team_state_data[match_id]
        
        # 测试特征提取
        features = extractor.extract_expert_features(
            {'matchId': match_id},  # 简单的比赛数据
            match_odds,             # 赔率数据
            state_data              # 球队状态数据
        )
        
        print(f"  提取的专家特征:")
        for feature_name, feature_value in features.items():
            print(f"    {feature_name}: {feature_value:.6f}")
        
        # 验证特征值范围
        for feature_name, feature_value in features.items():
            if not (0 <= feature_value <= 1):
                print(f"    警告: {feature_name} 值 {feature_value} 超出正常范围 (0-1)")
    
    print("\n=== 特征统计信息 ===")
    # 统计所有专家特征的分布
    all_features = {}
    feature_counts = {}
    
    for match_id, features in expert_features.items():
        for feature_name, feature_value in features.items():
            if feature_name not in all_features:
                all_features[feature_name] = []
            all_features[feature_name].append(feature_value)
            feature_counts[feature_name] = feature_counts.get(feature_name, 0) + 1
    
    for feature_name, values in all_features.items():
        values_array = np.array(values)
        print(f"\n{feature_name}:")
        print(f"  样本数: {feature_counts[feature_name]}")
        print(f"  最小值: {np.min(values_array):.6f}")
        print(f"  最大值: {np.max(values_array):.6f}")
        print(f"  平均值: {np.mean(values_array):.6f}")
        print(f"  标准差: {np.std(values_array):.6f}")
    
    print("\n测试完成！")

if __name__ == "__main__":
    test_expert_feature_extractor()