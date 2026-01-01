#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
实用版增强特征工程器
集成专家知识推理功能
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# 添加项目路径
sys.path.append('/Users/Williamhiler/Documents/my-project/train/v5')

from utils.feature_engineering.feature_engineer import FeatureEngineer
from utils.expert_knowledge.practical_expert_reasoner import PracticalExpertKnowledgeReasoner


class PracticalEnhancedFeatureEngineer(FeatureEngineer):
    """实用版增强特征工程器 - 集成专家知识"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # 初始化专家知识推理器
        self.expert_reasoner = PracticalExpertKnowledgeReasoner(config)
        self.expert_features_enabled = config.get("features", {}).get("expert_analysis", {}).get("enabled", True)
        
        print(f"✅ 实用版增强特征工程器初始化完成")
        print(f"   - 专家知识推理: {'启用' if self.expert_features_enabled else '禁用'}")
    
    
    def create_expert_knowledge_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """创建基于专家知识的特征"""
        if not self.expert_features_enabled or not self.expert_reasoner.knowledge_base:
            print("⚠️  专家知识特征创建被跳过")
            return features
        
        print("🧠 开始创建专家知识特征...")
        
        enhanced_features = features.copy()
        expert_feature_list = []
        
        # 对每场比赛应用专家知识推理
        for idx, row in features.iterrows():
            if idx % 500 == 0:
                print(f"   处理进度: {idx}/{len(features)}")
            
            # 构建比赛上下文
            match_context = self._build_match_context(row)
            
            try:
                # 生成专家特征
                expert_features = self.expert_reasoner.create_expert_features(match_context)
                expert_feature_list.append(expert_features)
            except Exception as e:
                print(f"⚠️  第{idx}场比赛专家特征生成失败: {e}")
                # 使用默认特征
                default_features = self._get_default_expert_features()
                expert_feature_list.append(default_features)
        
        # 转换为DataFrame
        expert_df = pd.DataFrame(expert_feature_list)
        
        # 合并特征
        final_features = pd.concat([enhanced_features.reset_index(drop=True), expert_df.reset_index(drop=True)], axis=1)
        
        print(f"✅ 专家知识特征创建完成")
        print(f"   - 新增特征数量: {len(expert_df.columns)}")
        print(f"   - 专家特征列: {list(expert_df.columns)}")
        
        return final_features
    
    
    def _build_match_context(self, row: pd.Series) -> Dict:
        """构建比赛上下文"""
        # 提取核心赔率信息
        home_odds = float(row.get("home_win_odds", 2.0))
        draw_odds = float(row.get("draw_odds", 3.0))
        away_odds = float(row.get("away_win_odds", 3.0))
        
        # 构建匹配上下文
        context = {
            "home_win_odds": home_odds,
            "draw_odds": draw_odds,
            "away_win_odds": away_odds
        }
        
        return context
    
    
    def _get_default_expert_features(self) -> Dict:
        """获取默认的专家特征"""
        return {
            "expert_knowledge_count": 0,
            "expert_total_relevance": 0.0,
            "expert_confidence": 0.3,
            "expert_home_win_prob": 0.35,
            "expert_draw_prob": 0.30,
            "expert_away_win_prob": 0.35,
            "expert_home_adjustment": 0.0,
            "expert_draw_adjustment": 0.0,
            "expert_away_adjustment": 0.0,
            "expert_vs_implied_diff": 0.0
        }
    
    
    def engineer_features(self, features: pd.DataFrame, target: pd.Series = None) -> Tuple[pd.DataFrame, List[str]]:
        """完整的特征工程流程 - 实用版"""
        print("=== 实用版增强特征工程流程 ===")
        
        # 第一步：基础特征工程（继承父类方法）
        print("1️⃣ 基础特征工程...")
        enhanced_features = self.create_interaction_features(features)
        enhanced_features = self.create_polynomial_features(enhanced_features, degree=2)
        enhanced_features = self.create_ranking_features(enhanced_features)
        enhanced_features = self.create_categorical_features(enhanced_features)
        
        # 第二步：专家知识特征
        print("2️⃣ 专家知识特征...")
        if self.expert_features_enabled:
            enhanced_features = self.create_expert_knowledge_features(enhanced_features)
        
        # 第三步：处理分类特征（独热编码）
        print("3️⃣ 分类特征处理...")
        categorical_cols = enhanced_features.select_dtypes(include=['category', 'object']).columns
        if len(categorical_cols) > 0:
            enhanced_features = pd.get_dummies(enhanced_features, columns=categorical_cols, drop_first=True)
        
        # 第四步：特征选择
        print("4️⃣ 特征选择...")
        if target is not None:
            selected_features = self.select_features(enhanced_features, target, method="correlation", k=150)
            final_features = enhanced_features[selected_features + [target.name] if target.name in enhanced_features.columns else selected_features]
        else:
            final_features = enhanced_features
            selected_features = enhanced_features.columns.tolist()
        
        print(f"✅ 实用版增强特征工程完成！")
        print(f"   - 最终特征数量: {len(final_features.columns)}")
        print(f"   - 专家知识特征: {'已集成' if self.expert_features_enabled else '未集成'}")
        
        return final_features, selected_features


def main():
    """测试实用版增强特征工程器"""
    print("=== 实用版增强特征工程器测试 ===")
    
    # 配置
    config = {
        "data": {
            "raw_data_path": "/Users/Williamhiler/Documents/my-project/train/original-data"
        },
        "features": {
            "expert_analysis": {
                "enabled": True,
                "knowledge_base_path": "/Users/Williamhiler/Documents/my-project/train/v5/data/expert_knowledge/expert_knowledge_base.json"
            }
        }
    }
    
    # 创建实用版增强特征工程器
    engineer = PracticalEnhancedFeatureEngineer(config)
    
    # 创建测试数据
    test_data = {
        'home_team_name': ['曼联', '利物浦', '切尔西'],
        'away_team_name': ['利物浦', '曼城', '阿森纳'],
        'home_win_odds': [2.1, 3.2, 1.8],
        'draw_odds': [3.2, 3.4, 3.5],
        'away_win_odds': [3.4, 2.1, 4.2],
        'home_recent_points': [12, 8, 15],
        'away_recent_points': [10, 14, 9],
        'home_recent_wins': [4, 2, 5],
        'away_recent_wins': [3, 4, 3],
        'h2h_home_wins': [2, 1, 3],
        'h2h_draws': [1, 2, 1],
        'h2h_away_wins': [3, 3, 2],
        'match_date': ['2023-12-01', '2023-12-02', '2023-12-03'],
        'result': [3, 1, 3]  # 3:主胜, 1:平局, 0:客胜
    }
    
    df = pd.DataFrame(test_data)
    target = df['result']
    
    print(f"测试数据: {len(df)} 场比赛")
    
    # 应用实用版增强特征工程
    enhanced_features, selected_features = engineer.engineer_features(df, target)
    
    print(f"\n增强特征结果:")
    print(f"- 总行数: {len(enhanced_features)}")
    print(f"- 总列数: {len(enhanced_features.columns)}")
    
    # 检查专家知识特征
    expert_feature_cols = [col for col in enhanced_features.columns if col.startswith('expert_')]
    if expert_feature_cols:
        print(f"\n专家知识特征 ({len(expert_feature_cols)}个):")
        for col in expert_feature_cols:
            print(f"  - {col}: {enhanced_features[col].mean():.3f}")
    
    print("\n✅ 实用版增强特征工程器测试完成！")


if __name__ == "__main__":
    main()