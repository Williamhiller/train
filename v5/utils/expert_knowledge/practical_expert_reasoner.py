#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
实用版专家知识智能推理模块
专注于赔率模式匹配和专家经验应用
"""

import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re


class PracticalExpertKnowledgeReasoner:
    """实用版专家知识推理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.knowledge_base = None
        self.reasoning_cache = {}
        
        # 加载知识库
        self.load_knowledge_base()
        
        # 赔率模式关键词映射
        self.odds_pattern_keywords = {
            "low_home": ["主胜", "主队", "信心", "强势", "低赔", "优势"],
            "medium_home": ["中庸", "平衡", "分散", "合理", "主胜"],
            "high_home": ["高赔", "阻力", "迷惑", "打击", "主胜"],
            
            "low_draw": ["低平", "分流", "保护", "平局", "平赔"],
            "medium_draw": ["中庸", "平衡", "合理", "平局", "平赔"],
            "high_draw": ["高平", "利诱", "任博", "平局", "平赔"],
            
            "low_away": ["低客胜", "客队", "信心", "客队优势"],
            "medium_away": ["中庸", "平衡", "客胜", "客队"],
            "high_away": ["高客胜", "阻力", "打击", "客胜"]
        }
    
    
    def load_knowledge_base(self):
        """加载知识库"""
        knowledge_base_path = "/Users/Williamhiler/Documents/my-project/train/v5/data/expert_knowledge/expert_knowledge_base.json"
        
        try:
            with open(knowledge_base_path, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
            print(f"✅ 成功加载知识库: {len(self.knowledge_base['knowledge_units'])} 个知识单元")
        except Exception as e:
            print(f"❌ 加载知识库失败: {e}")
            self.knowledge_base = None
    
    
    def analyze_match_with_expert_knowledge(self, match_data: Dict) -> Dict:
        """使用专家知识分析比赛"""
        # 提取赔率信息
        home_odds = float(match_data.get('home_win_odds', 2.0))
        draw_odds = float(match_data.get('draw_odds', 3.0))
        away_odds = float(match_data.get('away_win_odds', 3.0))
        
        # 赔率分类
        home_category = self._categorize_home_odds(home_odds)
        draw_category = self._categorize_draw_odds(draw_odds)
        away_category = self._categorize_away_odds(away_odds)
        
        print(f"赔率分类 - 主胜: {home_category}, 平局: {draw_category}, 客胜: {away_category}")
        
        # 找到相关专家知识
        relevant_knowledge = self._find_relevant_knowledge(
            home_category, draw_category, away_category, home_odds, draw_odds, away_odds
        )
        
        print(f"找到 {len(relevant_knowledge)} 条相关专家知识")
        
        # 生成专家预测
        expert_prediction = self._generate_expert_prediction(
            relevant_knowledge, home_category, draw_category, away_category
        )
        
        return expert_prediction
    
    
    def _categorize_home_odds(self, odds: float) -> str:
        """分类主胜赔率"""
        if odds <= 1.6:
            return "very_low_home"
        elif odds <= 2.0:
            return "low_home"
        elif odds <= 2.5:
            return "medium_home"
        elif odds <= 3.5:
            return "high_home"
        else:
            return "very_high_home"
    
    
    def _categorize_draw_odds(self, odds: float) -> str:
        """分类平局赔率"""
        if odds <= 2.9:
            return "low_draw"
        elif odds <= 3.3:
            return "medium_draw"
        elif odds <= 3.8:
            return "high_draw"
        else:
            return "very_high_draw"
    
    
    def _categorize_away_odds(self, odds: float) -> str:
        """分类客胜赔率"""
        if odds <= 2.1:
            return "low_away"
        elif odds <= 3.1:
            return "medium_away"
        elif odds <= 4.0:
            return "high_away"
        else:
            return "very_high_away"
    
    
    def _find_relevant_knowledge(self, home_category: str, draw_category: str, 
                               away_category: str, home_odds: float, draw_odds: float, away_odds: float) -> List[Dict]:
        """找到相关专家知识"""
        relevant_knowledge = []
        
        for i, unit in enumerate(self.knowledge_base["knowledge_units"]):
            # 计算相关性分数
            relevance_score = self._calculate_relevance_score(
                unit, home_category, draw_category, away_category, home_odds, draw_odds, away_odds
            )
            
            if relevance_score > 0.1:  # 降低阈值
                relevant_knowledge.append({
                    "index": i,
                    "unit": unit,
                    "relevance_score": relevance_score
                })
        
        # 按相关性排序
        relevant_knowledge.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return relevant_knowledge[:5]  # 返回最相关的5条
    
    
    def _calculate_relevance_score(self, unit: Dict, home_category: str, draw_category: str, 
                                 away_category: str, home_odds: float, draw_odds: float, away_odds: float) -> float:
        """计算相关性分数"""
        content = unit["content"].lower()
        relevance = 0.0
        
        # 赔率数值匹配
        odds_in_content = re.findall(r'\d+\.\d+', content)
        
        for odds_str in odds_in_content[:10]:  # 检查前10个赔率
            try:
                content_odds = float(odds_str)
                # 检查是否与当前赔率接近（容差±0.3）
                if abs(content_odds - home_odds) < 0.3:
                    relevance += 0.3
                if abs(content_odds - draw_odds) < 0.3:
                    relevance += 0.3
                if abs(content_odds - away_odds) < 0.3:
                    relevance += 0.3
            except ValueError:
                continue
        
        # 关键词模式匹配
        categories_to_check = [home_category, draw_category, away_category]
        
        for category in categories_to_check:
            base_category = category.replace("very_", "")  # 去掉very前缀
            if base_category in self.odds_pattern_keywords:
                keywords = self.odds_pattern_keywords[base_category]
                for keyword in keywords:
                    if keyword in content:
                        relevance += 0.1
                        break
        
        # 知识类型权重
        type_weights = {
            "patterns": 1.0,
            "techniques": 0.8,
            "practical": 0.7,
            "fundamentals": 0.6,
            "psychology": 0.5,
            "philosophy": 0.3,
            "risk_management": 0.6
        }
        
        relevance *= type_weights.get(unit["knowledge_type"], 0.5)
        
        return min(relevance, 1.0)
    
    
    def _generate_expert_prediction(self, relevant_knowledge: List[Dict], home_category: str, 
                                  draw_category: str, away_category: str) -> Dict:
        """生成专家预测"""
        if not relevant_knowledge:
            return self._get_default_prediction()
        
        # 基础概率（基于赔率分类）
        base_probs = self._get_base_probabilities(home_category, draw_category, away_category)
        
        # 根据专家知识调整
        adjustments = {"home": 0, "draw": 0, "away": 0}
        total_relevance = 0
        
        for knowledge in relevant_knowledge:
            unit = knowledge["unit"]
            relevance = knowledge["relevance_score"]
            content = unit["content"].lower()
            
            # 根据内容调整概率
            if "主胜" in content or "主队" in content:
                if "信心" in content or "优势" in content or "强势" in content:
                    adjustments["home"] += relevance * 0.15
                elif "阻力" in content or "迷惑" in content or "打击" in content:
                    adjustments["home"] -= relevance * 0.1
            
            if "平局" in content or "平赔" in content:
                if "分流" in content or "保护" in content or "合理" in content:
                    adjustments["draw"] += relevance * 0.12
                elif "利诱" in content or "任博" in content:
                    adjustments["draw"] -= relevance * 0.08
            
            if "客胜" in content or "客队" in content:
                if "信心" in content or "优势" in content:
                    adjustments["away"] += relevance * 0.15
                elif "阻力" in content or "打击" in content:
                    adjustments["away"] -= relevance * 0.1
            
            total_relevance += relevance
        
        # 应用调整
        final_probs = base_probs.copy()
        for outcome in adjustments:
            final_probs[outcome] += adjustments[outcome]
        
        # 归一化
        total_prob = sum(final_probs.values())
        for outcome in final_probs:
            final_probs[outcome] = max(0.01, final_probs[outcome] / total_prob)
        
        # 再次归一化
        total_prob = sum(final_probs.values())
        for outcome in final_probs:
            final_probs[outcome] = final_probs[outcome] / total_prob
        
        confidence = min(0.85, 0.4 + total_relevance * 0.15)  # 基础置信度0.4，最高0.85
        
        return {
            "home_win_prob": final_probs["home"],
            "draw_prob": final_probs["draw"],
            "away_win_prob": final_probs["away"],
            "confidence": confidence,
            "expert_knowledge_used": len(relevant_knowledge),
            "total_relevance": total_relevance,
            "adjustments": adjustments,
            "base_probabilities": base_probs
        }
    
    
    def _get_base_probabilities(self, home_category: str, draw_category: str, away_category: str) -> Dict:
        """获取基础概率"""
        # 基于赔率分类的基础概率
        base_probs = {
            "home": 0.35,  # 默认稍偏向主队
            "draw": 0.30,
            "away": 0.35
        }
        
        # 根据主胜赔率调整
        if "very_low_home" in home_category:
            base_probs["home"] = 0.55
            base_probs["draw"] = 0.25
            base_probs["away"] = 0.20
        elif "low_home" in home_category:
            base_probs["home"] = 0.45
            base_probs["draw"] = 0.28
            base_probs["away"] = 0.27
        elif "medium_home" in home_category:
            base_probs["home"] = 0.38
            base_probs["draw"] = 0.30
            base_probs["away"] = 0.32
        elif "high_home" in home_category:
            base_probs["home"] = 0.30
            base_probs["draw"] = 0.32
            base_probs["away"] = 0.38
        else:  # very_high_home
            base_probs["home"] = 0.25
            base_probs["draw"] = 0.30
            base_probs["away"] = 0.45
        
        return base_probs
    
    
    def _get_default_prediction(self) -> Dict:
        """获取默认预测"""
        return {
            "home_win_prob": 0.35,
            "draw_prob": 0.30,
            "away_win_prob": 0.35,
            "confidence": 0.3,
            "expert_knowledge_used": 0,
            "total_relevance": 0,
            "adjustments": {"home": 0, "draw": 0, "away": 0},
            "base_probabilities": {"home": 0.35, "draw": 0.30, "away": 0.35}
        }
    
    
    def create_expert_features(self, match_data: Dict) -> Dict:
        """创建基于专家知识的特征"""
        # 分析比赛
        expert_analysis = self.analyze_match_with_expert_knowledge(match_data)
        
        # 创建特征
        features = {
            # 基础特征
            "expert_knowledge_count": expert_analysis["expert_knowledge_used"],
            "expert_total_relevance": expert_analysis["total_relevance"],
            "expert_confidence": expert_analysis["confidence"],
            
            # 预测概率
            "expert_home_win_prob": expert_analysis["home_win_prob"],
            "expert_draw_prob": expert_analysis["draw_prob"],
            "expert_away_win_prob": expert_analysis["away_win_prob"],
            
            # 调整特征
            "expert_home_adjustment": expert_analysis["adjustments"]["home"],
            "expert_draw_adjustment": expert_analysis["adjustments"]["draw"],
            "expert_away_adjustment": expert_analysis["adjustments"]["away"],
            
            # 赔率模式特征
            "expert_vs_implied_diff": expert_analysis["home_win_prob"] - (1/float(match_data.get("home_win_odds", 2.0)) if match_data.get("home_win_odds", 2.0) > 0 else 0.5)
        }
        
        return features


def main():
    """测试实用版专家知识推理功能"""
    print("=== 实用版专家知识推理测试 ===")
    
    config = {
        "features": {
            "expert_analysis": {
                "enabled": True
            }
        }
    }
    
    # 创建推理器
    reasoner = PracticalExpertKnowledgeReasoner(config)
    
    if not reasoner.knowledge_base:
        print("知识库未加载，无法进行推理")
        return
    
    # 测试数据
    test_matches = [
        {
            "home_team": "曼城",
            "away_team": "利物浦",
            "home_win_odds": 1.8,
            "draw_odds": 3.4,
            "away_win_odds": 4.2
        },
        {
            "home_team": "曼联",
            "away_team": "切尔西",
            "home_win_odds": 2.5,
            "draw_odds": 3.2,
            "away_win_odds": 2.8
        },
        {
            "home_team": "阿森纳",
            "away_team": "热刺",
            "home_win_odds": 2.1,
            "draw_odds": 3.3,
            "away_win_odds": 3.4
        }
    ]
    
    for i, match in enumerate(test_matches, 1):
        print(f"\n{'='*50}")
        print(f"测试比赛 {i}: {match['home_team']} vs {match['away_team']}")
        print(f"赔率: 主胜{match['home_win_odds']} 平局{match['draw_odds']} 客胜{match['away_win_odds']}")
        
        # 分析比赛
        expert_analysis = reasoner.analyze_match_with_expert_knowledge(match)
        
        print(f"\n专家分析结果:")
        print(f"  - 使用专家知识: {expert_analysis['expert_knowledge_used']} 条")
        print(f"  - 总相关性: {expert_analysis['total_relevance']:.3f}")
        print(f"  - 置信度: {expert_analysis['confidence']:.3f}")
        print(f"  - 预测概率: 主胜{expert_analysis['home_win_prob']:.3f} 平局{expert_analysis['draw_prob']:.3f} 客胜{expert_analysis['away_win_prob']:.3f}")
        print(f"  - 调整幅度: 主胜{expert_analysis['adjustments']['home']:+.3f} 平局{expert_analysis['adjustments']['draw']:+.3f} 客胜{expert_analysis['adjustments']['away']:+.3f}")
        
        # 创建专家特征
        expert_features = reasoner.create_expert_features(match)
        
        print(f"\n专家特征:")
        for key, value in expert_features.items():
            if isinstance(value, float):
                print(f"  - {key}: {value:.3f}")
            else:
                print(f"  - {key}: {value}")
    
    print(f"\n{'='*50}")
    print("✅ 实用版专家知识推理测试完成！")


if __name__ == "__main__":
    main()