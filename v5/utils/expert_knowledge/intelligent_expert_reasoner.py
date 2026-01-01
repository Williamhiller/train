#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
智能专家知识推理器
结合Qwen大模型的语义理解能力，实现精准的专家知识匹配和推理
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# 导入Qwen知识匹配器
import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils.expert_knowledge.qwen_knowledge_matcher import QwenKnowledgeMatcher


class IntelligentExpertReasoner:
    """智能专家知识推理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 加载配置文件
        self._load_config_from_file()
        
        # 合并配置
        self.full_config = {**self.file_config, **config}
        
        # 初始化Qwen知识匹配器，传递完整配置
        self.qwen_matcher = QwenKnowledgeMatcher(self.full_config)
        
        # 知识类型权重（用于推理）
        self.knowledge_type_weights = {
            "patterns": 1.0,
            "techniques": 0.8,
            "practical": 0.7,
            "fundamentals": 0.6,
            "psychology": 0.5,
            "philosophy": 0.4,
            "risk_management": 0.6
        }
    
    
    def analyze_match_with_expert_knowledge(self, match_data: Dict) -> Dict:
        """使用专家知识分析比赛"""
        print(f"\n=== 智能专家知识分析 ===")
        
        # 兼容不同的球队名称字段
        home_team = match_data.get('home_team', match_data.get('homeTeam', match_data.get('homeTeamId', '主队')))
        away_team = match_data.get('away_team', match_data.get('awayTeam', match_data.get('awayTeamId', '客队')))
        print(f"比赛: {home_team} vs {away_team}")
        
        # 调试：打印match_data的关键字段
        print(f"调试信息: match_data包含的关键字段")
        print(f"  - 包含home_team: {'home_team' in match_data}")
        print(f"  - 包含away_team: {'away_team' in match_data}")
        print(f"  - 包含odds: {'odds' in match_data}")
        print(f"  - 包含history: {'history' in match_data}")
        print(f"  - 包含homeWinOdds: {'homeWinOdds' in match_data}")
        
        # 使用Qwen匹配器获取相关专家知识
        match_result = self.qwen_matcher.get_enhanced_match_result(match_data, top_k=8)
        relevant_knowledge = match_result['relevant_knowledge']
        
        print(f"匹配到 {len(relevant_knowledge)} 条相关专家知识")
        
        # 生成专家预测
        expert_prediction = self._generate_expert_prediction(
            relevant_knowledge, match_data
        )
        
        return expert_prediction
    
    
    def _generate_expert_prediction(self, relevant_knowledge: List[Dict], match_data: Dict) -> Dict:
        """生成专家预测"""
        if not relevant_knowledge:
            return self._get_default_prediction()
        
        # 基础概率（基于赔率）
        base_probs = self._get_base_probabilities_from_odds(match_data)
        
        # 根据专家知识调整概率
        adjustments = self._calculate_knowledge_adjustments(relevant_knowledge, match_data)
        
        # 应用调整
        final_probs = base_probs.copy()
        for outcome in adjustments:
            final_probs[outcome] += adjustments[outcome]
        
        # 归一化
        total_prob = sum(final_probs.values())
        for outcome in final_probs:
            final_probs[outcome] = max(0.01, final_probs[outcome] / total_prob)
        
        # 计算置信度
        confidence = self._calculate_confidence(relevant_knowledge)
        
        # 计算总相关性分数
        total_relevance = sum(k['relevance_score'] for k in relevant_knowledge)
        
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
    
    
    def _get_base_probabilities_from_odds(self, match_data: Dict) -> Dict:
        """从赔率获取基础概率"""
        # 提取赔率，使用正确的camelCase字段名
        home_odds = float(match_data.get('homeWinOdds', 2.0))
        draw_odds = float(match_data.get('drawOdds', 3.0))
        away_odds = float(match_data.get('awayWinOdds', 3.0))
        
        # 计算隐含概率（简化版，不考虑赔付率）
        implied_home = 1.0 / home_odds
        implied_draw = 1.0 / draw_odds
        implied_away = 1.0 / away_odds
        
        # 归一化
        total_implied = implied_home + implied_draw + implied_away
        
        return {
            "home": implied_home / total_implied,
            "draw": implied_draw / total_implied,
            "away": implied_away / total_implied
        }
    
    
    def _calculate_knowledge_adjustments(self, relevant_knowledge: List[Dict], 
                                        match_data: Dict) -> Dict:
        """基于专家知识计算概率调整"""
        adjustments = {"home": 0.0, "draw": 0.0, "away": 0.0}
        
        for knowledge in relevant_knowledge:
            unit = knowledge["unit"]
            relevance = knowledge["relevance_score"]
            knowledge_type = unit["knowledge_type"]
            content = unit["content"].lower()
            
            # 获取知识类型权重
            type_weight = self.knowledge_type_weights.get(knowledge_type, 0.5)
            
            # 计算该知识单元的总权重
            total_weight = relevance * type_weight
            
            # 根据内容进行调整
            # 1. 主胜相关调整
            if any(keyword in content for keyword in ["主胜", "主队", "主场", "优势"]):
                if any(positive in content for positive in ["信心", "强势", "看好", "支持"]):
                    adjustments["home"] += total_weight * 0.15
                elif any(negative in content for negative in ["阻力", "迷惑", "打击", "怀疑"]):
                    adjustments["home"] -= total_weight * 0.1
            
            # 2. 平局相关调整
            if any(keyword in content for keyword in ["平局", "平赔", "走盘"]):
                if any(positive in content for positive in ["分流", "保护", "合理", "可能"]):
                    adjustments["draw"] += total_weight * 0.12
                elif any(negative in content for negative in ["利诱", "任博", "陷阱"]):
                    adjustments["draw"] -= total_weight * 0.08
            
            # 3. 客胜相关调整
            if any(keyword in content for keyword in ["客胜", "客队", "客场", "客优"]):
                if any(positive in content for positive in ["信心", "强势", "看好", "支持"]):
                    adjustments["away"] += total_weight * 0.15
                elif any(negative in content for negative in ["阻力", "迷惑", "打击", "怀疑"]):
                    adjustments["away"] -= total_weight * 0.1
            
            # 4. 赔率模式相关调整
            if any(pattern in content for pattern in ["赔率", "盘口", "欧赔", "亚盘"]):
                # 检查是否有具体的赔率模式描述
                if "低赔" in content or "信心" in content:
                    # 低赔通常对应较强的一方
                    home_odds = float(match_data.get('homeWinOdds', 2.0))
                    away_odds = float(match_data.get('awayWinOdds', 3.0))
                    if home_odds < away_odds:
                        adjustments["home"] += total_weight * 0.1
                    else:
                        adjustments["away"] += total_weight * 0.1
                
                if "高平" in content or "利诱" in content:
                    adjustments["draw"] -= total_weight * 0.05
                
                if "分散" in content or "平衡" in content:
                    adjustments["draw"] += total_weight * 0.08
        
        return adjustments
    
    
    def _calculate_confidence(self, relevant_knowledge: List[Dict]) -> float:
        """计算置信度"""
        if not relevant_knowledge:
            return 0.3
        
        # 基于知识单元数量和平均相关度计算置信度
        num_knowledge = len(relevant_knowledge)
        avg_relevance = np.mean([k['relevance_score'] for k in relevant_knowledge])
        
        # 计算置信度（基础0.3，最高0.9）
        confidence = 0.3 + (min(num_knowledge, 8) / 8) * 0.3 + avg_relevance * 0.3
        
        return min(0.9, confidence)
    
    
    def _load_config_from_file(self):
        """从文件加载配置"""
        import yaml
        
        self.file_config = {}
        
        # 配置文件路径列表
        config_files = [
            "/Users/Williamhiler/Documents/my-project/train/config/config.yaml",
            "/Users/Williamhiler/Documents/my-project/train/v5/configs/v5_config.yaml"
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        file_config = yaml.safe_load(f)
                        # 合并配置，后面的配置会覆盖前面的
                        self.file_config.update(file_config)
                        print(f"✅ 加载配置文件: {config_file}")
                except Exception as e:
                    print(f"❌ 加载配置文件失败 {config_file}: {e}")
            else:
                print(f"⚠️  配置文件不存在: {config_file}")
    
    
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
        }
        
        return features
    
    
    def get_match_knowledge_summary(self, match_data: Dict) -> Dict:
        """获取比赛知识匹配摘要"""
        # 使用Qwen匹配器获取相关专家知识
        match_result = self.qwen_matcher.get_enhanced_match_result(match_data, top_k=5)
        
        # 提取关键信息
        knowledge_summary = {
            "match_info": {
                "home_team": match_data.get("homeTeamId", "主队"),
                "away_team": match_data.get("awayTeamId", "客队"),
                "home_win_odds": match_data.get("homeWinOdds"),
                "draw_odds": match_data.get("drawOdds"),
                "away_win_odds": match_data.get("awayWinOdds")
            },
            "knowledge_matching": {
                "total_matches": match_result["total_matches"],
                "average_relevance": match_result["match_analysis"]["average_relevance_score"],
                "highest_relevance": match_result["match_analysis"]["highest_relevance_score"],
                "knowledge_type_distribution": match_result["match_analysis"]["knowledge_type_distribution"]
            },
            "top_knowledge": [
                {
                    "title": k["unit"]["title"],
                    "knowledge_type": k["unit"]["knowledge_type"],
                    "relevance_score": k["relevance_score"],
                    "source_document": k["unit"]["source_document"]
                }
                for k in match_result["relevant_knowledge"]
            ]
        }
        
        return knowledge_summary


def main():
    """测试智能专家知识推理功能"""
    print("=== 智能专家知识推理器测试 ===")
    
    config = {
        "features": {
            "expert_analysis": {
                "enabled": True
            }
        }
    }
    
    try:
        # 创建推理器
        reasoner = IntelligentExpertReasoner(config)
        
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
            print(f"\n{'='*60}")
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
            
            # 获取知识匹配摘要
            knowledge_summary = reasoner.get_match_knowledge_summary(match)
            print(f"\n知识匹配摘要:")
            print(f"  - 匹配到 {knowledge_summary['knowledge_matching']['total_matches']} 条知识")
            print(f"  - 知识类型分布: {knowledge_summary['knowledge_matching']['knowledge_type_distribution']}")
            
            # 创建专家特征
            expert_features = reasoner.create_expert_features(match)
            print(f"\n专家特征 (部分):")
            for key, value in list(expert_features.items())[:5]:
                print(f"  - {key}: {value:.3f}")
        
        print(f"\n{'='*60}")
        print("✅ 智能专家知识推理器测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()