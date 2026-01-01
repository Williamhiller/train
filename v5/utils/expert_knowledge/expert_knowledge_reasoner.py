#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
专家知识智能推理模块
将预处理的知识与Qwen模型结合，实现智能推理
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re


class ExpertKnowledgeReasoner:
    """专家知识智能推理器"""
    
    def __init__(self, config: Dict, qwen_model=None):
        self.config = config
        self.qwen_model = qwen_model
        self.knowledge_base = None
        self.category_weights = {}
        self.reasoning_cache = {}
        
        # 加载知识库
        self.load_knowledge_base()
        
        # 初始化分类权重
        self.initialize_category_weights()
    
    
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
    
    
    def initialize_category_weights(self):
        """初始化分类权重"""
        self.category_weights = {
            "philosophy": 0.8,        # 哲学思维权重较高
            "fundamentals": 0.9,      # 基础理论权重最高
            "techniques": 0.85,       # 技巧方法
            "patterns": 0.9,          # 模式识别
            "practical": 0.7,         # 实战应用
            "psychology": 0.75,       # 心理分析
            "risk_management": 0.8   # 风险管理
        }
    
    
    def semantic_search(self, query: str, top_k: int = 10, category_filter: List[str] = None) -> List[Dict]:
        """语义搜索相关专家知识"""
        if not self.knowledge_base:
            return []
        
        # 简单的关键词匹配搜索（后续可替换为真正的语义搜索）
        relevant_units = []
        
        for i, unit in enumerate(self.knowledge_base["knowledge_units"]):
            # 分类过滤
            if category_filter and unit["knowledge_type"] not in category_filter:
                continue
            
            # 计算相关性分数
            relevance_score = self._calculate_relevance_score(query, unit)
            
            if relevance_score > 0.1:  # 阈值
                relevant_units.append({
                    "index": i,
                    "unit": unit,
                    "relevance_score": relevance_score
                })
        
        # 按相关性排序
        relevant_units.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return relevant_units[:top_k]
    
    
    def _calculate_relevance_score(self, query: str, unit: Dict) -> float:
        """计算相关性分数"""
        query_lower = query.lower()
        content_lower = unit["content"].lower()
        title_lower = unit.get("title", "").lower()
        
        score = 0.0
        
        # 标题匹配权重更高
        if query_lower in title_lower:
            score += 0.5
        
        # 内容匹配
        if query_lower in content_lower:
            score += 0.3
        
        # 关键词匹配
        keywords = query_lower.split()
        for keyword in keywords:
            if keyword in content_lower:
                score += 0.1
            if keyword in unit.get("key_concepts", []):
                score += 0.2
        
        # 分类权重
        category_weight = self.category_weights.get(unit["knowledge_type"], 0.5)
        score *= category_weight
        
        # 实用价值权重
        practical_value = unit.get("practical_value", {})
        avg_practical_score = np.mean([
            practical_value.get("actionability", 0),
            practical_value.get("specificity", 0),
            practical_value.get("uniqueness", 0),
            practical_value.get("clarity", 0)
        ])
        score *= (0.5 + 0.5 * avg_practical_score)
        
        return min(score, 1.0)
    
    
    def generate_expert_reasoning(self, match_context: Dict, expert_knowledge: List[Dict]) -> Dict:
        """生成专家推理"""
        if not self.qwen_model:
            # 如果没有Qwen模型，返回基于规则的推理
            return self._generate_rule_based_reasoning(match_context, expert_knowledge)
        
        # 构建推理prompt
        prompt = self._build_reasoning_prompt(match_context, expert_knowledge)
        
        # 使用Qwen生成推理
        try:
            reasoning_result = self.qwen_model.generate(prompt)
            return self._parse_reasoning_result(reasoning_result)
        except Exception as e:
            print(f"Qwen推理失败: {e}, 使用备用推理")
            return self._generate_rule_based_reasoning(match_context, expert_knowledge)
    
    
    def _build_reasoning_prompt(self, match_context: Dict, expert_knowledge: List[Dict]) -> str:
        """构建推理prompt"""
        # 比赛背景信息
        match_info = f"""
        比赛背景信息：
        - 主队: {match_context.get('home_team', '未知')} vs 客队: {match_context.get('away_team', '未知')}
        - 赔率: 主胜{match_context.get('home_odds', 'N/A')} 平局{match_context.get('draw_odds', 'N/A')} 客胜{match_context.get('away_odds', 'N/A')}
        - 近期状态: 主队{match_context.get('home_form', 'N/A')} 客队{match_context.get('away_form', 'N/A')}
        - 联赛: {match_context.get('league', 'N/A')} 轮次: {match_context.get('round', 'N/A')}
        """
        
        # 专家知识整合
        expert_insights = "\n相关专家知识：\n"
        for i, knowledge in enumerate(expert_knowledge, 1):
            unit = knowledge["unit"]
            expert_insights += f"""
            知识{i} (来源: {unit['source_document']} 第{unit['page_number']}页, 相关性: {knowledge['relevance_score']:.2f}):
            类型: {unit['knowledge_type']}
            内容: {unit['content'][:300]}...
            关键概念: {', '.join(unit.get('key_concepts', [])[:5])}
            """
        
        # 推理要求
        reasoning_requirements = """
        请基于以上比赛信息和专家知识，进行深度分析：
        
        1. 专家观点总结：这些专家知识的核心观点是什么？
        2. 适用性分析：这些观点在当前比赛场景下的适用程度如何？
        3. 推理逻辑：基于专家知识，应该如何推理这场比赛的结果？
        4. 风险评估：需要注意哪些潜在的风险和不确定性？
        5. 预测建议：综合专家知识，对这场比赛的预测建议是什么？
        
        请以结构化的方式输出分析结果，包括预测概率和置信度。
        """
        
        return match_info + expert_insights + reasoning_requirements
    
    
    def _parse_reasoning_result(self, reasoning_text: str) -> Dict:
        """解析推理结果"""
        # 简化的解析逻辑（可以根据Qwen的输出格式调整）
        result = {
            "expert_summary": "",
            "applicability_score": 0.5,
            "reasoning_logic": "",
            "risk_assessment": "",
            "prediction": {
                "home_win_prob": 0.33,
                "draw_prob": 0.34,
                "away_win_prob": 0.33,
                "confidence": 0.5,
                "main_factors": []
            },
            "raw_reasoning": reasoning_text
        }
        
        # 尝试提取概率信息
        prob_patterns = [
            r"主胜概率[:：]\s*(\d+(?:\.\d+)?)",
            r"平局概率[:：]\s*(\d+(?:\.\d+)?)",
            r"客胜概率[:：]\s*(\d+(?:\.\d+)?)",
            r"置信度[:：]\s*(\d+(?:\.\d+)?)"
        ]
        
        import re
        
        # 提取主胜概率
        home_match = re.search(prob_patterns[0], reasoning_text)
        if home_match:
            result["prediction"]["home_win_prob"] = float(home_match.group(1)) / 100 if float(home_match.group(1)) > 1 else float(home_match.group(1))
        
        # 提取平局概率
        draw_match = re.search(prob_patterns[1], reasoning_text)
        if draw_match:
            result["prediction"]["draw_prob"] = float(draw_match.group(1)) / 100 if float(draw_match.group(1)) > 1 else float(draw_match.group(1))
        
        # 提取客胜概率
        away_match = re.search(prob_patterns[2], reasoning_text)
        if away_match:
            result["prediction"]["away_win_prob"] = float(away_match.group(1)) / 100 if float(away_match.group(1)) > 1 else float(away_match.group(1))
        
        # 提取置信度
        confidence_match = re.search(prob_patterns[3], reasoning_text)
        if confidence_match:
            result["prediction"]["confidence"] = float(confidence_match.group(1)) / 100 if float(confidence_match.group(1)) > 1 else float(confidence_match.group(1))
        
        return result
    
    
    def _generate_rule_based_reasoning(self, match_context: Dict, expert_knowledge: List[Dict]) -> Dict:
        """基于规则的备用推理"""
        # 提取赔率信息
        home_odds = float(match_context.get('home_odds', 2.0))
        draw_odds = float(match_context.get('draw_odds', 3.0))
        away_odds = float(match_context.get('away_odds', 3.0))
        
        # 基于专家知识的权重调整
        weights = {
            "home_win": 1.0,
            "draw": 1.0,
            "away_win": 1.0
        }
        
        # 根据知识类型调整权重
        for knowledge in expert_knowledge:
            unit = knowledge["unit"]
            knowledge_type = unit["knowledge_type"]
            relevance = knowledge["relevance_score"]
            
            # 根据不同类型的知识调整预测权重
            if knowledge_type == "patterns":
                # 模式识别知识
                if "主胜" in unit["content"] or "主队" in unit["content"]:
                    weights["home_win"] += relevance * 0.3
                elif "平局" in unit["content"] or "平赔" in unit["content"]:
                    weights["draw"] += relevance * 0.3
                elif "客胜" in unit["content"] or "客队" in unit["content"]:
                    weights["away_win"] += relevance * 0.3
            
            elif knowledge_type == "techniques":
                # 技巧方法知识
                if "拉力" in unit["content"] or "信心" in unit["content"]:
                    # 根据赔率位置判断
                    if home_odds < 2.0:
                        weights["home_win"] += relevance * 0.2
                    elif away_odds < 2.0:
                        weights["away_win"] += relevance * 0.2
            
            elif knowledge_type == "psychology":
                # 心理分析知识
                if "心理" in unit["content"] or "信心" in unit["content"]:
                    # 心理因素影响相对较小
                    for key in weights:
                        weights[key] += relevance * 0.1
        
        # 归一化概率
        total_weight = sum(weights.values())
        home_prob = weights["home_win"] / total_weight
        draw_prob = weights["draw"] / total_weight
        away_prob = weights["away_win"] / total_weight
        
        # 计算平均相关性作为置信度
        avg_relevance = np.mean([k["relevance_score"] for k in expert_knowledge]) if expert_knowledge else 0.5
        
        return {
            "expert_summary": f"基于{len(expert_knowledge)}条专家知识的规则推理",
            "applicability_score": avg_relevance,
            "reasoning_logic": "基于专家知识类型的权重调整",
            "risk_assessment": "置信度取决于专家知识的相关性",
            "prediction": {
                "home_win_prob": home_prob,
                "draw_prob": draw_prob,
                "away_win_prob": away_prob,
                "confidence": min(avg_relevance + 0.3, 1.0),
                "main_factors": [k["unit"]["knowledge_type"] for k in expert_knowledge[:3]]
            },
            "raw_reasoning": "基于规则的备用推理"
        }
    
    
    def analyze_match_with_expert_knowledge(self, match_data: Dict) -> Dict:
        """使用专家知识分析比赛"""
        # 构建搜索查询
        search_queries = []
        
        # 基于赔率构建查询
        home_odds = match_data.get("home_win_odds", 0)
        draw_odds = match_data.get("draw_odds", 0)
        away_odds = match_data.get("away_win_odds", 0)
        
        if home_odds and draw_odds and away_odds:
            # 根据赔率区间构建查询
            if home_odds < 1.5:
                search_queries.append("低赔主胜")
            elif home_odds < 2.0:
                search_queries.append("中赔主胜")
            else:
                search_queries.append("高赔主胜")
            
            if draw_odds < 3.0:
                search_queries.append("低平赔")
            elif draw_odds > 3.5:
                search_queries.append("高平赔")
        
        # 基于球队状态构建查询
        home_form = match_data.get("home_recent_points", 0)
        away_form = match_data.get("away_recent_points", 0)
        
        if home_form is not None and away_form is not None:
            if home_form > away_form + 3:
                search_queries.append("主队状态好")
            elif away_form > home_form + 3:
                search_queries.append("客队状态好")
            else:
                search_queries.append("状态接近")
        
        # 搜索相关专家知识
        all_expert_knowledge = []
        
        for query in search_queries:
            results = self.semantic_search(query, top_k=5)
            all_expert_knowledge.extend(results)
        
        # 去重并排序
        seen_indices = set()
        unique_knowledge = []
        
        for knowledge in all_expert_knowledge:
            if knowledge["index"] not in seen_indices:
                seen_indices.add(knowledge["index"])
                unique_knowledge.append(knowledge)
        
        unique_knowledge.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # 限制最多使用10条知识
        unique_knowledge = unique_knowledge[:10]
        
        print(f"找到 {len(unique_knowledge)} 条相关专家知识")
        
        # 生成专家推理
        expert_reasoning = self.generate_expert_reasoning(match_data, unique_knowledge)
        
        return {
            "match_data": match_data,
            "expert_knowledge_used": unique_knowledge,
            "expert_reasoning": expert_reasoning,
            "search_queries": search_queries
        }
    
    
    def create_expert_features(self, match_data: Dict) -> Dict:
        """创建基于专家知识的特征"""
        analysis_result = self.analyze_match_with_expert_knowledge(match_data)
        
        expert_features = {
            # 基础特征
            "expert_knowledge_count": len(analysis_result["expert_knowledge_used"]),
            "avg_expert_relevance": np.mean([k["relevance_score"] for k in analysis_result["expert_knowledge_used"]]) if analysis_result["expert_knowledge_used"] else 0,
            "max_expert_relevance": max([k["relevance_score"] for k in analysis_result["expert_knowledge_used"]]) if analysis_result["expert_knowledge_used"] else 0,
            
            # 预测特征
            "expert_home_win_prob": analysis_result["expert_reasoning"]["prediction"]["home_win_prob"],
            "expert_draw_prob": analysis_result["expert_reasoning"]["prediction"]["draw_prob"],
            "expert_away_win_prob": analysis_result["expert_reasoning"]["prediction"]["away_win_prob"],
            "expert_confidence": analysis_result["expert_reasoning"]["prediction"]["confidence"],
            
            # 知识类型特征
            "philosophy_knowledge_count": sum(1 for k in analysis_result["expert_knowledge_used"] if k["unit"]["knowledge_type"] == "philosophy"),
            "techniques_knowledge_count": sum(1 for k in analysis_result["expert_knowledge_used"] if k["unit"]["knowledge_type"] == "techniques"),
            "patterns_knowledge_count": sum(1 for k in analysis_result["expert_knowledge_used"] if k["unit"]["knowledge_type"] == "patterns"),
            "practical_knowledge_count": sum(1 for k in analysis_result["expert_knowledge_used"] if k["unit"]["knowledge_type"] == "practical"),
            
            # 适用性特征
            "expert_applicability_score": analysis_result["expert_reasoning"]["applicability_score"],
            
            # 搜索查询特征
            "search_query_count": len(analysis_result["search_queries"]),
            "search_queries": " | ".join(analysis_result["search_queries"])
        }
        
        return expert_features


def main():
    """测试专家知识推理功能"""
    print("=== 专家知识智能推理测试 ===")
    
    # 配置
    config = {
        "features": {
            "expert_analysis": {
                "enabled": True,
                "knowledge_base_path": "/Users/Williamhiler/Documents/my-project/train/v5/data/expert_knowledge/expert_knowledge_base.json"
            }
        }
    }
    
    # 创建推理器
    reasoner = ExpertKnowledgeReasoner(config)
    
    if not reasoner.knowledge_base:
        print("知识库未加载，无法进行推理")
        return
    
    # 测试数据
    test_match = {
        "home_team": "曼联",
        "away_team": "利物浦",
        "home_win_odds": 2.1,
        "draw_odds": 3.2,
        "away_win_odds": 3.4,
        "home_recent_points": 12,
        "away_recent_points": 10,
        "league": "英超",
        "round": "第15轮"
    }
    
    print("测试比赛数据:")
    print(f"{test_match['home_team']} vs {test_match['away_team']}")
    print(f"赔率: 主胜{test_match['home_win_odds']} 平局{test_match['draw_odds']} 客胜{test_match['away_win_odds']}")
    
    # 分析比赛
    analysis_result = reasoner.analyze_match_with_expert_knowledge(test_match)
    
    print(f"\n专家推理结果:")
    print(f"使用专家知识: {len(analysis_result['expert_knowledge_used'])} 条")
    print(f"搜索查询: {analysis_result['search_queries']}")
    
    reasoning = analysis_result["expert_reasoning"]
    prediction = reasoning["prediction"]
    
    print(f"\n预测结果:")
    print(f"主胜概率: {prediction['home_win_prob']:.3f}")
    print(f"平局概率: {prediction['draw_prob']:.3f}")
    print(f"客胜概率: {prediction['away_win_prob']:.3f}")
    print(f"置信度: {prediction['confidence']:.3f}")
    
    # 生成专家特征
    expert_features = reasoner.create_expert_features(test_match)
    
    print(f"\n专家特征:")
    for feature, value in expert_features.items():
        if isinstance(value, float):
            print(f"{feature}: {value:.3f}")
        else:
            print(f"{feature}: {value}")
    
    print("\n✅ 专家知识推理测试完成！")


if __name__ == "__main__":
    main()