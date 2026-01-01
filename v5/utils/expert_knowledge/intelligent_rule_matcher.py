#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
智能规则匹配器
利用Qwen的语义理解能力，为每场比赛智能匹配相关的专家规则
"""

import json
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
import torch


@dataclass
class MatchedRule:
    """匹配的规则"""
    rule_id: str
    title: str
    category: str
    content: str
    relevance_score: float  # 相关度分数
    match_reason: str  # 匹配原因


class IntelligentRuleMatcher:
    """智能规则匹配器"""
    
    def __init__(self, config: Dict, rules_path: str):
        self.config = config
        self.rules_path = rules_path
        self.rules: List[Dict] = []
        self.rule_embeddings: np.ndarray = None
        
        # 初始化Qwen模型用于语义理解
        self.model_path = config.get("qwen", {}).get("model_path")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"正在加载Qwen模型用于规则匹配...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        print(f"✅ Qwen模型加载成功")
        
        # 加载规则
        self.load_rules()
    
    def load_rules(self):
        """加载专家规则"""
        print(f"正在加载专家规则: {self.rules_path}")
        
        if not os.path.exists(self.rules_path):
            print(f"❌ 规则文件不存在: {self.rules_path}")
            return
        
        with open(self.rules_path, 'r', encoding='utf-8') as f:
            self.rules = json.load(f)
        
        print(f"✅ 加载了 {len(self.rules)} 条专家规则")
        
        # 生成规则嵌入
        self._generate_rule_embeddings()
    
    def _generate_rule_embeddings(self):
        """生成所有规则的嵌入向量"""
        print("正在生成规则嵌入向量...")
        
        embeddings = []
        for rule in self.rules:
            # 构建规则文本表示
            rule_text = self._construct_rule_text(rule)
            embedding = self._generate_embedding(rule_text)
            embeddings.append(embedding)
        
        self.rule_embeddings = np.array(embeddings)
        print(f"✅ 规则嵌入生成完成: {self.rule_embeddings.shape}")
    
    def _construct_rule_text(self, rule: Dict) -> str:
        """构建规则文本表示
        
        Args:
            rule: 规则字典
            
        Returns:
            规则文本
        """
        text_parts = []
        
        # 添加标题和类别
        text_parts.append(f"规则类别: {rule.get('category', '其他')}")
        text_parts.append(f"规则标题: {rule.get('title', '')}")
        
        # 添加内容
        text_parts.append(f"规则内容: {rule.get('content', '')}")
        
        # 添加关键要点
        key_points = rule.get('key_points', [])
        if key_points:
            text_parts.append("关键要点:")
            for point in key_points:
                text_parts.append(f"- {point}")
        
        # 添加适用条件
        conditions = rule.get('conditions', [])
        if conditions:
            text_parts.append("适用条件:")
            for condition in conditions:
                text_parts.append(f"- {condition}")
        
        return '\n'.join(text_parts)
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """生成文本嵌入向量
        
        Args:
            text: 文本
            
        Returns:
            嵌入向量
        """
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用[CLS]标记的输出
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            
            # 归一化
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
            
        except Exception as e:
            print(f"❌ 生成嵌入失败: {e}")
            return np.zeros(768)
    
    def match_rules_for_match(self, match_data: Dict, top_k: int = 3) -> List[MatchedRule]:
        """为比赛匹配相关的专家规则
        
        Args:
            match_data: 比赛数据
            top_k: 返回前k个最相关的规则
            
        Returns:
            匹配的规则列表
        """
        # 构建比赛查询文本
        query_text = self._construct_match_query(match_data)
        query_embedding = self._generate_embedding(query_text)
        
        # 计算与所有规则的相似度
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.rule_embeddings
        )[0]
        
        # 获取top-k最相关的规则
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        matched_rules = []
        for idx in top_indices:
            rule = self.rules[idx]
            relevance_score = similarities[idx]
            
            # 生成匹配原因
            match_reason = self._generate_match_reason(match_data, rule, relevance_score)
            
            matched_rule = MatchedRule(
                rule_id=rule['rule_id'],
                title=rule['title'],
                category=rule['category'],
                content=rule['content'],
                relevance_score=float(relevance_score),
                match_reason=match_reason
            )
            matched_rules.append(matched_rule)
        
        return matched_rules
    
    def _construct_match_query(self, match_data: Dict) -> str:
        """构建比赛查询文本
        
        Args:
            match_data: 比赛数据
            
        Returns:
            查询文本
        """
        query_parts = []
        
        # 添加比赛基本信息
        home_team = match_data.get('homeTeamName', match_data.get('home_team_name', '主队'))
        away_team = match_data.get('awayTeamName', match_data.get('away_team_name', '客队'))
        query_parts.append(f"比赛: {home_team} vs {away_team}")
        
        # 添加赔率信息
        william_home = match_data.get('william_home_odds', 0)
        william_draw = match_data.get('william_draw_odds', 0)
        william_away = match_data.get('william_away_odds', 0)
        
        if william_home > 0:
            query_parts.append(f"威廉赔率: 主胜{william_home:.2f} 平局{william_draw:.2f} 客胜{william_away:.2f}")
        
        # 添加赔率变化
        william_home_change = match_data.get('william_home_odds_change', 0)
        if abs(william_home_change) > 0.01:
            query_parts.append(f"主胜赔率变化: {william_home_change:+.2f}")
        
        # 添加排名信息
        home_rank = match_data.get('home_rank', 0)
        away_rank = match_data.get('away_rank', 0)
        if home_rank > 0 and away_rank > 0:
            query_parts.append(f"排名: 主队第{home_rank}名 客队第{away_rank}名")
        
        # 添加近期状态
        home_form = match_data.get('home_form', '')
        away_form = match_data.get('away_form', '')
        if home_form and away_form:
            query_parts.append(f"近期状态: 主队{home_form} 客队{away_form}")
        
        # 添加对战历史
        h2h = match_data.get('h2h_home_wins', 0)
        h2h_draws = match_data.get('h2h_draws', 0)
        h2h_away_wins = match_data.get('h2h_away_wins', 0)
        if h2h + h2h_draws + h2h_away_wins > 0:
            query_parts.append(f"对战历史: 主队{h2h}胜 {h2h_draws}平 客队{h2h_away_wins}胜")
        
        return '\n'.join(query_parts)
    
    def _generate_match_reason(self, match_data: Dict, rule: Dict, relevance_score: float) -> str:
        """生成匹配原因
        
        Args:
            match_data: 比赛数据
            rule: 规则
            relevance_score: 相关度分数
            
        Returns:
            匹配原因
        """
        reasons = []
        
        # 基于规则类别
        category = rule.get('category', '')
        if '赔率' in category:
            reasons.append(f"规则类别为{category}，适用于赔率分析")
        elif '开盘' in category:
            reasons.append(f"规则类别为{category}，适用于开盘思路")
        elif '平衡' in category:
            reasons.append(f"规则类别为{category}，适用于平衡分析")
        
        # 基于赔率数值
        william_home = match_data.get('william_home_odds', 0)
        if william_home > 0:
            if william_home < 1.75:
                reasons.append(f"主胜赔率{william_home:.2f}低于1.75，符合低赔率分析场景")
            elif william_home < 2.25:
                reasons.append(f"主胜赔率{william_home:.2f}在1.75-2.25区间")
            elif william_home < 3.25:
                reasons.append(f"主胜赔率{william_home:.2f}在2.25-3.25区间")
            else:
                reasons.append(f"主胜赔率{william_home:.2f}高于3.25，属于高赔率场景")
        
        # 基于排名差距
        home_rank = match_data.get('home_rank', 0)
        away_rank = match_data.get('away_rank', 0)
        if home_rank > 0 and away_rank > 0:
            rank_diff = abs(home_rank - away_rank)
            if rank_diff > 10:
                reasons.append(f"排名差距{rank_diff}较大，需要考虑实力对比")
        
        # 基于相关度
        reasons.append(f"语义相关度: {relevance_score:.3f}")
        
        return '; '.join(reasons)
    
    def format_matched_rules(self, matched_rules: List[MatchedRule]) -> str:
        """格式化匹配的规则为文本
        
        Args:
            matched_rules: 匹配的规则列表
            
        Returns:
            格式化的文本
        """
        if not matched_rules:
            return "未找到相关的专家规则"
        
        formatted_parts = ["专家分析思路："]
        
        for i, rule in enumerate(matched_rules, 1):
            formatted_parts.append(f"\n{i}. {rule.title}（{rule.category}）")
            formatted_parts.append(f"   相关度: {rule.relevance_score:.3f}")
            formatted_parts.append(f"   匹配原因: {rule.match_reason}")
            formatted_parts.append(f"   规则内容: {rule.content[:200]}...")
        
        return '\n'.join(formatted_parts)


if __name__ == "__main__":
    import sys
    import yaml
    
    # 加载配置
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/v5_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建匹配器
    rules_path = "data/expert_knowledge/expert_rules.json"
    matcher = IntelligentRuleMatcher(config, rules_path)
    
    # 测试匹配
    test_match = {
        "homeTeamName": "纽卡斯尔联",
        "awayTeamName": "莱斯特城",
        "william_home_odds": 1.44,
        "william_draw_odds": 4.60,
        "william_away_odds": 7.00,
        "william_home_odds_change": -0.16,
        "home_rank": 3,
        "away_rank": 19,
        "home_form": "WWWDLL",
        "away_form": "WWWDDL",
        "h2h_home_wins": 4,
        "h2h_draws": 1,
        "h2h_away_wins": 5
    }
    
    # 匹配规则
    matched_rules = matcher.match_rules_for_match(test_match, top_k=3)
    
    # 格式化输出
    formatted = matcher.format_matched_rules(matched_rules)
    print("\n" + "="*80)
    print(formatted)
    print("="*80)
