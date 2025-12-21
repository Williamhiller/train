#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专家分析特征提取模块
将专家的分析思路转化为可用于机器学习的特征
"""

import os
import json
import re
from datetime import datetime

class ExpertFeatureExtractor:
    def __init__(self):
        """
        初始化专家特征提取器
        """
        self.expert_features_path = "/Users/Williamhiler/Documents/my-project/train/train-data/expert/expert_features_analysis.json"
        self.expertise_path = "/Users/Williamhiler/Documents/my-project/train/train-data/expert/expertise_analysis.json"
        self.pdf_texts_path = "/Users/Williamhiler/Documents/my-project/train/train-data/expert/pdf_texts.json"
        
        # 加载专家分析数据
        self.expert_features = self._load_expert_features()
        self.expertise = self._load_expertise()
        self.pdf_texts = self._load_pdf_texts()
    
    def _load_expert_features(self):
        """
        加载专家特征分析数据
        """
        if os.path.exists(self.expert_features_path):
            with open(self.expert_features_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _load_expertise(self):
        """
        加载专家分析思路数据
        """
        if os.path.exists(self.expertise_path):
            with open(self.expertise_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _load_pdf_texts(self):
        """
        加载PDF文本数据
        """
        if os.path.exists(self.pdf_texts_path):
            with open(self.pdf_texts_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def extract_expert_features(self, match_data, odds_data, team_state_data):
        """
        提取专家特征
        
        参数:
            match_data: 比赛基本数据
            odds_data: 赔率数据
            team_state_data: 球队状态数据
            
        返回:
            dict: 专家特征字典
        """
        expert_features = {
            'odds_match_degree': 0.0,  # 赔率与球队表现的匹配度
            'head_to_head_consistency': 0.0,  # 历史对阵与当前赔率的一致性
            'home_away_odds_factor': 0.0,  # 主客场因素与赔率的交互作用
            'recent_form_odds_correlation': 0.0,  # 近期状态与赔率变化的相关性
            'expert_confidence_score': 0.0  # 专家分析的信心评分
        }
        
        # 计算赔率与球队实际的匹配程度
        odds_match_degree = self._calculate_odds_match_degree(odds_data, team_state_data)
        expert_features['odds_match_degree'] = odds_match_degree
        
        # 计算历史对阵与当前赔率的一致性
        head_to_head_consistency = self._calculate_head_to_head_consistency(team_state_data)
        expert_features['head_to_head_consistency'] = head_to_head_consistency
        
        # 计算主客场因素与赔率的交互作用
        home_away_odds_factor = self._calculate_home_away_odds_factor(odds_data, team_state_data)
        expert_features['home_away_odds_factor'] = home_away_odds_factor
        
        # 计算近期状态与赔率变化的相关性
        recent_form_odds_correlation = self._calculate_recent_form_odds_correlation(odds_data, team_state_data)
        expert_features['recent_form_odds_correlation'] = recent_form_odds_correlation
        
        # 计算专家分析的信心评分
        expert_confidence_score = self._calculate_expert_confidence_score(odds_data, team_state_data)
        expert_features['expert_confidence_score'] = expert_confidence_score
        
        return expert_features
    
    def _get_representative_odds(self, odds_data):
        """
        获取代表性赔率（从多个博彩公司中选择一个）
        
        参数:
            odds_data: 赔率数据
            
        返回:
            tuple: (win_odds, draw_odds, lose_odds)
        """
        try:
            bookmakers = odds_data.get('bookmakers', {})
            
            # 优先选择ladbrokes
            if 'ladbrokes' in bookmakers:
                ladbrokes = bookmakers['ladbrokes']
                closing_odds = ladbrokes.get('closing_odds', {})
                return float(closing_odds.get('win', 1.0)), float(closing_odds.get('draw', 1.0)), float(closing_odds.get('lose', 1.0))
            
            # 否则选择第一个可用的博彩公司
            for bookmaker_name, bookmaker_data in bookmakers.items():
                closing_odds = bookmaker_data.get('closing_odds', {})
                if 'win' in closing_odds and 'draw' in closing_odds and 'lose' in closing_odds:
                    return float(closing_odds.get('win', 1.0)), float(closing_odds.get('draw', 1.0)), float(closing_odds.get('lose', 1.0))
            
            # 如果没有找到赔率，返回默认值
            return 1.0, 1.0, 1.0
        except Exception as e:
            return 1.0, 1.0, 1.0
    
    def _calculate_odds_match_degree(self, odds_data, team_state_data):
        """
        计算赔率与球队实际实力的匹配程度
        专家思路：赔率应该反映球队的真实实力
        
        参数:
            odds_data: 赔率数据
            team_state_data: 球队状态数据
            
        返回:
            float: 匹配度分数 (0-1)
        """
        try:
            # 获取主队和客队的胜率
            home_win_rate = team_state_data['home_team']['recent_form']['win_rate']
            away_win_rate = team_state_data['away_team']['recent_form']['win_rate']
            
            # 获取赔率
            win_odds, draw_odds, lose_odds = self._get_representative_odds(odds_data)
            
            # 计算预期赔率（基于胜率）
            # 修复：添加合理的最小胜率限制，避免预期赔率过大
            min_win_rate = 0.1
            home_win_rate = max(home_win_rate, min_win_rate)
            away_win_rate = max(away_win_rate, min_win_rate)
            
            expected_win_odds = 1 / home_win_rate
            expected_lose_odds = 1 / away_win_rate
            
            # 计算赔率差异（差异越小，匹配度越高）
            # 修复：添加归一化，避免差异过大
            max_odds_diff = 5.0  # 假设最大的合理赔率差异是5.0
            
            # 计算绝对差异
            win_odds_abs_diff = abs(win_odds - expected_win_odds)
            lose_odds_abs_diff = abs(lose_odds - expected_lose_odds)
            
            # 归一化差异
            win_odds_diff = min(win_odds_abs_diff / max_odds_diff, 1.0)
            lose_odds_diff = min(lose_odds_abs_diff / max_odds_diff, 1.0)
            
            # 计算匹配度分数
            match_degree = 1 - (win_odds_diff + lose_odds_diff) / 2
            match_degree = max(0, min(1, match_degree))  # 确保在0-1范围内
            
            return match_degree
        except Exception as e:
            return 0.5  # 默认值
    
    def _calculate_head_to_head_consistency(self, team_state_data):
        """
        计算历史对阵与当前赔率的一致性
        专家思路：历史对阵结果应该与当前赔率有一定的一致性
        
        参数:
        odds_data: 赔率数据
        team_state_data: 球队状态数据
        
        返回:
        float: 0-1范围的一致性分数，1表示完全一致
        """
        try:
            # 获取历史对阵数据
            head_to_head = team_state_data.get('head_to_head', {})
            
            # 获取近期战绩
            home_win_rate = team_state_data['home_team']['recent_form']['win_rate']
            away_win_rate = team_state_data['away_team']['recent_form']['win_rate']
            
            if head_to_head.get('head_to_head_matches', 0) == 0:
                return 0.5  # 没有历史对阵数据，返回默认值
            
            # 计算历史对阵胜率
            head_to_head_home_win_rate = head_to_head.get('home_win_rate', 0.5)
            head_to_head_away_win_rate = head_to_head.get('away_win_rate', 0.5)
            
            # 计算一致性（差异越小，一致性越高）
            home_consistency = 1 - abs(home_win_rate - head_to_head_home_win_rate)
            away_consistency = 1 - abs(away_win_rate - head_to_head_away_win_rate)
            
            # 计算平均一致性
            consistency = (home_consistency + away_consistency) / 2
            consistency = max(0, min(1, consistency))  # 确保在0-1范围内
            
            return consistency
        except Exception as e:
            return 0.5  # 默认值
    
    def _calculate_home_away_odds_factor(self, odds_data, team_state_data):
        """
        计算主客场因素与赔率的交互作用
        专家思路：主场优势应该反映在赔率上
        
        参数:
            odds_data: 赔率数据
            team_state_data: 球队状态数据
            
        返回:
            float: 交互作用分数 (0-1)
        """
        try:
            # 获取主客场胜率
            # 修复：使用正确的字段名season_win_rate，而不是不存在的home_win_rate和away_win_rate
            home_season_win_rate = team_state_data['home_team']['season_data'].get('season_win_rate', 0.5)
            away_season_win_rate = team_state_data['away_team']['season_data'].get('season_win_rate', 0.5)
            
            # 获取赔率
            win_odds, draw_odds, lose_odds = self._get_representative_odds(odds_data)
            
            # 计算主客场优势比例
            home_advantage = home_season_win_rate / (home_season_win_rate + away_season_win_rate) if (home_season_win_rate + away_season_win_rate) > 0 else 0.5
            
            # 计算赔率优势比例
            odds_advantage = lose_odds / (win_odds + lose_odds) if (win_odds + lose_odds) > 0 else 0.5
            
            # 计算交互作用（差异越小，交互作用越好）
            interaction = 1 - abs(home_advantage - odds_advantage)
            interaction = max(0, min(1, interaction))  # 确保在0-1范围内
            
            return interaction
        except Exception as e:
            return 0.5  # 默认值
    
    def _calculate_recent_form_odds_correlation(self, odds_data, team_state_data):
        """
        计算近期状态变化与赔率变化的相关性
        专家思路：近期表现好的球队应该得到更好的赔率
        
        参数:
            odds_data: 赔率数据
            team_state_data: 球队状态数据
            
        返回:
            float: 相关性分数 (0-1)
        """
        try:
            # 获取近期胜率变化
            home_recent_win_rate = team_state_data['home_team']['recent_form']['win_rate']
            away_recent_win_rate = team_state_data['away_team']['recent_form']['win_rate']
            
            # 获取赔率
            win_odds, draw_odds, lose_odds = self._get_representative_odds(odds_data)
            
            # 计算状态差异（主场相对于客场的优势）
            form_diff = home_recent_win_rate - away_recent_win_rate
            
            # 计算赔率差异（客胜赔率 - 主胜赔率，反映主场相对于客场的赔率优势）
            odds_diff = lose_odds - win_odds
            
            # 优化：使用更合理的相关性计算
            # 1. 将状态差异和赔率差异标准化到[-1, 1]范围
            normalized_form_diff = form_diff  # 已经在[-1, 1]范围
            
            # 赔率差异可能很大，需要标准化
            # 假设合理的赔率范围是1.0-10.0，所以最大可能的差异是9.0
            max_odds_diff = 9.0
            normalized_odds_diff = min(max(odds_diff / max_odds_diff, -1.0), 1.0)
            
            # 2. 计算相关性分数
            # 使用余弦相似度的概念，计算两个向量的相似度
            # 这里将两者视为一维向量，所以相似度就是符号是否相同乘以归一化的乘积
            if (form_diff > 0 and odds_diff > 0) or (form_diff < 0 and odds_diff < 0) or (form_diff == 0 and odds_diff == 0):
                # 符号相同，计算相关程度
                correlation = 1.0 - abs(abs(normalized_form_diff) - abs(normalized_odds_diff))
            else:
                # 符号不同，相关程度较低
                correlation = 0.0 + abs(normalized_form_diff - normalized_odds_diff) * 0.1
            
            correlation = max(0, min(1, correlation))  # 确保在0-1范围内
            
            return correlation
        except Exception as e:
            return 0.5  # 默认值
    
    def _extract_pdf_expert_factor(self, odds_data, team_state_data):
        """
        从PDF专家知识中提取额外的专家因素
        
        参数:
            odds_data: 赔率数据
            team_state_data: 球队状态数据
            
        返回:
            float: PDF专家因素 (0-1)
        """
        try:
            # 获取比赛相关数据
            home_win_rate = team_state_data['home_team']['recent_form']['win_rate']
            away_win_rate = team_state_data['away_team']['recent_form']['win_rate']
            win_odds, draw_odds, lose_odds = self._get_representative_odds(odds_data)
            
            # 专家通常只会分析那些有一定特征的比赛，而不是所有比赛
            # 我们需要识别出哪些比赛更适合应用专家知识
            is_expert_analysis_relevant = False
            
            # 检查比赛是否具有专家分析的典型特征
            # 特征1: 近期状态差异较大
            state_diff = abs(home_win_rate - away_win_rate)
            # 特征2: 赔率在合理范围内且有一定差异
            has_reasonable_odds = win_odds < 5.0 and lose_odds < 5.0
            # 特征3: 有一定的赔率变化
            has_odds_change = False
            if 'ladbrokes' in odds_data.get('bookmakers', {}):
                ladbrokes = odds_data['bookmakers']['ladbrokes']
                if 'opening_odds' in ladbrokes and 'closing_odds' in ladbrokes:
                    open_win_odds = float(ladbrokes['opening_odds'].get('win', win_odds))
                    has_odds_change = abs(open_win_odds - win_odds) > 0.1
            
            # 如果比赛具有以上任何一个特征，就认为适合应用专家知识
            if state_diff > 0.2 or has_reasonable_odds or has_odds_change:
                is_expert_analysis_relevant = True
            
            # 初始化专家因素
            pdf_expert_factor = 0.5  # 默认值
            
            # 只有当比赛适合专家分析时，才应用专家知识规则
            if is_expert_analysis_relevant:
                # 获取威廉希尔和立博的赔率数据
                bookmakers = odds_data.get('bookmakers', {})
                williamhill_data = bookmakers.get('williamhill', {})
                ladbrokes_data = bookmakers.get('ladbrokes', {})
                
                # 获取威廉希尔的平赔率
                williamhill_draw_odds = float(williamhill_data.get('closing_odds', {}).get('draw', draw_odds))
                
                # 获取立博的胜负赔率
                ladbrokes_win_odds = float(ladbrokes_data.get('closing_odds', {}).get('win', win_odds))
                ladbrokes_lose_odds = float(ladbrokes_data.get('closing_odds', {}).get('lose', lose_odds))
                
                # 规则1: 近期状态差异分析（专家非常重视球队近期状态）
                state_diff = home_win_rate - away_win_rate
                if abs(state_diff) > 0.3:  # 近期状态差异较大
                    if state_diff > 0.3 and win_odds < 2.0:  # 主队状态好且主胜赔率合理
                        pdf_expert_factor += 0.2
                    elif state_diff < -0.3 and lose_odds < 2.0:  # 客队状态好且客胜赔率合理
                        pdf_expert_factor += 0.2
                
                # 规则2: 赔率组合合理性分析（专家核心分析内容）
                # 检查赔率组合是否符合逻辑
                if win_odds < 1.5 and home_win_rate < 0.6:
                    # 主胜赔率过低但主队胜率不高，可能有陷阱
                    pdf_expert_factor -= 0.1
                elif lose_odds < 1.5 and away_win_rate < 0.6:
                    # 客胜赔率过低但客队胜率不高，可能有陷阱
                    pdf_expert_factor -= 0.1
                
                # 规则3: 主客场优势分析
                home_advantage = team_state_data['home_team']['recent_form']['home_win_rate']
                if home_advantage > 0.6 and win_odds < 2.5:
                    # 主队有明显主场优势且主胜赔率合理
                    pdf_expert_factor += 0.15
                
                # 规则4: 平局可能性分析 - 考虑威廉希尔的特点
                if 0.1 < abs(home_win_rate - away_win_rate) < 0.2 and 3.0 < draw_odds < 4.5:
                    # 双方胜率接近且平局赔率在合理范围
                    # 考虑威廉希尔的平赔率特点
                    if williamhill_data:
                        # 如果威廉希尔的平赔率低于立博或平均水平，增加平局可能性评分
                        if williamhill_draw_odds < draw_odds:
                            pdf_expert_factor += 0.2  # 威廉希尔平赔率偏低，平局可能性增加
                        else:
                            pdf_expert_factor += 0.1  # 正常平局可能性评分
                    else:
                        pdf_expert_factor += 0.15  # 没有威廉希尔数据时的默认评分
                
                # 规则5: 胜负可能性分析 - 考虑立博的特点
                if abs(state_diff) > 0.3:
                    if state_diff > 0.3 and ladbrokes_data:
                        # 主队状态好，且立blogger胜赔率合理
                        if ladbrokes_win_odds < 2.5:
                            pdf_expert_factor += 0.15  # 立blogger胜赔率合理，增加主胜可能性
                    elif state_diff < -0.3 and ladbrokes_data:
                        # 客队状态好，且立博客胜赔率合理
                        if ladbrokes_lose_odds < 2.5:
                            pdf_expert_factor += 0.15  # 立博客胜赔率合理，增加客胜可能性
                
                # 规则6: 威廉希尔平赔率分析
                if williamhill_data:
                    # 威廉希尔的玩家喜欢平局，平赔率压力大
                    # 如果威廉希尔的平赔率显著低于其他公司，可能预示平局
                    if williamhill_draw_odds < draw_odds - 0.3:
                        pdf_expert_factor += 0.15  # 威廉希尔平赔率明显偏低，增加平局预期
                    # 如果威廉希尔的平赔率显著高于其他公司，可能预示非平局
                    elif williamhill_draw_odds > draw_odds + 0.3:
                        pdf_expert_factor -= 0.1  # 威廉希尔平赔率明显偏高，降低平局预期
                
                # 规则7: 赔率变化趋势分析
                # 检查赔率是否有明显变化（如果有变化数据）
                if 'ladbrokes' in odds_data.get('bookmakers', {}):
                    ladbrokes = odds_data['bookmakers']['ladbrokes']
                    if 'opening_odds' in ladbrokes and 'closing_odds' in ladbrokes:
                        open_win_odds = float(ladbrokes['opening_odds'].get('win', win_odds))
                        if abs(open_win_odds - win_odds) > 0.2:  # 赔率变化较大
                            if open_win_odds > win_odds and home_win_rate > 0.5:
                                # 主胜赔率降低且主队胜率高，看好主胜
                                pdf_expert_factor += 0.1
                            elif open_win_odds < win_odds and home_win_rate < 0.4:
                                # 主胜赔率升高且主队胜率低，不看好主胜
                                pdf_expert_factor -= 0.1
            else:
                # 对于不适合专家分析的比赛，减少专家知识的影响
                pdf_expert_factor = 0.5  # 保持默认值，不做调整
            
            # 确保在0-1范围内
            pdf_expert_factor = max(0, min(1, pdf_expert_factor))
            
            return pdf_expert_factor
        except Exception as e:
            return 0.5  # 默认值

    def _calculate_expert_confidence_score(self, odds_data, team_state_data):
        """
        计算专家分析的信心评分
        
        参数:
            odds_data: 赔率数据
            team_state_data: 球队状态数据
            
        返回:
            float: 信心评分 (0-1)
        """
        try:
            # 计算各项指标
            odds_match_degree = self._calculate_odds_match_degree(odds_data, team_state_data)
            head_to_head_consistency = self._calculate_head_to_head_consistency(team_state_data)
            home_away_odds_factor = self._calculate_home_away_odds_factor(odds_data, team_state_data)
            recent_form_odds_correlation = self._calculate_recent_form_odds_correlation(odds_data, team_state_data)
            
            # 从PDF专家知识中提取的额外因素
            pdf_expert_factor = self._extract_pdf_expert_factor(odds_data, team_state_data)
            
            # 专家通常只会分析那些有一定特征的比赛，我们需要识别这些比赛
            is_expert_analysis_relevant = False
            
            # 检查比赛是否具有专家分析的典型特征
            home_win_rate = team_state_data['home_team']['recent_form']['win_rate']
            away_win_rate = team_state_data['away_team']['recent_form']['win_rate']
            state_diff = abs(home_win_rate - away_win_rate)
            win_odds, draw_odds, lose_odds = self._get_representative_odds(odds_data)
            
            # 特征1: 近期状态差异较大
            # 特征2: 赔率在合理范围内且有一定差异
            has_reasonable_odds = win_odds < 5.0 and lose_odds < 5.0
            
            if state_diff > 0.2 or has_reasonable_odds:
                is_expert_analysis_relevant = True
            
            # 根据比赛是否适合专家分析，调整各个因素的权重
            if is_expert_analysis_relevant:
                # 对于适合专家分析的比赛，增加PDF专家因素的权重
                confidence_score = (
                    0.20 * odds_match_degree +
                    0.20 * head_to_head_consistency +
                    0.20 * home_away_odds_factor +
                    0.20 * recent_form_odds_correlation +
                    0.20 * pdf_expert_factor  # 增加PDF专家因素的权重到20%
                )
            else:
                # 对于不适合专家分析的比赛，减少PDF专家因素的权重
                confidence_score = (
                    0.25 * odds_match_degree +
                    0.25 * head_to_head_consistency +
                    0.25 * home_away_odds_factor +
                    0.25 * recent_form_odds_correlation +
                    0.00 * pdf_expert_factor  # 完全不使用PDF专家因素
                )
            
            confidence_score = max(0, min(1, confidence_score))  # 确保在0-1范围内
            
            return confidence_score
        except Exception as e:
            return 0.5  # 默认值

    def save_expert_features(self, season, expert_features_data):
        """
        保存专家特征数据
        
        参数:
            season: 赛季
            expert_features_data: 专家特征数据
        """
        output_path = f"/Users/Williamhiler/Documents/my-project/train/train-data/expert/{season}_expert_features.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(expert_features_data, f, ensure_ascii=False, indent=4)
        
        print(f"赛季 {season} 的专家特征已保存到 {output_path}")

if __name__ == "__main__":
    # 测试专家特征提取器
    extractor = ExpertFeatureExtractor()
    
    # 示例数据
    match_data = {
        'matchId': '123456',
        'homeTeamId': 62,
        'awayTeamId': 49
    }
    
    odds_data = {
        'bookmakers': {
            'ladbrokes': {
                'closing_odds': {
                    'win': '1.8',
                    'draw': '3.2',
                    'lose': '4.0'
                }
            }
        }
    }
    
    team_state_data = {
        'home_team': {
            'recent_form': {
                'win_rate': 0.6
            },
            'season_data': {
                'home_win_rate': 0.7
            }
        },
        'away_team': {
            'recent_form': {
                'win_rate': 0.4
            },
            'season_data': {
                'away_win_rate': 0.3
            }
        },
        'head_to_head': {
            'head_to_head_matches': 5,
            'home_win_rate': 0.4,
            'away_win_rate': 0.6
        }
    }
    
    expert_features = extractor.extract_expert_features(match_data, odds_data, team_state_data)
    print("专家特征示例:")
    print(json.dumps(expert_features, indent=4))