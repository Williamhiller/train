#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取专家分析思路中的具体特征，将其转化为可用于机器学习的特征
"""

import json
import re
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter

class ExpertFeatureExtractor:
    """
    专家特征提取器，用于从PDF文本中提取结构化的专家分析特征
    """
    
    def __init__(self, use_cleaned_text: bool = True):
        """
        初始化提取器
        
        参数:
            use_cleaned_text: 是否使用清洗后的文本
        """
        self.use_cleaned_text = use_cleaned_text
        
        # 赔率相关特征模式
        self.odds_patterns = [
            (r'欧赔(\d+\.\d+)/(\d+\.\d+)/(\d+\.\d+)', 'europe_odds'),
            (r'盘口(\d+\.\d+)', 'handicap'),
            (r'水位(\d+\.\d+)', 'water_level'),
            (r'赔率变化', 'odds_change'),
            (r'初盘', 'initial_odds'),
            (r'终盘', 'final_odds'),
            (r'胜赔(?:(\d+\.\d+))?', 'home_odds'),
            (r'平赔(?:(\d+\.\d+))?', 'draw_odds'),
            (r'负赔(?:(\d+\.\d+))?', 'away_odds')
        ]
        
        # 球队表现相关特征模式
        self.team_performance_patterns = [
            (r'(主队|客队)?近(\d+)场(胜(\d+)平(\d+)负(\d+))?', 'recent_form'),
            (r'(主队|客队)?(胜率|平率|负率)(\d+\.\d+)%', 'win_rate'),
            (r'(主队|客队)?积分(\d+)', 'points'),
            (r'(主队|客队)?排名(\d+)', 'ranking'),
            (r'(主队|客队)?(进球|失球)(\d+)', 'goals'),
            (r'(主队|客队)主场', 'home_performance'),
            (r'(主队|客队)客场', 'away_performance')
        ]
        
        # 历史对阵相关特征模式
        self.head_to_head_patterns = [
            (r'历史交锋', 'history_battle'),
            (r'近(\d+)次对阵', 'recent_battles'),
            (r'交锋记录', 'battle_record'),
            (r'(主队|客队)对战(\d+)胜(\d+)平(\d+)负', 'battle_results')
        ]
        
        # 分析规则模式
        self.rules_patterns = [
            (r'当(.*?)时(.*?)(?:\.|，|。|\n)', 'when_rule'),
            (r'如果(.*?)则(.*?)(?:\.|，|。|\n)', 'if_rule'),
            (r'(.*?)情况下(.*?)(?:\.|，|。|\n)', 'condition_rule'),
            (r'应(.*?)选择(.*?)(?:\.|，|。|\n)', 'should_choose_rule'),
            (r'(.*?)赔率(.*?)(?:合理|不合理|倾向于)(.*?)(?:\.|，|。|\n)', 'odds_analysis_rule'),
            (r'(主队|客队)(.*?)优势(.*?)(?:\.|，|。|\n)', 'team_advantage_rule')
        ]
    
    def load_pdf_texts(self, json_path: str) -> Dict[str, str]:
        """
        加载PDF文本数据
        
        参数:
            json_path: JSON文件路径
            
        返回:
            Dict[str, str]: PDF文本字典
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_features(self, pdf_texts: Dict[str, str]) -> Dict[str, Any]:
        """
        提取所有专家特征
        
        参数:
            pdf_texts: PDF文本字典
            
        返回:
            Dict[str, Any]: 结构化的专家特征
        """
        # 初始化特征存储结构
        expert_features = {
            'odds_features': {
                'patterns': [],
                'values': [],
                'weighted_features': defaultdict(float)
            },
            'team_performance_features': {
                'patterns': [],
                'values': [],
                'weighted_features': defaultdict(float)
            },
            'head_to_head_features': {
                'patterns': [],
                'values': [],
                'weighted_features': defaultdict(float)
            },
            'analysis_rules': {
                'patterns': [],
                'complete_rules': [],
                'weighted_features': defaultdict(float)
            }
        }
        
        # 统计各特征出现频率
        feature_counters = {
            'odds_features': Counter(),
            'team_performance_features': Counter(),
            'head_to_head_features': Counter(),
            'analysis_rules': Counter()
        }
        
        for pdf_file, text in pdf_texts.items():
            # 提取赔率相关特征
            for pattern, feature_type in self.odds_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    for match in matches:
                        if isinstance(match, tuple):
                            # 过滤空匹配
                            filtered_match = [m for m in match if m]
                            if filtered_match:
                                feature_str = f"{feature_type}:{'-'.join(filtered_match)}"
                                expert_features['odds_features']['values'].append(feature_str)
                                feature_counters['odds_features'][feature_str] += 1
                        else:
                            feature_str = f"{feature_type}:{match}"
                            expert_features['odds_features']['values'].append(feature_str)
                            feature_counters['odds_features'][feature_str] += 1
            
            # 提取球队表现相关特征
            for pattern, feature_type in self.team_performance_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    for match in matches:
                        if isinstance(match, tuple):
                            filtered_match = [m for m in match if m]
                            if filtered_match:
                                feature_str = f"{feature_type}:{'-'.join(filtered_match)}"
                                expert_features['team_performance_features']['values'].append(feature_str)
                                feature_counters['team_performance_features'][feature_str] += 1
                        else:
                            feature_str = f"{feature_type}:{match}"
                            expert_features['team_performance_features']['values'].append(feature_str)
                            feature_counters['team_performance_features'][feature_str] += 1
            
            # 提取历史对阵相关特征
            for pattern, feature_type in self.head_to_head_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    for match in matches:
                        if isinstance(match, tuple):
                            filtered_match = [m for m in match if m]
                            if filtered_match:
                                feature_str = f"{feature_type}:{'-'.join(filtered_match)}"
                                expert_features['head_to_head_features']['values'].append(feature_str)
                                feature_counters['head_to_head_features'][feature_str] += 1
                        else:
                            feature_str = f"{feature_type}:{match}"
                            expert_features['head_to_head_features']['values'].append(feature_str)
                            feature_counters['head_to_head_features'][feature_str] += 1
            
            # 提取分析规则
            for pattern, rule_type in self.rules_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    for match in matches:
                        if isinstance(match, tuple):
                            filtered_match = [m.strip() for m in match if m.strip()]
                            if filtered_match and len(filtered_match) >= 2:
                                rule_str = f"{rule_type}:条件-{filtered_match[0]} 结果-{filtered_match[1]}"
                                expert_features['analysis_rules']['complete_rules'].append(rule_str)
                                feature_counters['analysis_rules'][rule_str] += 1
                        else:
                            match_str = match.strip()
                            if match_str:
                                rule_str = f"{rule_type}:{match_str}"
                                expert_features['analysis_rules']['complete_rules'].append(rule_str)
                                feature_counters['analysis_rules'][rule_str] += 1
        
        # 计算特征权重（基于出现频率）
        for feature_type, counter in feature_counters.items():
            if counter:
                total_count = sum(counter.values())
                for feature, count in counter.items():
                    weight = count / total_count
                    expert_features[feature_type]['weighted_features'][feature] = weight
        
        # 去重并保留唯一特征
        for feature_type in expert_features:
            if 'values' in expert_features[feature_type]:
                expert_features[feature_type]['values'] = list(set(expert_features[feature_type]['values']))
            if 'complete_rules' in expert_features[feature_type]:
                expert_features[feature_type]['complete_rules'] = list(set(expert_features[feature_type]['complete_rules']))
        
        return expert_features
    
    def generate_feature_suggestions(self, expert_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        根据专家特征生成机器学习特征建议
        
        参数:
            expert_features: 提取的专家特征
            
        返回:
            List[Dict[str, Any]]: 特征建议
        """
        suggestions = []
        
        # 赔率相关特征建议
        suggestions.append({
            'category': '赔率特征',
            'features': [
                '胜赔、平赔、负赔的初盘值和终盘值',
                '赔率变化率（初盘到终盘的变化百分比）',
                '赔率组合的合理性（如胜平负赔率之和是否在合理范围内）',
                '主客场赔率差异',
                '水位变化率（初盘到终盘的水位变化）',
                '欧赔组合模式识别（如常见的赔率组合）'
            ]
        })
        
        # 球队表现相关特征建议
        suggestions.append({
            'category': '球队表现特征',
            'features': [
                '近5场、近10场比赛的胜率、平率、负率',
                '近5场、近10场比赛的进球数和失球数',
                '主场和客场的胜率、进球数、失球数',
                '本赛季的积分、排名、进球数、失球数',
                '近期状态的变化趋势（如最近5场比赛的胜率变化）',
                '主客场表现差异'
            ]
        })
        
        # 历史对阵相关特征建议
        suggestions.append({
            'category': '历史对阵特征',
            'features': [
                '近5次、近10次对阵的胜负平记录',
                '近5次、近10次对阵的进球数和失球数',
                '主场和客场对阵的历史战绩',
                '上次对阵的结果和赔率',
                '历史对阵中的胜率、平率、负率'
            ]
        })
        
        # 组合特征建议
        suggestions.append({
            'category': '组合特征',
            'features': [
                '球队表现与赔率的匹配度（如实力强的球队是否有合理的低赔率）',
                '历史对阵结果与当前赔率的一致性',
                '主客场因素与赔率的交互作用',
                '近期状态与赔率变化的关系',
                '专家规则匹配度（当前比赛与专家分析规则的匹配程度）'
            ]
        })
        
        return suggestions
    
    def save_analysis_results(self, expert_features: Dict[str, Any], suggestions: List[Dict[str, Any]], 
                             output_file: str = '/Users/Williamhiler/Documents/my-project/train/utils/pdf/expert_features_analysis.json') -> None:
        """
        保存分析结果到JSON文件
        
        参数:
            expert_features: 提取的专家特征
            suggestions: 特征建议
            output_file: 输出文件路径
        """
        result = {
            'expert_features': expert_features,
            'feature_suggestions': suggestions
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        
        print(f"分析结果已保存到 {output_file}")
    
    def run_analysis(self, input_file: str, output_file: str) -> None:
        """
        运行完整的分析流程
        
        参数:
            input_file: 输入PDF文本文件路径
            output_file: 输出分析结果文件路径
        """
        # 加载文本
        pdf_texts = self.load_pdf_texts(input_file)
        
        # 提取特征
        print("正在提取专家特征...")
        expert_features = self.extract_features(pdf_texts)
        
        # 生成特征建议
        print("正在生成特征建议...")
        suggestions = self.generate_feature_suggestions(expert_features)
        
        # 保存结果
        self.save_analysis_results(expert_features, suggestions, output_file)
        
        # 打印摘要
        self.print_analysis_summary(expert_features, suggestions)
    
    def print_analysis_summary(self, expert_features: Dict[str, Any], suggestions: List[Dict[str, Any]]) -> None:
        """
        打印分析结果摘要
        
        参数:
            expert_features: 提取的专家特征
            suggestions: 特征建议
        """
        print("\n=== 专家分析特征提取结果摘要 ===")
        
        # 赔率特征
        odds_count = len(expert_features['odds_features']['values'])
        odds_weighted_count = len(expert_features['odds_features']['weighted_features'])
        print(f"赔率相关特征: {odds_count}个唯一特征, {odds_weighted_count}个加权特征")
        
        # 球队表现特征
        team_count = len(expert_features['team_performance_features']['values'])
        team_weighted_count = len(expert_features['team_performance_features']['weighted_features'])
        print(f"球队表现相关特征: {team_count}个唯一特征, {team_weighted_count}个加权特征")
        
        # 历史对阵特征
        head_count = len(expert_features['head_to_head_features']['values'])
        head_weighted_count = len(expert_features['head_to_head_features']['weighted_features'])
        print(f"历史对阵相关特征: {head_count}个唯一特征, {head_weighted_count}个加权特征")
        
        # 分析规则
        rules_count = len(expert_features['analysis_rules']['complete_rules'])
        rules_weighted_count = len(expert_features['analysis_rules']['weighted_features'])
        print(f"完整分析规则: {rules_count}个, {rules_weighted_count}个加权规则")
        
        # 打印前几个规则示例
        print("\n=== 分析规则示例 ===")
        for rule in list(expert_features['analysis_rules']['complete_rules'])[:5]:
            print(f"  - {rule}")
        
        print("\n=== 机器学习特征建议 ===")
        for suggestion in suggestions:
            print(f"\n{suggestion['category']}:")
            for feature in suggestion['features'][:3]:  # 只显示前3个
                print(f"  - {feature}")
            if len(suggestion['features']) > 3:
                print(f"  ... 等{len(suggestion['features'])}个特征")

def main():
    """
    主函数
    """
    # 设置输入输出文件路径
    if True:  # 使用清洗后的文本
        INPUT_FILE = "/Users/Williamhiler/Documents/my-project/train/train-data/expert/cleaned_pdf_texts.json"
    else:
        INPUT_FILE = "/Users/Williamhiler/Documents/my-project/train/train-data/expert/pdf_texts.json"
    
    OUTPUT_FILE = "/Users/Williamhiler/Documents/my-project/train/train-data/expert/expert_features_analysis.json"
    
    # 创建提取器实例
    extractor = ExpertFeatureExtractor(use_cleaned_text=True)
    
    # 运行分析
    extractor.run_analysis(INPUT_FILE, OUTPUT_FILE)

if __name__ == "__main__":
    main()
