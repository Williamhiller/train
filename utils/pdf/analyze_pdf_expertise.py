#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析PDF文本内容，提取专家的分析思路和方法
"""

import json
import re

def analyze_pdf_expertise(json_path):
    """
    分析PDF文本内容，提取专家的分析思路和方法
    
    参数:
        json_path: JSON文件路径
        
    返回:
        dict: 提取的专家分析思路
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        pdf_texts = json.load(f)
    
    expertise = {
        'analyzed_files': [],
        'key_factors': [],
        'analysis_methods': [],
        'odds_analysis': [],
        'team_performance_analysis': []
    }
    
    for pdf_file, text in pdf_texts.items():
        expertise['analyzed_files'].append(pdf_file)
        
        # 提取关键因素 - 可以根据实际内容调整正则表达式
        key_factors = re.findall(r'关键因素|核心指标|重要考虑|重点关注', text)
        if key_factors:
            expertise['key_factors'].extend(key_factors)
        
        # 提取分析方法 - 可以根据实际内容调整正则表达式
        analysis_methods = re.findall(r'分析方法|分析思路|分析步骤|预测模型', text)
        if analysis_methods:
            expertise['analysis_methods'].extend(analysis_methods)
        
        # 提取赔率分析相关内容 - 可以根据实际内容调整正则表达式
        odds_analysis = re.findall(r'赔率分析|欧赔分析|盘口分析|水位变化', text)
        if odds_analysis:
            expertise['odds_analysis'].extend(odds_analysis)
        
        # 提取球队表现分析相关内容 - 可以根据实际内容调整正则表达式
        team_performance = re.findall(r'球队表现|近期战绩|历史对阵|状态分析', text)
        if team_performance:
            expertise['team_performance_analysis'].extend(team_performance)
    
    # 去重
    expertise['key_factors'] = list(set(expertise['key_factors']))
    expertise['analysis_methods'] = list(set(expertise['analysis_methods']))
    expertise['odds_analysis'] = list(set(expertise['odds_analysis']))
    expertise['team_performance_analysis'] = list(set(expertise['team_performance_analysis']))
    
    return expertise

def main():
    JSON_PATH = "/Users/Williamhiler/Documents/my-project/train/train-data/expert/pdf_texts.json"
    expertise = analyze_pdf_expertise(JSON_PATH)
    
    print("=== 专家分析思路提取结果 ===")
    print(f"分析的文件: {expertise['analyzed_files']}")
    print(f"\n关键因素: {expertise['key_factors']}")
    print(f"\n分析方法: {expertise['analysis_methods']}")
    print(f"\n赔率分析相关: {expertise['odds_analysis']}")
    print(f"\n球队表现分析相关: {expertise['team_performance_analysis']}")
    
    # 保存分析结果
    with open('/Users/Williamhiler/Documents/my-project/train/train-data/expert/expertise_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(expertise, f, ensure_ascii=False, indent=4)
    
    print("\n分析结果已保存到 expertise_analysis.json")

if __name__ == "__main__":
    main()
