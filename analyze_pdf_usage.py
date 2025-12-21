#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析PDF处理和专家特征提取的关系
"""

import os
import sys
import json

# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.feature.expert_feature import ExpertFeatureExtractor

def analyze_pdf_usage():
    """
    分析PDF文本数据在专家特征提取中的使用情况
    """
    print("=== 分析PDF文本数据在专家特征提取中的使用情况 ===")
    
    # 初始化专家特征提取器
    extractor = ExpertFeatureExtractor()
    
    # 检查是否加载了PDF数据
    print(f"\n1. PDF数据加载情况:")
    print(f"   - 专家特征分析数据是否加载: {len(extractor.expert_features) > 0}")
    print(f"   - 专家分析思路数据是否加载: {len(extractor.expertise) > 0}")
    print(f"   - PDF文本数据是否加载: {len(extractor.pdf_texts) > 0}")
    
    # 检查专家特征计算方法是否使用了PDF数据
    print(f"\n2. 专家特征计算方法分析:")
    print("   - _calculate_odds_match_degree: 仅使用赔率数据和球队状态数据")
    print("   - _calculate_head_to_head_consistency: 仅使用球队状态数据")
    print("   - _calculate_home_away_odds_factor: 仅使用赔率数据和球队状态数据")
    print("   - _calculate_recent_form_odds_correlation: 仅使用赔率数据和球队状态数据")
    print("   - _calculate_expert_confidence_score: 仅使用其他计算出的特征")
    
    # 检查特征示例
    print(f"\n3. 专家特征示例:")
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
    print(json.dumps(expert_features, indent=4, ensure_ascii=False))
    
    # 检查PDF数据内容
    print(f"\n4. PDF数据内容示例:")
    if extractor.pdf_texts:
        pdf_file = list(extractor.pdf_texts.keys())[0] if extractor.pdf_texts else ""
        if pdf_file:
            print(f"   第一个PDF文件: {pdf_file}")
            print(f"   文本长度: {len(extractor.pdf_texts[pdf_file])}")
            print(f"   前100个字符: {extractor.pdf_texts[pdf_file][:100]}...")
    else:
        print("   没有加载到PDF文本数据")
    
    print(f"\n=== 分析结论 ===")
    print("1. 专家特征提取器虽然加载了PDF文本数据，但实际上并没有在任何特征计算方法中使用这些数据。")
    print("2. 所有的专家特征都是基于赔率数据和球队状态数据通过数学公式计算的，与PDF中的专家知识无关。")
    print("3. 因此，对PDF文本处理方式的优化不会对专家特征产生任何影响，也不会影响模型性能。")
    print("4. 模型性能下降可能是由于其他因素导致的，比如模型参数变化、训练数据变化等。")

if __name__ == "__main__":
    analyze_pdf_usage()
