#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用LLM专家分析结果更新专家特征
将LLM处理后的专家知识融入到传统专家特征中
"""

import os
import json
import pandas as pd
from typing import Dict, List

def update_expert_features_with_llm():
    """
    使用LLM专家分析结果更新所有赛季的专家特征
    """
    print("=== 使用LLM专家分析更新专家特征 ===")
    
    # 配置路径
    data_root = "/Users/Williamhiler/Documents/my-project/train/train-data"
    llm_analysis_path = os.path.join(data_root, "expert/llm_analysis/combined_llm_analysis.json")
    expert_features_dir = os.path.join(data_root, "expert")
    
    # 加载LLM分析结果
    print(f"加载LLM分析结果: {llm_analysis_path}")
    with open(llm_analysis_path, 'r', encoding='utf-8') as f:
        llm_analysis = json.load(f)
    
    # 计算LLM专家分析的统计特征
    llm_stats = calculate_llm_statistics(llm_analysis)
    
    # 获取所有赛季
    seasons = [f"{year}-{year+1}" for year in range(2015, 2025)]
    
    # 更新每个赛季的专家特征
    for season in seasons:
        expert_features_file = os.path.join(expert_features_dir, f"{season}_expert_features.json")
        if not os.path.exists(expert_features_file):
            print(f"跳过赛季 {season}: 专家特征文件不存在")
            continue
        
        print(f"\n更新赛季 {season} 的专家特征")
        
        # 加载原始专家特征
        with open(expert_features_file, 'r', encoding='utf-8') as f:
            original_features = json.load(f)
        
        # 使用LLM分析结果增强专家特征
        enhanced_features = enhance_expert_features(original_features, llm_stats)
        
        # 保存增强后的专家特征
        enhanced_file = os.path.join(expert_features_dir, f"{season}_expert_features_llm.json")
        with open(enhanced_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_features, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 保存增强后的专家特征: {enhanced_file}")
    
    print("\n=== LLM专家特征更新完成 ===")

def calculate_llm_statistics(llm_analysis: Dict) -> Dict:
    """
    计算LLM分析结果的统计特征
    """
    print("计算LLM分析统计特征...")
    
    # 计算预测分布的比例
    total_predictions = sum(llm_analysis["prediction_distribution"].values())
    prediction_ratios = {
        f"llm_prediction_{key}_ratio": value / total_predictions
        for key, value in llm_analysis["prediction_distribution"].items()
    }
    
    # 计算情感分布的比例
    total_sentiments = sum(llm_analysis["sentiment_distribution"].values())
    sentiment_ratios = {
        f"llm_sentiment_{key}_ratio": value / total_sentiments
        for key, value in llm_analysis["sentiment_distribution"].items()
    }
    
    # 计算文件级别的统计特征
    file_analysis = llm_analysis["files_analysis"]
    avg_file_confidence = sum(
        file_data["average_confidence"] for file_data in file_analysis.values()
    ) / len(file_analysis)
    
    # 合并所有统计特征
    llm_stats = {
        "llm_average_confidence": llm_analysis["average_confidence"],
        "llm_avg_file_confidence": avg_file_confidence,
        "llm_total_chapters": llm_analysis["total_chapters"],
        **prediction_ratios,
        **sentiment_ratios
    }
    
    print(f"✓ LLM统计特征: {list(llm_stats.keys())}")
    return llm_stats

def enhance_expert_features(original_features: Dict, llm_stats: Dict) -> Dict:
    """
    使用LLM统计特征增强原始专家特征
    """
    enhanced_features = {}
    
    for match_id, features in original_features.items():
        # 创建增强后的特征字典
        enhanced = features.copy()
        
        # 融合LLM统计特征
        for key, value in llm_stats.items():
            # 将LLM特征与原始特征结合，创建新的特征
            if key == "llm_average_confidence":
                # 增强专家信心评分
                enhanced["expert_confidence_score"] = (
                    enhanced["expert_confidence_score"] * 0.7 + value * 0.3
                )
            elif "llm_prediction" in key:
                # 添加LLM预测分布特征
                enhanced[key] = value
            elif "llm_sentiment" in key:
                # 添加LLM情感分布特征
                enhanced[key] = value
        
        # 添加LLM增强标记
        enhanced["llm_enhanced"] = True
        
        enhanced_features[match_id] = enhanced
    
    return enhanced_features

# 直接修改DataLoader文件，避免字符串嵌套问题
def update_data_loader():
    """
    更新DataLoader，使其能够使用LLM增强后的专家特征
    """
    print("\n更新DataLoader以支持LLM专家特征...")
    
    data_loader_path = "/Users/Williamhiler/Documents/my-project/train/trainers/data_loader.py"
    
    # 读取当前DataLoader内容
    with open(data_loader_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 替换load_expert_features方法
    new_lines = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "def load_expert_features(self, season):":
            # 找到方法定义，替换整个方法
            new_lines.append("    def load_expert_features(self, season, use_llm=False):\n")
            new_lines.append("        \"\"\"加载专家特征\"\"\"\n")
            new_lines.append("        if use_llm:\n")
            new_lines.append("            file_path = os.path.join(self.data_root, 'expert', f'{season}_expert_features_llm.json')\n")
            new_lines.append("        else:\n")
            new_lines.append("            file_path = os.path.join(self.data_root, 'expert', f'{season}_expert_features.json')\n")
            new_lines.append("        \n")
            new_lines.append("        if not os.path.exists(file_path):\n")
            new_lines.append("            # 如果LLM增强特征文件不存在，回退到原始特征\n")
            new_lines.append("            if use_llm:\n")
            new_lines.append("                print(f\"LLM增强专家特征文件不存在，回退到原始特征: {file_path}\")\n")
            new_lines.append("                file_path = os.path.join(self.data_root, 'expert', f'{season}_expert_features.json')\n")
            new_lines.append("            \n")
            new_lines.append("        if not os.path.exists(file_path):\n")
            new_lines.append("            raise FileNotFoundError(f\"专家特征文件不存在: {file_path}\")\n")
            new_lines.append("        \n")
            new_lines.append("        with open(file_path, 'r', encoding='utf-8') as f:\n")
            new_lines.append("            data = json.load(f)\n")
            new_lines.append("        \n")
            new_lines.append("        matches = []\n")
            new_lines.append("        for match_id, features in data.items():\n")
            new_lines.append("            match_data = {\n")
            new_lines.append("                'match_id': match_id,\n")
            new_lines.append("                'season': season,\n")
            new_lines.append("                'odds_match_degree': features['odds_match_degree'],\n")
            new_lines.append("                'head_to_head_consistency': features['head_to_head_consistency'],\n")
            new_lines.append("                'home_away_odds_factor': features['home_away_odds_factor'],\n")
            new_lines.append("                'recent_form_odds_correlation': features['recent_form_odds_correlation'],\n")
            new_lines.append("                'expert_confidence_score': features['expert_confidence_score']\n")
            new_lines.append("            }\n")
            new_lines.append("            \n")
            new_lines.append("            # 添加LLM增强特征（如果存在）\n")
            new_lines.append("            if 'llm_enhanced' in features and features['llm_enhanced']:\n")
            new_lines.append("                for key, value in features.items():\n")
            new_lines.append("                    if key.startswith('llm_'):\n")
            new_lines.append("                        match_data[key] = value\n")
            new_lines.append("            \n")
            new_lines.append("            matches.append(match_data)\n")
            new_lines.append("        \n")
            new_lines.append("        return pd.DataFrame(matches)\n")
            # 跳过原方法的所有行
            while i < len(lines) and not (lines[i].strip() and not lines[i].startswith(' ') and lines[i].strip() != '"""' and lines[i].strip() != "'""'"):
                i += 1
        elif lines[i].strip() == "def load_combined_features(self, season, include_team_state=False, include_expert=False):":
            # 找到方法定义，替换整个方法
            new_lines.append("    def load_combined_features(self, season, include_team_state=False, include_expert=False, use_llm=False):\n")
            new_lines.append("        \"\"\"加载组合特征\"\"\"\n")
            new_lines.append("        # 加载赔率特征（基础特征）\n")
            new_lines.append("        odds_df = self.load_odds_features(season)\n")
            new_lines.append("        \n")
            new_lines.append("        # 合并球队状态特征\n")
            new_lines.append("        if include_team_state:\n")
            new_lines.append("            team_state_df = self.load_team_state_features(season)\n")
            new_lines.append("            # 将match_id转换为字符串进行合并\n")
            new_lines.append("            odds_df['match_id'] = odds_df['match_id'].astype(str)\n")
            new_lines.append("            team_state_df['match_id'] = team_state_df['match_id'].astype(str)\n")
            new_lines.append("            odds_df = odds_df.merge(team_state_df, on=['match_id', 'season'], how='inner')\n")
            new_lines.append("        \n")
            new_lines.append("        # 合并专家特征\n")
            new_lines.append("        if include_expert:\n")
            new_lines.append("            expert_df = self.load_expert_features(season, use_llm)\n")
            new_lines.append("            # 将match_id转换为字符串进行合并\n")
            new_lines.append("            odds_df['match_id'] = odds_df['match_id'].astype(str)\n")
            new_lines.append("            expert_df['match_id'] = expert_df['match_id'].astype(str)\n")
            new_lines.append("            odds_df = odds_df.merge(expert_df, on=['match_id', 'season'], how='inner')\n")
            new_lines.append("        \n")
            new_lines.append("        return odds_df\n")
            # 跳过原方法的所有行
            while i < len(lines) and not (lines[i].strip() and not lines[i].startswith(' ') and lines[i].strip() != '"""' and lines[i].strip() != "'""'"):
                i += 1
        elif lines[i].strip() == "def prepare_training_data(self, seasons, include_team_state=False, include_expert=False):":
            # 找到方法定义，替换整个方法
            new_lines.append("    def prepare_training_data(self, seasons, include_team_state=False, include_expert=False, use_llm=False):\n")
            new_lines.append("        \"\"\"准备训练数据\"\"\"\n")
            new_lines.append("        all_data = []\n")
            new_lines.append("        \n")
            new_lines.append("        for season in seasons:\n")
            new_lines.append("            try:\n")
            new_lines.append("                season_data = self.load_combined_features(season, include_team_state, include_expert, use_llm)\n")
            new_lines.append("                all_data.append(season_data)\n")
            new_lines.append("            except FileNotFoundError as e:\n")
            new_lines.append("                print(f\"跳过赛季 {season}: {e}\")\n")
            new_lines.append("        \n")
            new_lines.append("        if not all_data:\n")
            new_lines.append("            raise ValueError(\"没有找到任何训练数据\")\n")
            new_lines.append("        \n")
            new_lines.append("        # 合并所有赛季的数据\n")
            new_lines.append("        df = pd.concat(all_data, ignore_index=True)\n")
            new_lines.append("        \n")
            new_lines.append("        # 处理缺失值\n")
            new_lines.append("        df = df.dropna()\n")
            new_lines.append("        \n")
            new_lines.append("        # 不再过滤赔率数据，保留所有比赛\n")
            new_lines.append("        print(f\"数据量: {len(df)}\")\n")
            new_lines.append("        \n")
            new_lines.append("        # 保存完整特征数据，包含赔率信息\n")
            new_lines.append("        self.X_full = df.copy()\n")
            new_lines.append("        \n")
            new_lines.append("        # 特征和标签分离\n")
            new_lines.append("        features = df.drop(['match_id', 'season', 'result_code', 'home_score', 'away_score'], axis=1, errors='ignore')\n")
            new_lines.append("        labels = df['result_code']\n")
            new_lines.append("        \n")
            new_lines.append("        # 转换分类特征\n")
            new_lines.append("        categorical_cols = features.select_dtypes(include=['object']).columns\n")
            new_lines.append("        if len(categorical_cols) > 0:\n")
            new_lines.append("            features = pd.get_dummies(features, columns=categorical_cols)\n")
            new_lines.append("        \n")
            new_lines.append("        # 分割训练集和测试集\n")
            new_lines.append("        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)\n")
            new_lines.append("        \n")
            new_lines.append("        # 保存分割后的索引，以便后续匹配\n")
            new_lines.append("        self.train_indices = X_train.index\n")
            new_lines.append("        self.test_indices = X_test.index\n")
            new_lines.append("        \n")
            new_lines.append("        return X_train, X_test, y_train, y_test, features.columns.tolist()\n")
            # 跳过原方法的所有行
            while i < len(lines) and not (lines[i].strip() and not lines[i].startswith(' ') and lines[i].strip() != '"""' and lines[i].strip() != "'""'"):
                i += 1
        else:
            new_lines.append(lines[i])
            i += 1
    
    # 保存更新后的DataLoader
    with open(data_loader_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"✓ DataLoader更新完成: {data_loader_path}")

if __name__ == "__main__":
    update_expert_features_with_llm()
    update_data_loader()