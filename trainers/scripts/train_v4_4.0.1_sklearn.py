#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4.0.1模型训练脚本 - 使用sklearn/lightgbm，集成LLM专家分析
无需TensorFlow依赖，使用最新的LLM专家分析处理后的特征进行训练
"""

import os
import sys
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trainers.model_trainer import BaseModelTrainer

# Configuration
DATA_ROOT = "/Users/Williamhiler/Documents/my-project/train/train-data"
MODEL_DIR = "/Users/Williamhiler/Documents/my-project/models"
# Use all seasons from 2015-2016 to 2024-2025
SEASONS = [f"{year}-{year+1}" for year in range(2015, 2025)]  # 2015-2016 to 2024-2025 seasons

def train_v4_0_1_sklearn():
    """
    Train v4.0.1 model with LLM expert analysis integration using sklearn/lightgbm
    """
    print(f"{'='*70}")
    print(f"TRAINING v4.0.1 MODEL WITH LLM EXPERT ANALYSIS")
    print(f"{'='*70}")
    print(f"Seasons: {SEASONS}")
    print(f"Data Root: {DATA_ROOT}")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"{'='*70}")
    
    # Initialize trainer with BaseModelTrainer (uses sklearn/lightgbm, no TensorFlow)
    trainer = BaseModelTrainer(DATA_ROOT, MODEL_DIR)
    
    # Train with all seasons combined, including LLM expert features
    print(f"\n{'='*70}")
    print(f"LLM EXPERT ANALYSIS INTEGRATION")
    print(f"{'='*70}")
    print(f"✓ Using LLM-processed expert features from: train-data/expert/llm_analysis/")
    print(f"✓ Model version: 4.0.1")
    print(f"✓ Model type: LightGBM (no TensorFlow required)")
    print(f"✓ Seasons: {len(SEASONS)} seasons (2015-2025)")
    print(f"✓ Features: Team state + Expert analysis + LLM processing")
    print(f"{'='*70}")
    
    # Train with LightGBM (best performing model from previous tests)
    model, model_info = trainer.train(
        seasons=SEASONS,
        model_name='lightgbm',  # 使用LightGBM，无需TensorFlow
        include_team_state=True,  # 包含球队状态特征
        include_expert=True,  # 包含专家特征
        tune_hyperparams=True,  # 启用超参数调优
        custom_version='4.0.1',  # 自定义版本号
        class_weight='balanced',  # 处理类别不平衡
        thresholds=[0.33, 0.33, 0.34],  # 自定义阈值
        use_llm=True  # 启用LLM增强专家特征！！！
    )
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")
    
    # Print detailed performance
    print(f"\n{'='*70}")
    print(f"FINAL v4.0.1 MODEL PERFORMANCE")
    print(f"{'='*70}")
    print(f"Model Name: {model_info['model_name']}")
    print(f"Version: 4.0.1")
    print(f"Accuracy: {model_info['accuracy']:.4f}")
    print(f"Home Win F1: {model_info['classification_report']['主胜']['f1-score']:.4f}")
    print(f"Draw F1: {model_info['classification_report']['平局']['f1-score']:.4f}")
    print(f"Away Win F1: {model_info['classification_report']['客胜']['f1-score']:.4f}")
    print(f"{'='*70}")
    
    return model, model_info

if __name__ == "__main__":
    train_v4_0_1_sklearn()