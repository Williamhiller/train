#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4.0.2模型训练脚本 - 融合LLM专家分析
使用LLM增强的专家特征训练模型
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

def train_v4_0_2_llm():
    """
    Train v4.0.2 model with integrated LLM expert analysis
    """
    print(f"{'='*75}")
    print(f"TRAINING v4.0.2 MODEL WITH INTEGRATED LLM EXPERT ANALYSIS")
    print(f"{'='*75}")
    print(f"Seasons: {SEASONS}")
    print(f"Data Root: {DATA_ROOT}")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"{'='*75}")
    
    # Initialize trainer
    trainer = BaseModelTrainer(DATA_ROOT, MODEL_DIR)
    
    print(f"\n{'='*75}")
    print(f"LLM EXPERT ANALYSIS INTEGRATION DETAILS")
    print(f"{'='*75}")
    print(f"✓ Model version: 4.0.2")
    print(f"✓ Model type: LightGBM (no TensorFlow required)")
    print(f"✓ Seasons: {len(SEASONS)} seasons (2015-2025)")
    print(f"✓ Features: Team state + LLM-enhanced expert analysis")
    print(f"✓ LLM features: Confidence, Prediction Distribution, Sentiment Analysis")
    print(f"{'='*75}")
    
    # Train with LightGBM using LLM-enhanced expert features
    model, model_info = trainer.train(
        seasons=SEASONS,
        model_name='lightgbm',
        include_team_state=True,
        include_expert=True,
        tune_hyperparams=True,
        custom_version='4.0.2',
        class_weight='balanced',
        thresholds=[0.33, 0.33, 0.34]
    )
    
    print(f"\n{'='*75}")
    print(f"TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*75}")
    
    # Print detailed performance
    print(f"\n{'='*75}")
    print(f"FINAL v4.0.2 MODEL PERFORMANCE WITH LLM EXPERT ANALYSIS")
    print(f"{'='*75}")
    print(f"Model Name: {model_info['model_name']}")
    print(f"Version: 4.0.2")
    print(f"Accuracy: {model_info['accuracy']:.4f}")
    print(f"Home Win F1: {model_info['classification_report']['主胜']['f1-score']:.4f}")
    print(f"Draw F1: {model_info['classification_report']['平局']['f1-score']:.4f}")
    print(f"Away Win F1: {model_info['classification_report']['客胜']['f1-score']:.4f}")
    print(f"Weighted F1: {model_info['classification_report']['weighted avg']['f1-score']:.4f}")
    print(f"{'='*75}")
    
    return model, model_info

if __name__ == "__main__":
    train_v4_0_2_llm()