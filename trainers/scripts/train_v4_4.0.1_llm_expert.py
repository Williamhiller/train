#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4.0.1模型训练脚本 - 集成LLM专家分析
使用最新的LLM专家分析处理后的特征进行训练
"""

import os
import sys
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trainers.neural_network.lstm_trainer import LSTMTrainer

# Configuration
DATA_ROOT = "/Users/Williamhiler/Documents/my-project/train/train-data"
MODEL_DIR = "/Users/Williamhiler/Documents/my-project/models"
# Use all seasons from 2015-2016 to 2024-2025
SEASONS = [f"{year}-{year+1}" for year in range(2015, 2025)]  # 2015-2016 to 2024-2025 seasons

def train_v4_0_1_llm_expert():
    """
    Train v4.0.1 model with LLM expert analysis integration
    """
    print(f"{'='*70}")
    print(f"TRAINING LSTM MODEL v4.0.1 WITH LLM EXPERT ANALYSIS")
    print(f"{'='*70}")
    print(f"Seasons: {SEASONS}")
    print(f"Data Root: {DATA_ROOT}")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"{'='*70}")
    
    # Initialize trainer with v4.0.1 configuration
    trainer = LSTMTrainer(DATA_ROOT, MODEL_DIR)
    
    # Set version to 4.0.1 explicitly
    trainer.version = "4.0.1"
    trainer.model_save_dir = os.path.join(MODEL_DIR, "v4", "4.0.1")
    os.makedirs(trainer.model_save_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"LLM EXPERT ANALYSIS INTEGRATION")
    print(f"{'='*70}")
    print(f"✓ Using LLM-processed expert features from: train-data/expert/llm_analysis/")
    print(f"✓ Model version: {trainer.version}")
    print(f"✓ Model save directory: {trainer.model_save_dir}")
    print(f"{'='*70}")
    
    # Train with all seasons combined and LLM expert features
    metrics = trainer.train(SEASONS, use_tuning=True, use_llm_expert=True)
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")
    
    # Print detailed performance
    print(f"\n{'='*70}")
    print(f"FINAL LSTM v4.0.1 MODEL PERFORMANCE")
    print(f"{'='*70}")
    print(f"Accuracy: {metrics['metrics']['accuracy']:.4f}")
    print(f"Home Win F1: {metrics['metrics']['home_win_f1']:.4f}")
    print(f"Draw F1: {metrics['metrics']['draw_f1']:.4f}")
    print(f"Away Win F1: {metrics['metrics']['away_win_f1']:.4f}")
    print(f"Weighted F1: {metrics['metrics']['weighted_f1']:.4f}")
    print(f"{'='*70}")
    
    print(f"\nModel saved to: {metrics['model_path']}")
    print(f"Model Info: {metrics['model_path'].replace('lstm_model_4.0.1.h5', 'model_info_4.0.1.json')}")
    return metrics

if __name__ == "__main__":
    train_v4_0_1_llm_expert()