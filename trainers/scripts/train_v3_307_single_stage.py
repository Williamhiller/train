#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用2017-2018赛季数据训练V3.0.7单阶段多分类模型
"""

import sys
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# 将项目根目录添加到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 导入修改后的分层模型训练器（现在是单阶段）
from trainers.hierarchical.hierarchical_trainer import HierarchicalModelTrainer

def main():
    print("\n=== 使用2017-2018赛季数据训练V3.0.7单阶段多分类模型 ===")
    
    # 配置参数
    data_root = '/Users/Williamhiler/Documents/my-project/train/train-data'
    model_dir = '/Users/Williamhiler/Documents/my-project/train/models'
    
    # 使用2017-2018单赛季数据
    seasons = ['2017-2018']
    
    # 创建模型训练器
    trainer = HierarchicalModelTrainer(data_root, model_dir)
    
    # 训练模型，确保包含专家特征
    metrics = trainer.train(seasons, include_team_state=True, include_expert=True)
    
    print(f"\n=== 模型训练完成 ===")
    print(f"使用赛季: {seasons[0]}")
    print(f"模型版本: {trainer.version}")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"加权F1分数: {metrics['weighted_f1']:.4f}")
    print(f"主队获胜F1分数: {metrics['home_win_f1']:.4f}")
    print(f"平局F1分数: {metrics['draw_f1']:.4f}")
    print(f"客队获胜F1分数: {metrics['away_win_f1']:.4f}")
    print(f"保存路径: {trainer.model_save_dir}")

if __name__ == "__main__":
    main()