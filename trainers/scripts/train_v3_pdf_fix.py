#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用2017-2018单赛季数据训练V3.0.3模型（包含PDF专家知识修复）
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trainers.data_loader import DataLoader
# 移除不需要的导入
# from trainers.model_trainer import ModelTrainer

def train_v3_pdf_fix():
    """
    训练V3.0.3模型，包含PDF专家知识修复
    """
    print("=== 训练V3.0.3模型（包含PDF专家知识修复） ===")
    print("使用2017-2018单赛季数据进行训练")
    
    # 设置数据根目录
    data_root = "/Users/Williamhiler/Documents/my-project/train/train-data"
    
    # 初始化数据加载器
    data_loader = DataLoader(data_root)
    
    # 加载2017-2018赛季数据
    season = "2017-2018"
    print(f"\n1. 正在加载 {season} 赛季数据...")
    
    # 使用DataLoader的prepare_training_data方法加载并准备数据
    X_train, X_test, y_train, y_test, feature_names = data_loader.prepare_training_data(
        [season], include_team_state=True, include_expert=True
    )
    
    print(f"   ✓ 训练集大小: {len(X_train)}")
    print(f"   ✓ 测试集大小: {len(X_test)}")
    print(f"   ✓ 特征数量: {len(feature_names)}")
    print(f"   ✓ 标签分布: {np.unique(y_train, return_counts=True)}")
    
    # 设置模型参数（与V3.0.2相同）
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'learning_rate': 0.1,
        'max_depth': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # 转换为XGBoost的DMatrix格式
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # 训练模型
    print(f"\n2. 正在训练模型...")
    model = xgb.train(params, dtrain, num_boost_round=100)
    
    # 评估模型
    print(f"\n3. 正在评估模型...")
    y_pred = model.predict(dtest)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred_labels)
    print(f"   ✓ 模型准确率: {accuracy:.4f}")
    
    # 分类报告
    print(f"\n4. 分类报告:")
    print(classification_report(y_test, y_pred_labels, target_names=['客胜', '平局', '主胜']))
    
    # 混淆矩阵
    print(f"\n5. 混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred_labels)
    print(cm)
    
    # 保存模型和配置
    print(f"\n6. 正在保存模型...")
    model_version = "v3.0.3"
    model_name = "xgboost"
    
    # 创建模型目录
    model_dir = os.path.join("/Users/Williamhiler/Documents/my-project/train/models", model_version.split('.')[0])
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存模型文件
    model_path = os.path.join(model_dir, f"model_{model_version}_{model_name}.joblib")
    import joblib
    joblib.dump(model, model_path)
    print(f"   ✓ 模型已保存到: {model_path}")
    
    # 保存模型配置
    model_config = {
        "model_name": model_name,
        "model_version": model_version,
        "accuracy": accuracy,
        "season": season,
        "include_odds": True,
        "include_team_state": True,
        "include_expert": True,
        "tune_hyperparams": False,
        "params": params,
        "classification_report": classification_report(y_test, y_pred_labels, target_names=['客胜', '平局', '主胜'], output_dict=True),
        "confusion_matrix": cm.tolist(),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_names": feature_names,
        "notes": "V3.0.3模型，修复了PDF专家知识使用问题，增加了PDF专家因素"
    }
    
    config_path = os.path.join(model_dir, f"model_{model_version}_{model_name}_info.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(model_config, f, ensure_ascii=False, indent=4)
    print(f"   ✓ 模型配置已保存到: {config_path}")
    
    print(f"\n=== 训练完成 ===")
    print(f"模型版本: {model_version}")
    print(f"模型准确率: {accuracy:.4f}")
    print(f"模型文件: {model_path}")
    print(f"配置文件: {config_path}")
    
    return model_version, accuracy

if __name__ == "__main__":
    train_v3_pdf_fix()