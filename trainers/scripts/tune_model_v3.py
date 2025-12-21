#!/usr/bin/env python3
"""
带超参数调优的版本3模型训练脚本
版本3：使用赔率特征 + team_state特征 + 专家特征
"""

import os
from trainers.model_trainer import ModelTrainerV3

# 配置参数
data_root = '/Users/Williamhiler/Documents/my-project/train/train-data'
model_save_dir = '/Users/Williamhiler/Documents/my-project/train/models'
season = '2017-2018'  # 训练的赛季
model_name = 'xgboost'  # 模型类型

def main():
    print("========================================================")
    print("开始训练版本3模型（带超参数调优）：")
    print("使用赔率特征 + team_state特征 + 专家特征")
    print(f"赛季：{season}")
    print(f"模型：{model_name}")
    print("========================================================")
    
    # 创建训练器实例
    trainer_v3 = ModelTrainerV3(data_root, model_save_dir)
    
    # 训练模型（开启超参数调优）
    model, model_info = trainer_v3.train(
        seasons=[season],
        model_name=model_name,
        tune_hyperparams=True  # 开启超参数调优
    )
    
    print("\n========================================================")
    print("模型训练完成！")
    print(f"训练完成后的准确率: {model_info['accuracy']:.4f}")
    if model_info['tune_hyperparams']:
        print("\n最佳参数配置:")
        for param, value in model_info['best_params'].items():
            if param in ['n_estimators', 'learning_rate', 'max_depth', 'subsample', 'colsample_bytree', 'gamma', 'reg_alpha', 'reg_lambda']:
                print(f"  {param}: {value}")
    print("========================================================")

if __name__ == "__main__":
    main()