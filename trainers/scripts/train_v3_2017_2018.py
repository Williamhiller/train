#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用2017-2018赛季数据训练V3.0.4模型
"""

import sys
import os

# 将项目根目录添加到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 现在可以正确导入模块了
from trainers.model_trainer import ModelTrainerV3

def main():
    print("\n=== 使用2017-2018赛季数据训练V3.0.4模型 ===")
    
    # 配置参数
    data_root = '/Users/Williamhiler/Documents/my-project/train/train-data'
    model_dir = '/Users/Williamhiler/Documents/my-project/train/models'
    
    # 使用2017-2018单赛季数据
    seasons = ['2017-2018']
    model_name = "xgboost"
    custom_version = "3.0.4"  # 指定自定义版本号
    
    # 创建模型训练器
    trainer = ModelTrainerV3(data_root, model_dir)
    
    # 训练模型
    model, model_info = trainer.train(seasons, model_name, custom_version=custom_version)
    
    print(f"\n=== 模型训练完成 ===")
    print(f"使用赛季: {seasons[0]}")
    print(f"模型版本: {custom_version}")
    print(f"准确率: {model_info['accuracy']:.4f}")
    print(f"特征数量: {len(model_info['feature_names'])}")
    print(f"保存路径: {os.path.join(model_dir, 'v3')}")

if __name__ == "__main__":
    main()