#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练V3.0.4模型
"""

import sys
import os

# 将项目根目录添加到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 导入模型训练器
from trainers.model_trainer import ModelTrainerV3

def main():
    print("\n=== 开始训练V3.0.4模型 ===")
    
    # 配置参数
    data_root = '/Users/Williamhiler/Documents/my-project/train/train-data'
    model_dir = '/Users/Williamhiler/Documents/my-project/train/models'
    
    # 选择要训练的赛季 - 使用所有赛季
    seasons = ['2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020', 
               '2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
    model_name = "xgboost"
    
    # 创建模型训练器
    trainer = ModelTrainerV3(data_root, model_dir)
    
    # 训练模型并指定自定义版本号3.0.4
    model, model_info = trainer.train(seasons, model_name, custom_version="3.0.4")
    
    print(f"\n=== V3.0.4模型训练完成 ===")
    print(f"使用赛季: {', '.join(seasons)}")
    print(f"准确率: {model_info['accuracy']:.4f}")
    print(f"特征数量: {len(model_info['feature_names'])}")
    print(f"模型已保存至: {model_dir}/v3/")

if __name__ == "__main__":
    main()