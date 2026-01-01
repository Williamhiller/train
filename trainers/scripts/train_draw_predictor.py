#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门用于训练和评估平局预测模型的脚本
"""

import os
import json
import numpy as np
from datetime import datetime
from trainers.model_trainer import DrawPredictor

# 配置参数
data_root = '/Users/Williamhiler/Documents/my-project/train/train-data'
model_dir = '/Users/Williamhiler/Documents/my-project/train/models'
model_name = 'xgboost'  # 使用xgboost模型

# 要训练的10个赛季
seasons = ['2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020',
           '2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']

# 用于存储每个赛季的结果
results = {}

# 使用DrawPredictor进行训练
trainer = DrawPredictor(data_root, model_dir)

# 遍历每个赛季，分别训练模型
for season in seasons:
    print(f"\n{'='*60}")
    print(f"开始训练{season}赛季的平局预测模型")
    print(f"{'='*60}")
    
    try:
        # 训练模型
        model, model_info = trainer.train(
            [season],
            model_name=model_name,
            include_team_state=True,  # 使用球队状态特征
            include_expert=True,  # 使用专家特征
            tune_hyperparams=False,  # 不进行超参数调优
            custom_version="3.0.4",
            class_weight='balanced'  # 使用平衡的类权重
        )
        
        # 保存结果
        results[season] = {
            'accuracy': model_info['accuracy'],
            'precision': model_info['precision'],
            'recall': model_info['recall'],
            'f1_score': model_info['f1_score'],
            'classification_report': model_info['classification_report']
        }
        
        print(f"\n{'='*60}")
        print(f"{season}赛季训练完成")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n训练{season}赛季时发生错误: {e}")
        continue

# 输出所有赛季的结果
print(f"\n{'='*80}")
print("所有赛季平局预测模型训练完成")
print(f"{'='*80}")

print("\n各赛季平局预测结果汇总:")
print(f"{'赛季':<15}{'准确率':<10}{'精确率':<10}{'召回率':<10}{'F1分数':<10}")
print(f"{'-'*55}")

# 计算平均值
avg_accuracy = []
avg_precision = []
avg_recall = []
avg_f1 = []

for season, result in results.items():
    print(f"{season:<15}{result['accuracy']:<10.4f}{result['precision']:<10.4f}{result['recall']:<10.4f}{result['f1_score']:<10.4f}")
    
    avg_accuracy.append(result['accuracy'])
    avg_precision.append(result['precision'])
    avg_recall.append(result['recall'])
    avg_f1.append(result['f1_score'])

if avg_accuracy:
    print(f"{'-'*55}")
    print(f"{'平均值':<15}{np.mean(avg_accuracy):<10.4f}{np.mean(avg_precision):<10.4f}{np.mean(avg_recall):<10.4f}{np.mean(avg_f1):<10.4f}")

# 保存结果到文件
results_dir = '/Users/Williamhiler/Documents/my-project/train/results'
os.makedirs(results_dir, exist_ok=True)

# 生成文件名
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
results_filename = f"draw_predictor_results_{current_time}.json"
results_path = os.path.join(results_dir, results_filename)

with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n详细结果已保存至: {results_path}")
print(f"{'='*80}")