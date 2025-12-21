#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为每个单独赛季训练V3.0.4模型并比较准确率
"""

import sys
import os

# 将项目根目录添加到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 导入模型训练器
from trainers.model_trainer import ModelTrainerV3

def main():
    print("\n=== 开始为所有单独赛季训练V3.0.4模型 ===")
    
    # 配置参数
    data_root = '/Users/Williamhiler/Documents/my-project/train/train-data'
    model_dir = '/Users/Williamhiler/Documents/my-project/train/models'
    model_name = "xgboost"
    
    # 所有可用赛季
    seasons = ['2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020', 
               '2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
    
    # 存储每个赛季的训练结果
    results = {}
    
    # 为每个赛季单独训练模型
    for season in seasons:
        print(f"\n--- 开始训练{season}赛季的V3.0.4模型 ---")
        
        # 创建模型训练器
        trainer = ModelTrainerV3(data_root, model_dir)
        
        try:
            # 训练模型并指定自定义版本号
            model, model_info = trainer.train([season], model_name, custom_version="3.0.4")
            
            # 记录结果
            results[season] = {
                'accuracy': model_info['accuracy'],
                'precision': model_info.get('precision', {}),
                'recall': model_info.get('recall', {}),
                'f1_score': model_info.get('f1_score', {})
            }
            
            print(f"{season}赛季训练完成，准确率: {model_info['accuracy']:.4f}")
        except Exception as e:
            print(f"{season}赛季训练失败: {str(e)}")
            results[season] = {
                'accuracy': 0.0,
                'error': str(e)
            }
    
    # 输出所有赛季的准确率对比
    print("\n=== 所有赛季V3.0.4模型详细准确率对比 ===")
    print("赛季        总准确率   主胜准确率  平局准确率  客胜准确率")
    print("------------------------------------------------")
    
    for season in seasons:
        if 'error' in results[season]:
            print(f"{season}: 训练失败 - {results[season]['error']}")
        else:
            accuracy = results[season]['accuracy']
            precision = results[season]['precision']
            
            # 获取胜平负的精确率
            home_win_precision = precision.get('主胜', 0.0)
            draw_precision = precision.get('平局', 0.0)
            away_win_precision = precision.get('客胜', 0.0)
            
            print(f"{season}: {accuracy:10.4f}  {home_win_precision:10.4f}  {draw_precision:10.4f}  {away_win_precision:10.4f}")
    
    # 计算平均准确率
    valid_results = [r['accuracy'] for r in results.values() if 'error' not in r]
    if valid_results:
        avg_accuracy = sum(valid_results) / len(valid_results)
        print(f"\n平均总准确率: {avg_accuracy:.4f}")
    
    # 输出召回率对比
    print("\n=== 所有赛季V3.0.4模型召回率对比 ===")
    print("赛季        主胜召回率  平局召回率  客胜召回率")
    print("------------------------------------")
    
    for season in seasons:
        if 'error' in results[season]:
            continue
        
        recall = results[season]['recall']
        
        # 获取胜平负的召回率
        home_win_recall = recall.get('主胜', 0.0)
        draw_recall = recall.get('平局', 0.0)
        away_win_recall = recall.get('客胜', 0.0)
        
        print(f"{season}: {home_win_recall:10.4f}  {draw_recall:10.4f}  {away_win_recall:10.4f}")
    
    # 输出F1分数对比
    print("\n=== 所有赛季V3.0.4模型F1分数对比 ===")
    print("赛季        主胜F1     平局F1     客胜F1")
    print("------------------------------------")
    
    for season in seasons:
        if 'error' in results[season]:
            continue
        
        f1_score = results[season]['f1_score']
        
        # 获取胜平负的F1分数
        home_win_f1 = f1_score.get('主胜', 0.0)
        draw_f1 = f1_score.get('平局', 0.0)
        away_win_f1 = f1_score.get('客胜', 0.0)
        
        print(f"{season}: {home_win_f1:10.4f}  {draw_f1:10.4f}  {away_win_f1:10.4f}")
    
    print("\n=== 所有赛季训练完成 ===")

if __name__ == "__main__":
    main()