#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析训练数据中的标签分布
"""

import os
import sys
import numpy as np
from collections import Counter

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trainers.data_loader import DataLoader

def main():
    # 初始化数据加载器
    data_root = '/Users/Williamhiler/Documents/my-project/train/train-data'
    data_loader = DataLoader(data_root)
    
    # 加载所有赛季数据
    seasons = ['2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020', 
               '2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
    
    # 准备训练数据
    print("正在加载训练数据...")
    X_train, X_test, y_train, y_test, features = data_loader.prepare_training_data(
        seasons, include_team_state=True, include_expert=True
    )
    
    print(f"总训练样本数: {len(y_train)}")
    print(f"总测试样本数: {len(y_test)}")
    
    # 分析标签分布
    print('\n=== 训练集标签分布 ===')
    train_counts = Counter(y_train)
    print(f'客队胜 (0): {train_counts[0]} ({train_counts[0]/len(y_train)*100:.1f}%)')
    print(f'平局 (1): {train_counts[1]} ({train_counts[1]/len(y_train)*100:.1f}%)')
    print(f'主队胜 (2): {train_counts[2]} ({train_counts[2]/len(y_train)*100:.1f}%)')
    
    print('\n=== 测试集标签分布 ===')
    test_counts = Counter(y_test)
    print(f'客队胜 (0): {test_counts[0]} ({test_counts[0]/len(y_test)*100:.1f}%)')
    print(f'平局 (1): {test_counts[1]} ({test_counts[1]/len(y_test)*100:.1f}%)')
    print(f'主队胜 (2): {test_counts[2]} ({test_counts[2]/len(y_test)*100:.1f}%)')
    
    # 计算类别不平衡比例
    print('\n=== 类别不平衡分析 ===')
    max_class = max(train_counts.values())
    for label, count in sorted(train_counts.items()):
        ratio = max_class / count
        print(f'标签 {label}: 数量={count}, 与最大类别的比例={ratio:.2f}:1')

if __name__ == "__main__":
    main()