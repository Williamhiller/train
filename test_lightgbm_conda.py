#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试LightGBM是否能在conda环境中正常运行
"""

import sys
print(f"Python版本: {sys.version}")

# 测试LightGBM导入
try:
    import lightgbm as lgb
    print("✓ LightGBM导入成功")
    print(f"LightGBM版本: {lgb.__version__}")
    
    # 测试创建模型
    model = lgb.LGBMClassifier(objective='binary', random_state=42)
    print("✓ LightGBM模型创建成功")
    
    # 测试基本功能
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    # 加载示例数据
    data = load_iris()
    X, y = data.data, data.target
    
    # 转换为二分类问题
    y = (y == 0).astype(int)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    import time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"✓ LightGBM模型训练成功，耗时: {training_time:.2f}秒")
    
    # 测试预测
    y_pred = model.predict(X_test)
    print("✓ LightGBM模型预测成功")
    print(f"测试集预测样本数: {len(y_pred)}")
    
    print("\n🎉 LightGBM环境测试全部通过!")
    
except Exception as e:
    print(f"✗ LightGBM测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)