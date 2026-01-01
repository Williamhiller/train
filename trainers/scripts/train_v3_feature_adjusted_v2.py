#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3模型特征调整训练脚本V2
- 使用与原始模型相同的特征集（包含team_state和expert特征）
- 改进特征调整策略
- 解决平局反向指标问题，优化主模型预测性能
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 添加项目根目录到路径
sys.path.append('/Users/Williamhiler/Documents/my-project/train')

from trainers.model_trainer import ModelTrainerV3
from trainers.data_loader import DataLoader


class ImprovedFeatureAdjustedModelTrainer(ModelTrainerV3):
    """改进的带有特征调整功能的模型训练器"""
    
    def __init__(self, data_root, model_dir):
        super().__init__(data_root, model_dir)
        # 从特征重要性比较中获取需要调整的特征
        self.adjust_features = self._load_adjust_features()
        print(f"已加载需要调整的特征: {list(self.adjust_features.keys())}")
    
    def _load_adjust_features(self):
        """从特征重要性比较文件加载需要调整的特征"""
        feature_importance_file = '/Users/Williamhiler/Documents/my-project/train/feature_importance_comparison.csv'
        
        if not os.path.exists(feature_importance_file):
            raise FileNotFoundError(f"特征重要性比较文件不存在: {feature_importance_file}")
        
        # 加载特征重要性比较数据
        df = pd.read_csv(feature_importance_file)
        
        # 识别需要调整的特征：平局模型中权重过高的特征（差异为负且绝对值较大）
        # 只选择在原始模型中也存在的特征
        adjusted_features = {}
        for _, row in df.iterrows():
            if row['importance_diff'] < -0.01 and row['importance_main'] > 0:
                # 改进的调整系数计算：使用更精细的调整策略
                # 1. 基于特征重要性差异的基础调整
                # 2. 确保调整系数在合理范围内（0.7-1.0）
                max_importance = max(row['importance_main'], row['importance_draw'])
                
                # 使用更温和的调整：只调整25%的差异
                importance_adjust = abs(row['importance_diff']) / max_importance * 0.25
                
                # 基础调整系数
                adjust_coef = 1 - importance_adjust
                
                # 确保调整系数在合理范围内
                adjust_coef = max(0.75, min(1.0, adjust_coef))  # 0.75-1.0之间
                
                adjusted_features[row['feature']] = adjust_coef
        
        return adjusted_features
    
    def _adjust_features(self, X, feature_names):
        """调整特征，减少平局模型中过度权重的特征影响"""
        X_adjusted = X.copy()
        
        for feature_name in feature_names:
            if feature_name in self.adjust_features:
                adjust_coef = self.adjust_features[feature_name]
                X_adjusted[feature_name] = X_adjusted[feature_name] * adjust_coef
                print(f"已调整特征 '{feature_name}'，缩放系数: {adjust_coef:.4f}")
        
        return X_adjusted
    
    def predict(self, model, X, thresholds=None):
        """使用模型进行预测"""
        # 先调整特征
        feature_names = X.columns.tolist()
        X_adjusted = self._adjust_features(X, feature_names)
        
        # 再进行预测
        X_scaled = self.scaler.transform(X_adjusted)
        
        # 确保thresholds是有效的数字列表
        valid_thresholds = None
        if thresholds is not None:
            try:
                # 尝试将thresholds转换为浮点数列表
                valid_thresholds = [float(t) for t in thresholds]
            except (ValueError, TypeError):
                # 如果转换失败，使用默认阈值
                valid_thresholds = None
        
        if valid_thresholds is not None:
            y_pred_proba = model.predict_proba(X_scaled)
            y_pred = self.predict_with_threshold(y_pred_proba, valid_thresholds)
            return y_pred, y_pred_proba
        else:
            return model.predict(X_scaled), model.predict_proba(X_scaled)
    
    def train(self, seasons, model_name='xgboost', include_team_state=True, include_expert=True, 
              tune_hyperparams=False, custom_version=None, class_weight=None, thresholds=None):
        """训练带有特征调整的模型"""
        print(f"开始训练{model_name}模型...")
        
        # 加载数据
        X_train, X_test, y_train, y_test, feature_names = self.data_loader.prepare_training_data(
            seasons, include_team_state, include_expert
        )
        
        print(f"训练集大小: {X_train.shape}")
        print(f"测试集大小: {X_test.shape}")
        print(f"特征数量: {len(feature_names)}")
        
        # 特征调整
        print("\n开始特征调整...")
        X_train_adjusted = self._adjust_features(X_train, feature_names)
        X_test_adjusted = self._adjust_features(X_test, feature_names)
        
        # 数据标准化
        X_train_scaled = self.scaler.fit_transform(X_train_adjusted)
        X_test_scaled = self.scaler.transform(X_test_adjusted)
        
        # 选择模型
        if model_name not in self.models:
            raise ValueError(f"不支持的模型: {model_name}")
        
        if tune_hyperparams:
            # 超参数调优
            model = self.tune_hyperparameters(X_train_scaled, y_train, model_name, class_weight=class_weight)
        else:
            # 使用默认参数模型
            model = self.models[model_name]
            # 训练模型
            if hasattr(model, 'class_weight') and class_weight is not None:
                model.set_params(class_weight=class_weight)
            elif model_name == 'xgboost' and class_weight is not None:
                # XGBoost使用scale_pos_weight处理不平衡数据
                class_counts = np.bincount(y_train)
                if len(class_counts) == 3:  # 确保有三个类别
                    # 为平局类(索引1)设置更高的权重
                    scale_pos_weight = {1: class_counts[0]/class_counts[1]} if class_counts[1] > 0 else {}
                    model.set_params(scale_pos_weight=scale_pos_weight.get(1, 1))
            model.fit(X_train_scaled, y_train)
        
        # 模型评估
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        # 根据阈值进行预测
        if thresholds is not None:
            y_pred = self.predict_with_threshold(y_pred_proba, thresholds)
            print("\n使用自定义阈值进行预测:")
            print(f"阈值设置: 客胜={thresholds[0]}, 平局={thresholds[1]}, 主胜={thresholds[2]}")
        else:
            y_pred = model.predict(X_test_scaled)
            print("\n使用默认阈值进行预测:")
        
        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n模型准确率: {accuracy:.4f}")
        
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=['客胜', '平局', '主胜']))
        
        print("\n混淆矩阵:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # 保存模型和配置
        model_info = {
            'model_name': model_name,
            'include_team_state': include_team_state,
            'include_expert': include_expert,
            'tune_hyperparams': tune_hyperparams,
            'seasons': seasons,
            'feature_names': feature_names,
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, target_names=['客胜', '平局', '主胜'], output_dict=True),
            'best_params': model.get_params() if tune_hyperparams else None,
            'adjusted_features': self.adjust_features
        }
        
        # 保存模型
        if custom_version:
            version = custom_version
        else:
            version = f"3.0.4_feature_adjusted_v3"
        
        version_dir = f"v3"
        model_path = os.path.join(self.model_dir, version_dir, f"model_v{version}_xgboost.joblib")
        info_path = os.path.join(self.model_dir, version_dir, f"model_v{version}_xgboost_info.json")
        
        os.makedirs(os.path.join(self.model_dir, version_dir), exist_ok=True)
        
        # 保存模型
        import joblib
        joblib.dump(model, model_path)
        
        # 保存模型信息
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        print(f"\n模型已保存至: {model_path}")
        print(f"配置信息已保存至: {info_path}")
        
        return model, model_info


def train_with_improved_feature_adjustment():
    """使用改进的特征调整策略训练模型"""
    data_root = '/Users/Williamhiler/Documents/my-project/train/train-data'
    model_dir = '/Users/Williamhiler/Documents/my-project/train/models'
    model_name = 'xgboost'
    seasons = ['2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020',
               '2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
    
    # 配置优化策略
    # 策略1: 针对平局的自定义权重
    class_weight = {0: 1.0, 1: 2.0, 2: 1.0}  # 平局权重设为2倍
    
    # 策略2: 调整分类阈值，降低平局阈值
    thresholds = [0.35, 0.25, 0.35]  # 客胜:0.35, 平局:0.25, 主胜:0.35
    
    print("=== V3模型改进特征调整训练开始 ===")
    print(f"使用策略: 改进特征调整 + 加权损失函数 + 自定义分类阈值")
    print(f"平局权重: {class_weight[1]}倍")
    print(f"分类阈值: 客胜={thresholds[0]}, 平局={thresholds[1]}, 主胜={thresholds[2]}")
    print()
    
    # 训练和评估每个赛季
    results = {}
    for season in seasons:
        print(f"\n{'='*50}")
        print(f"训练{season}赛季模型...")
        print(f"{'='*50}")
        
        try:
            # 使用带有特征调整功能的训练器
            trainer = ImprovedFeatureAdjustedModelTrainer(data_root, model_dir)
            
            # 使用优化策略训练模型
            model, model_info = trainer.train(
                [season], 
                model_name=model_name, 
                include_team_state=True,
                include_expert=True,
                custom_version="3.0.4_feature_adjusted_v3",
                class_weight=class_weight,
                thresholds=thresholds
            )
            
            # 提取结果
            classification_rep = model_info['classification_report']
            draw_metrics = classification_rep.get('平局', {})
            
            results[season] = {
                'accuracy': model_info['accuracy'],
                'draw_precision': draw_metrics.get('precision', 0.0),
                'draw_recall': draw_metrics.get('recall', 0.0),
                'draw_f1': draw_metrics.get('f1-score', 0.0),
                'draw_support': draw_metrics.get('support', 0)
            }
            
            print(f"\n{season}赛季模型训练完成！")
            print(f"总准确率: {model_info['accuracy']:.4f}")
            print(f"平局精确率: {draw_metrics.get('precision', 0.0):.4f}")
            print(f"平局召回率: {draw_metrics.get('recall', 0.0):.4f}")
            print(f"平局F1分数: {draw_metrics.get('f1-score', 0.0):.4f}")
            print(f"平局样本数: {draw_metrics.get('support', 0)}")
            
        except Exception as e:
            print(f"{season}赛季训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 计算平均结果
    print(f"\n{'='*50}")
    print("所有赛季平均结果")
    print(f"{'='*50}")
    
    avg_accuracy = np.mean([r['accuracy'] for r in results.values()])
    avg_draw_precision = np.mean([r['draw_precision'] for r in results.values()])
    avg_draw_recall = np.mean([r['draw_recall'] for r in results.values()])
    avg_draw_f1 = np.mean([r['draw_f1'] for r in results.values()])
    
    print(f"平均总准确率: {avg_accuracy:.4f}")
    print(f"平均平局精确率: {avg_draw_precision:.4f}")
    print(f"平均平局召回率: {avg_draw_recall:.4f}")
    print(f"平均平局F1分数: {avg_draw_f1:.4f}")
    
    # 保存结果
    results['average'] = {
        'accuracy': avg_accuracy,
        'draw_precision': avg_draw_precision,
        'draw_recall': avg_draw_recall,
        'draw_f1': avg_draw_f1
    }
    
    results_file = '/Users/Williamhiler/Documents/my-project/train/feature_adjusted_results_v2.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n训练结果已保存至: {results_file}")


if __name__ == "__main__":
    train_with_improved_feature_adjustment()
