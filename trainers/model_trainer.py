import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import xgboost as xgb
from .data_loader import DataLoader

class BaseModelTrainer:
    def __init__(self, data_root, model_dir):
        self.data_root = data_root
        self.model_dir = model_dir
        self.data_loader = DataLoader(data_root)
        self.models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
        }
        self.scaler = StandardScaler()
    
    def train(self, seasons, model_name='xgboost', include_team_state=False, include_expert=False, tune_hyperparams=False, custom_version=None):
        """训练模型"""
        print(f"开始训练{model_name}模型...")
        
        # 加载数据
        X_train, X_test, y_train, y_test, feature_names = self.data_loader.prepare_training_data(
            seasons, include_team_state, include_expert
        )
        
        print(f"训练集大小: {X_train.shape}")
        print(f"测试集大小: {X_test.shape}")
        print(f"特征数量: {len(feature_names)}")
        
        # 数据标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 选择模型
        if model_name not in self.models:
            raise ValueError(f"不支持的模型: {model_name}")
        
        if tune_hyperparams:
            # 超参数调优
            model = self.tune_hyperparameters(X_train_scaled, y_train, model_name)
        else:
            # 使用默认参数模型
            model = self.models[model_name]
            # 训练模型
            model.fit(X_train_scaled, y_train)
        
        # 模型评估
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
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
            'best_params': model.get_params() if tune_hyperparams else None
        }
        
        self.save_model(model, model_info, custom_version)
        
        return model, model_info
    
    def save_model(self, model, model_info, custom_version=None):
        """保存模型和配置信息"""
        # 创建模型目录
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # 生成模型文件名
        version = "1"
        if model_info['include_team_state'] and model_info['include_expert']:
            version = "3"
        elif model_info['include_team_state']:
            version = "2"
        
        # 根据版本设置文件名
        if model_info['include_team_state'] and model_info['include_expert']:
            # 使用自定义版本号或默认版本号
            if custom_version:
                version_str = custom_version
            else:
                version_str = "3.0.2"
            
            model_filename = f"model_v{version_str}_{model_info['model_name']}.joblib"
            info_filename = f"model_v{version_str}_{model_info['model_name']}_info.json"
            version_dir = "v3"
        elif model_info['include_team_state']:
            version_str = "2.0.2"
            model_filename = f"model_v{version_str}_{model_info['model_name']}.joblib"
            info_filename = f"model_v{version_str}_{model_info['model_name']}_info.json"
            version_dir = "v2"
        else:
            version_str = "1.0.2"
            model_filename = f"model_v{version_str}_{model_info['model_name']}.joblib"
            info_filename = f"model_v{version_str}_{model_info['model_name']}_info.json"
            version_dir = "v1"
        
        # 确保版本子目录存在
        version_dir_path = os.path.join(self.model_dir, version_dir)
        os.makedirs(version_dir_path, exist_ok=True)
        
        model_path = os.path.join(version_dir_path, model_filename)
        
        # 保存模型
        joblib.dump(model, model_path)
        
        info_path = os.path.join(version_dir_path, info_filename)
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        print(f"\n模型已保存至: {model_path}")
        print(f"配置信息已保存至: {info_path}")
    
    def load_model(self, version, model_name):
        """加载模型"""
        # 确定版本子目录
        if version == 3:
            version_dir = "v3"
            model_filename = "model_v3.0.1_xgboost.joblib"  # 特殊处理v3.0.1版本
            info_filename = "model_v3.0.1_xgboost_info.json"  # 特殊处理v3.0.1版本
        else:
            version_dir = f"v{version}"
            model_filename = f"model_v{version}_{model_name}.joblib"
            info_filename = f"model_v{version}_{model_name}_info.json"
        
        # 构造模型和信息文件路径
        version_dir_path = os.path.join(self.model_dir, version_dir)
        model_path = os.path.join(version_dir_path, model_filename)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载模型
        model = joblib.load(model_path)
        
        # 加载配置信息
        info_path = os.path.join(version_dir_path, info_filename)
        
        with open(info_path, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        
        return model, model_info
    
    def predict(self, model, X):
        """使用模型进行预测"""
        X_scaled = self.scaler.transform(X)
        return model.predict(X_scaled), model.predict_proba(X_scaled)

class ModelTrainerV1(BaseModelTrainer):
    """版本1：仅使用赔率特征"""
    def __init__(self, data_root, model_dir):
        super().__init__(data_root, model_dir)
    
    def train(self, seasons, model_name='xgboost', tune_hyperparams=False):
        return super().train(seasons, model_name, include_team_state=False, include_expert=False, tune_hyperparams=tune_hyperparams)

class ModelTrainerV2(BaseModelTrainer):
    """版本2：使用赔率特征 + team_state特征"""
    def __init__(self, data_root, model_dir):
        super().__init__(data_root, model_dir)
    
    def train(self, seasons, model_name='xgboost', tune_hyperparams=False):
        return super().train(seasons, model_name, include_team_state=True, include_expert=False, tune_hyperparams=tune_hyperparams)

class ModelTrainerV3(BaseModelTrainer):
    """版本3：使用赔率特征 + team_state特征 + 专家特征"""
    def __init__(self, data_root, model_dir):
        super().__init__(data_root, model_dir)
    
    def train(self, seasons, model_name='xgboost', tune_hyperparams=False, custom_version=None):
        return super().train(seasons, model_name, include_team_state=True, include_expert=True, tune_hyperparams=tune_hyperparams, custom_version=custom_version)

    def tune_hyperparameters(self, X_train, y_train, model_name='xgboost'):
        """使用GridSearchCV调优模型超参数"""
        print(f"开始调优{model_name}模型超参数...")
        
        if model_name == 'xgboost':
            # XGBoost参数网格
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'gamma': [0, 0.1, 0.2],
                'reg_alpha': [0, 0.01, 0.1],
                'reg_lambda': [0.1, 1, 10]
            }
            
            model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
        elif model_name == 'random_forest':
            # 随机森林参数网格
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            model = RandomForestClassifier(random_state=42)
        elif model_name == 'logistic_regression':
            # 逻辑回归参数网格
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            
            model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        # 使用分层k折交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 创建GridSearchCV对象
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )
        
        # 执行网格搜索
        grid_search.fit(X_train, y_train)
        
        print(f"调优完成！最佳参数: {grid_search.best_params_}")
        print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_

if __name__ == "__main__":
    # 示例用法
    data_root = '/Users/Williamhiler/Documents/my-project/train/train-data'
    model_dir = '/Users/Williamhiler/Documents/my-project/train/models'
    
    # 训练版本1模型（仅赔率特征）
    trainer_v1 = ModelTrainerV1(data_root, model_dir)
    trainer_v1.train(seasons=['2023-2024'], model_name='xgboost')
    
    # 训练版本2模型（赔率特征 + team_state特征）
    trainer_v2 = ModelTrainerV2(data_root, model_dir)
    trainer_v2.train(seasons=['2023-2024'], model_name='xgboost')
    
    # 训练版本3模型（赔率特征 + team_state特征 + 专家特征）
    trainer_v3 = ModelTrainerV3(data_root, model_dir)
    trainer_v3.train(seasons=['2023-2024'], model_name='xgboost')