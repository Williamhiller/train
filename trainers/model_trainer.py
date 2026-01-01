import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import lightgbm as lgb
from .data_loader import DataLoader

class BaseModelTrainer:
    def __init__(self, data_root, model_dir):
        self.data_root = data_root
        self.model_dir = model_dir
        self.data_loader = DataLoader(data_root)
        self.models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'lightgbm': lgb.LGBMClassifier(n_estimators=100, random_state=42, objective='multiclass')
        }
        self.scaler = StandardScaler()
    
    def train(self, seasons, model_name='lightgbm', include_team_state=False, include_expert=False, tune_hyperparams=False, custom_version=None, class_weight=None, thresholds=None, use_llm=False):
        """训练模型"""
        print(f"开始训练{model_name}模型...")
        
        # 加载数据
        X_train, X_test, y_train, y_test, feature_names = self.data_loader.prepare_training_data(
            seasons, include_team_state, include_expert, use_llm
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
            model = self.tune_hyperparameters(X_train_scaled, y_train, model_name, class_weight=class_weight)
        else:
            # 使用默认参数模型
            model = self.models[model_name]
            # 训练模型
            if hasattr(model, 'class_weight') and class_weight is not None:
                model.set_params(class_weight=class_weight)
            elif model_name == 'lightgbm' and class_weight is not None:
                # LightGBM支持class_weight参数
                model.set_params(class_weight=class_weight)
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
            
            # 参考3.0.7版本的保存结构，在v3目录下创建版本子目录
            version_subdir = os.path.join(self.model_dir, version_dir, version_str)
            os.makedirs(version_subdir, exist_ok=True)
            
            # 将模型文件保存到版本子目录
            model_path = os.path.join(version_subdir, model_filename)
            info_path = os.path.join(version_subdir, info_filename)
        elif model_info['include_team_state']:
            # 使用自定义版本号或默认版本号
            if custom_version:
                version_str = custom_version
            else:
                version_str = "2.0.2"
            
            model_filename = f"model_v{version_str}_{model_info['model_name']}.joblib"
            info_filename = f"model_v{version_str}_{model_info['model_name']}_info.json"
            version_dir = "v2"
            
            # 参考3.0.7版本的保存结构，在v2目录下创建版本子目录
            version_subdir = os.path.join(self.model_dir, version_dir, version_str)
            os.makedirs(version_subdir, exist_ok=True)
            
            # 将模型文件保存到版本子目录
            model_path = os.path.join(version_subdir, model_filename)
            info_path = os.path.join(version_subdir, info_filename)
        else:
            version_str = "1.0.2"
            model_filename = f"model_v{version_str}_{model_info['model_name']}.joblib"
            info_filename = f"model_v{version_str}_{model_info['model_name']}_info.json"
            version_dir = "v1"
            
            # 确保版本子目录存在
            version_dir_path = os.path.join(self.model_dir, version_dir)
            os.makedirs(version_dir_path, exist_ok=True)
            
            model_path = os.path.join(version_dir_path, model_filename)
            info_path = os.path.join(version_dir_path, info_filename)
        
        # 保存模型
        joblib.dump(model, model_path)
        
        # 保存scaler
        scaler_path = os.path.join(version_subdir, f"scaler_v{version_str}.joblib")
        joblib.dump(self.scaler, scaler_path)
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        print(f"\n模型已保存至: {model_path}")
        print(f"配置信息已保存至: {info_path}")
    
    def load_model(self, version, model_name, custom_version=None):
        """加载模型"""
        # 确定版本子目录和版本字符串
        if version == 3:
            version_dir = "v3"
            if custom_version:
                version_str = custom_version
            else:
                version_str = "3.0.1"  # 特殊处理v3.0.1版本
            model_filename = f"model_v{version_str}_{model_name}.joblib"
            info_filename = f"model_v{version_str}_{model_name}_info.json"
        elif version == 2:
            version_dir = "v2"
            version_str = custom_version if custom_version else "2.0.2"
            model_filename = f"model_v{version_str}_{model_name}.joblib"
            info_filename = f"model_v{version_str}_{model_name}_info.json"
        else:
            version_dir = f"v{version}"
            version_str = custom_version if custom_version else f"{version}.0.2"
            model_filename = f"model_v{version_str}_{model_name}.joblib"
            info_filename = f"model_v{version_str}_{model_name}_info.json"
        
        # 构造模型和信息文件路径
        version_subdir = os.path.join(self.model_dir, version_dir, version_str)
        model_path = os.path.join(version_subdir, model_filename)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载模型
        model = joblib.load(model_path)
        
        # 加载scaler
        scaler_path = os.path.join(version_subdir, f"scaler_v{version_str}.joblib")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler文件不存在: {scaler_path}")
        scaler = joblib.load(scaler_path)
        
        # 加载配置信息
        info_path = os.path.join(version_subdir, info_filename)
        
        with open(info_path, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        
        return model, scaler, model_info
    
    def predict(self, model, X, thresholds=None):
        """使用模型进行预测"""
        X_scaled = self.scaler.transform(X)
        if thresholds is not None:
            y_pred_proba = model.predict_proba(X_scaled)
            y_pred = self.predict_with_threshold(y_pred_proba, thresholds)
            return y_pred, y_pred_proba
        else:
            return model.predict(X_scaled), model.predict_proba(X_scaled)
    
    def predict_with_threshold(self, y_pred_proba, thresholds):
        """根据自定义阈值进行预测
        
        Args:
            y_pred_proba: 模型预测的概率数组，shape (n_samples, 3)
            thresholds: 三个类别的阈值列表，[客胜阈值, 平局阈值, 主胜阈值]
        
        Returns:
            y_pred: 调整后的预测结果
        """
        y_pred = []
        for proba in y_pred_proba:
            # 标准化概率，确保总和为1
            total = sum(proba)
            if total == 0:
                total = 1
            norm_proba = [p/total for p in proba]
            
            # 比较标准化后的概率与阈值
            if norm_proba[0] >= thresholds[0]:
                y_pred.append(0)  # 客胜
            elif norm_proba[1] >= thresholds[1]:
                y_pred.append(1)  # 平局
            else:
                y_pred.append(2)  # 主胜
        
        return np.array(y_pred)

    def tune_hyperparameters(self, X_train, y_train, model_name='lightgbm', class_weight=None):
        """使用RandomizedSearchCV调优模型超参数，减少训练时间"""
        from sklearn.model_selection import RandomizedSearchCV
        import time
        
        print(f"开始调优{model_name}模型超参数...")
        
        if model_name == 'lightgbm':
            # LightGBM参数分布
            param_dist = {
                'n_estimators': [100, 200, 300, 400, 500],
                'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
                'max_depth': [3, 4, 5, 6, 7],
                'subsample': [0.7, 0.75, 0.8, 0.85, 0.9],
                'colsample_bytree': [0.7, 0.75, 0.8, 0.85, 0.9],
                'reg_alpha': [0, 0.01, 0.05, 0.1, 0.2],
                'reg_lambda': [0.01, 0.1, 1, 5, 10],
                'num_leaves': [15, 31, 47, 63, 127]
            }
            
            model = lgb.LGBMClassifier(random_state=42, objective='multiclass', class_weight=class_weight)
            n_iter = 30  # 限制迭代次数，减少训练时间
        elif model_name == 'random_forest':
            # 随机森林参数分布
            param_dist = {
                'n_estimators': [100, 200, 300, 400],
                'max_depth': [3, 5, 7, 10, 15],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', None]
            }
            
            model = RandomForestClassifier(random_state=42, class_weight=class_weight)
            n_iter = 30
        elif model_name == 'logistic_regression':
            # 逻辑回归参数分布
            param_dist = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            
            model = LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weight)
            n_iter = 15
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        # 创建RandomizedSearchCV对象
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        # 执行超参数搜索
        start_time = time.time()
        random_search.fit(X_train, y_train)
        tuning_time = time.time() - start_time
        
        # 打印最佳参数
        print(f"调优时间: {tuning_time:.2f}秒")
        print(f"最佳参数: {random_search.best_params_}")
        print(f"最佳交叉验证分数: {random_search.best_score_:.4f}")
        
        # 返回最佳模型
        return random_search.best_estimator_

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
    
    def train(self, seasons, model_name='xgboost', tune_hyperparams=False, custom_version=None):
        return super().train(seasons, model_name, include_team_state=True, include_expert=False, tune_hyperparams=tune_hyperparams, custom_version=custom_version)

class ModelTrainerV3(BaseModelTrainer):
    """版本3：使用赔率特征 + team_state特征 + 专家特征"""
    def __init__(self, data_root, model_dir):
        super().__init__(data_root, model_dir)
    
    def train(self, seasons, model_name='lightgbm', tune_hyperparams=False, custom_version=None, class_weight=None, thresholds=None):
        return super().train(seasons, model_name, include_team_state=True, include_expert=True, tune_hyperparams=tune_hyperparams, custom_version=custom_version, class_weight=class_weight, thresholds=thresholds)


class DrawPredictor(BaseModelTrainer):
    """专门用于预测平局的模型
    
    将问题转换为二分类：1表示平局，0表示非平局
    """
    def __init__(self, data_root, model_dir):
        super().__init__(data_root, model_dir)
        # 更新模型配置为二分类
        self.models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'lightgbm': lgb.LGBMClassifier(n_estimators=100, random_state=42, objective='binary')
        }
    
    def train(self, seasons, model_name='lightgbm', include_team_state=False, include_expert=False, tune_hyperparams=False, custom_version=None, class_weight=None, threshold=0.5):
        """训练平局预测模型"""
        print(f"开始训练专门的{model_name}平局预测模型...")
        
        # 加载数据
        X_train, X_test, y_train, y_test, feature_names = self.data_loader.prepare_training_data(
            seasons, include_team_state, include_expert
        )
        
        # 将标签转换为二分类：1=平局, 0=非平局
        y_train_binary = np.where(y_train == 1, 1, 0)
        y_test_binary = np.where(y_test == 1, 1, 0)
        
        print(f"训练集大小: {X_train.shape}")
        print(f"测试集大小: {X_test.shape}")
        print(f"特征数量: {len(feature_names)}")
        print(f"训练集中平局样本: {np.sum(y_train_binary)} ({np.sum(y_train_binary)/len(y_train_binary)*100:.2f}%)")
        print(f"测试集中平局样本: {np.sum(y_test_binary)} ({np.sum(y_test_binary)/len(y_test_binary)*100:.2f}%)")
        
        # 数据标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 选择模型
        if model_name not in self.models:
            raise ValueError(f"不支持的模型: {model_name}")
        
        if tune_hyperparams:
            # 超参数调优
            model = self.tune_hyperparameters_draw(X_train_scaled, y_train_binary, model_name, class_weight=class_weight)
        else:
            # 使用默认参数模型
            model = self.models[model_name]
            # 训练模型
            if hasattr(model, 'class_weight') and class_weight is not None:
                model.set_params(class_weight=class_weight)
            elif model_name == 'lightgbm' and class_weight is not None:
                # LightGBM支持class_weight参数
                model.set_params(class_weight=class_weight)
            model.fit(X_train_scaled, y_train_binary)
        
        # 模型评估
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        # 阈值分析：测试不同阈值的效果
        print("\n=== 阈值分析 ===")
        thresholds_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        best_threshold = threshold
        best_f1 = 0
        
        for t in thresholds_to_test:
            temp_pred = (y_pred_proba[:, 1] >= t).astype(int)
            temp_precision = precision_score(y_test_binary, temp_pred)
            temp_recall = recall_score(y_test_binary, temp_pred)
            temp_f1 = f1_score(y_test_binary, temp_pred)
            
            print(f"阈值 {t:.1f}: 精确率={temp_precision:.4f}, 召回率={temp_recall:.4f}, F1分数={temp_f1:.4f}")
            
            if temp_f1 > best_f1:
                best_f1 = temp_f1
                best_threshold = t
        
        print(f"\n最佳阈值: {best_threshold:.1f}, 最佳F1分数: {best_f1:.4f}")
        
        # 使用最佳阈值或自定义阈值进行预测
        final_threshold = best_threshold if threshold == 0.5 else threshold
        print(f"\n使用最终阈值 {final_threshold:.2f} 进行预测:")
        y_pred = (y_pred_proba[:, 1] >= final_threshold).astype(int)
        
        # 计算评估指标
        accuracy = accuracy_score(y_test_binary, y_pred)
        print(f"\n模型准确率: {accuracy:.4f}")
        
        print("\n分类报告:")
        print(classification_report(y_test_binary, y_pred, target_names=['非平局', '平局']))
        
        print("\n混淆矩阵:")
        cm = confusion_matrix(y_test_binary, y_pred)
        print(cm)
        
        # 计算精确率、召回率和F1分数
        precision = precision_score(y_test_binary, y_pred)
        recall = recall_score(y_test_binary, y_pred)
        f1 = f1_score(y_test_binary, y_pred)
        
        print(f"\n平局预测精确率: {precision:.4f}")
        print(f"平局预测召回率: {recall:.4f}")
        print(f"平局预测F1分数: {f1:.4f}")
        
        # 分析特征相关性与重要性的一致性
        print("\n=== 特征相关性与重要性分析 ===")
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # 计算特征与平局结果的相关性
            import pandas as pd
            X_df = pd.DataFrame(X_train_scaled, columns=feature_names)
            y_df = pd.Series(y_train_binary, name="draw_result")
            correlations = X_df.corrwith(y_df)
            
            # 打印前10个重要特征及其相关性
            print("\n前10个重要特征及其与平局的相关性:")
            for f in range(min(10, len(feature_names))):
                feature_name = feature_names[indices[f]]
                importance = importances[indices[f]]
                correlation = correlations[feature_name] if feature_name in correlations else 0
                print(f"{f+1}. {feature_name}: 重要性={importance:.4f}, 相关性={correlation:.4f}")
        
        # 保存模型和配置
        model_info = {
            'model_name': model_name,
            'include_team_state': include_team_state,
            'include_expert': include_expert,
            'tune_hyperparams': tune_hyperparams,
            'seasons': seasons,
            'feature_names': feature_names,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_report(y_test_binary, y_pred, target_names=['非平局', '平局'], output_dict=True),
            'best_params': model.get_params() if tune_hyperparams else None
        }
        
        # 自定义保存路径
        self.save_draw_model(model, model_info, custom_version)
        
        return model, model_info
    
    def save_draw_model(self, model, model_info, custom_version=None):
        """保存平局预测模型和配置信息"""
        # 创建模型目录
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # 生成模型文件名
        version = "1"
        if model_info['include_team_state'] and model_info['include_expert']:
            version = "3"
        elif model_info['include_team_state']:
            version = "2"
        
        # 使用专门的平局模型版本标识
        if custom_version:
            version_str = f"draw_{custom_version}"
        else:
            version_str = f"draw_v{version}_0.1"
        
        model_filename = f"model_{version_str}_{model_info['model_name']}.joblib"
        info_filename = f"model_{version_str}_{model_info['model_name']}_info.json"
        version_dir = "draw_models"
        
        # 确保版本子目录存在
        version_dir_path = os.path.join(self.model_dir, version_dir)
        os.makedirs(version_dir_path, exist_ok=True)
        
        model_path = os.path.join(version_dir_path, model_filename)
        
        # 保存模型
        joblib.dump(model, model_path)
        
        info_path = os.path.join(version_dir_path, info_filename)
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        print(f"\n平局预测模型已保存至: {model_path}")
        print(f"配置信息已保存至: {info_path}")
    
    def tune_hyperparameters_draw(self, X_train, y_train, model_name='xgboost', class_weight=None):
        """使用GridSearchCV调优平局预测模型超参数"""
        print(f"开始调优{model_name}平局预测模型超参数...")
        
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
            
            # 计算平局样本的权重
            class_counts = np.bincount(y_train)
            if class_weight is not None and len(class_counts) == 2 and class_counts[1] > 0:
                # 为XGBoost设置scale_pos_weight
                scale_pos_weight = class_counts[0]/class_counts[1]
                model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
            else:
                model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        elif model_name == 'random_forest':
            # 随机森林参数网格
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            model = RandomForestClassifier(random_state=42, class_weight=class_weight)
        elif model_name == 'logistic_regression':
            # 逻辑回归参数网格
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            
            model = LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weight)
        
        # 使用GridSearchCV进行调优
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1',  # 使用F1分数作为评估指标，更适合不平衡数据
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳F1分数: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator()
    
    def predict_draw(self, model, X, threshold=0.5):
        """使用模型预测平局概率
        
        Args:
            model: 训练好的平局预测模型
            X: 特征数据
            threshold: 预测阈值
            
        Returns:
            y_pred: 预测结果 (1=平局, 0=非平局)
            y_pred_proba: 预测概率
        """
        X_scaled = self.scaler.transform(X)
        y_pred_proba = model.predict_proba(X_scaled)
        y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)
        return y_pred, y_pred_proba

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