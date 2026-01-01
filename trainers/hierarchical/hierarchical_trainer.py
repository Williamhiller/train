import os
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import joblib
import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from ..model_trainer import BaseModelTrainer
from ..data_loader import DataLoader

class HierarchicalModelTrainer(BaseModelTrainer):
    def __init__(self, data_root, model_dir):
        super().__init__(data_root, model_dir)
        self.version = "4.0.1"
        self.model_save_dir = os.path.join(model_dir, "v4", "4.0.1")
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # 从best_params.json加载最佳配置
        self.best_params_path = os.path.join(os.path.dirname(__file__), "best_params.json")
        with open(self.best_params_path, 'r') as f:
            self.best_params = json.load(f)
        
        # 使用best_params中的阈值
        self.draw_threshold = self.best_params['draw_threshold']
        self.home_win_threshold = self.best_params['home_win_threshold']
        
        print(f"Loaded best parameters from {self.best_params_path}")
        print(f"Initial draw threshold: {self.draw_threshold}")
        print(f"Initial home win threshold: {self.home_win_threshold}")
        
    def train(self, seasons=None, model_name='lightgbm', include_team_state=True, include_expert=True, use_tuning=False):
        # 如果没有指定赛季，使用所有可用赛季
        if seasons is None:
            seasons = ['2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020', 
                      '2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
            print(f"Training hybrid model v{self.version} for all seasons: {seasons}")
        else:
            print(f"Training hybrid model v{self.version} for seasons: {seasons}")
        
        # Prepare training data using BaseModelTrainer's data_loader
        print("\n=== Preparing Training Data ===")
        X_train, X_test, y_train, y_test, features = self.data_loader.prepare_training_data(
            seasons, include_team_state, include_expert
        )
        
        print(f"Train Data Shape: {X_train.shape}")
        print(f"Test Data Shape: {X_test.shape}")
        print(f"Features: {len(features)}")
        print(f"Original Label Distribution: {np.unique(y_train, return_counts=True)}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 创建平/非平标签 (1: 平局, 0: 非平局)
        y_train_draw_binary = (y_train == 1).astype(int)
        y_test_draw_binary = (y_test == 1).astype(int)
        
        print(f"\nDraw Binary Label Distribution (Train): {np.unique(y_train_draw_binary, return_counts=True)}")
        print(f"Draw Binary Label Distribution (Test): {np.unique(y_test_draw_binary, return_counts=True)}")
        
        # 对平局样本进行更平衡的采样，结合过采样和欠采样，提高平局样本的质量
        print("\n=== Balancing Training Data for Draw/Non-Draw Model ===")
        from imblearn.over_sampling import SVMSMOTE
        from imblearn.under_sampling import NearMiss
        from imblearn.pipeline import Pipeline
        
        # 计算原始数据中的平局样本比例
        non_draw_count = np.sum(y_train_draw_binary == 0)
        draw_count = np.sum(y_train_draw_binary == 1)
        original_draw_ratio = draw_count / non_draw_count if non_draw_count > 0 else 0
        
        print(f"Original Draw Ratio: {original_draw_ratio:.4f} (Draw: {draw_count}, Non-Draw: {non_draw_count})")
        
        # 组合过采样和欠采样，创建更平衡的数据集
        # 首先使用SMOTE增加平局样本，然后使用NearMiss减少非平局样本
        over_sampler = SVMSMOTE(random_state=42, sampling_strategy=1.0, k_neighbors=3)  # 增加平局样本到与非平局样本相等
        under_sampler = NearMiss(version=2, sampling_strategy=1.0)  # 减少非平局样本到与平局样本相等，使用version=2提高质量
        
        sampling_pipeline = Pipeline([
            ('over', over_sampler),
            ('under', under_sampler)
        ])
        
        # 应用采样管道
        X_train_draw_scaled, y_train_draw_binary_balanced = sampling_pipeline.fit_resample(X_train_scaled, y_train_draw_binary)
        
        print(f"\nAfter SVM-SMOTE Over Sampling for Draw Model:")
        print(f"Draw Model Train Data Shape: {X_train_draw_scaled.shape}")
        print(f"Draw Binary Label Distribution (Train): {np.unique(y_train_draw_binary_balanced, return_counts=True)}")
        
        # 创建胜负标签（仅用于非平局情况，0: 客胜, 2: 主胜）
        # 筛选出非平局样本 - 胜负模型使用原始的非平局样本
        non_draw_train_indices = y_train != 1
        non_draw_test_indices = y_test != 1
        
        X_train_non_draw = X_train_scaled[non_draw_train_indices]
        y_train_non_draw = y_train[non_draw_train_indices]
        
        X_test_non_draw = X_test_scaled[non_draw_test_indices]
        y_test_non_draw = y_test[non_draw_test_indices]
        
        # 将胜负标签转换为二元分类 (0: 客胜, 1: 主胜)
        y_train_win_loss = (y_train_non_draw == 2).astype(int)
        y_test_win_loss = (y_test_non_draw == 2).astype(int)
        
        print(f"\nWin/Loss Label Distribution (Train): {np.unique(y_train_win_loss, return_counts=True)}")
        print(f"Win/Loss Label Distribution (Test): {np.unique(y_test_win_loss, return_counts=True)}")
        
        # 对胜负模型进行SVM-SMOTE过采样，使用更精确的采样比例
        print("\n=== Balancing Training Data for Win/Loss Model ===")
        from imblearn.over_sampling import SVMSMOTE
        
        # 计算原始数据中的主胜样本比例
        away_win_count = np.sum(y_train_win_loss == 0)
        home_win_count = np.sum(y_train_win_loss == 1)
        original_home_win_ratio = home_win_count / away_win_count if away_win_count > 0 else 0
        
        print(f"Original Home Win Ratio: {original_home_win_ratio:.4f} (Home Win: {home_win_count}, Away Win: {away_win_count})")
        
        # 设置固定的采样比例，使主胜和客胜样本更加平衡
        target_sampling_strategy = 1.0  # 主胜样本数/客胜样本数 = 1.0，完全平衡
        
        # 应用SVM-SMOTE过采样到胜负模型的训练数据
        smote_win_loss = SVMSMOTE(random_state=42, sampling_strategy=target_sampling_strategy, k_neighbors=5)
        X_train_win_loss_scaled, y_train_win_loss_balanced = smote_win_loss.fit_resample(X_train_non_draw, y_train_win_loss)
        
        print(f"\nAfter SVM-SMOTE Over Sampling for Win/Loss Model:")
        print(f"Win/Loss Model Train Data Shape: {X_train_win_loss_scaled.shape}")
        print(f"Win/Loss Label Distribution (Train): {np.unique(y_train_win_loss_balanced, return_counts=True)}")
        
        # 如果使用超参数调优，使用最终加权F1分数作为优化目标
        if use_tuning:
            print("\n=== Hyperparameter Tuning with Weighted F1 as Objective ===")
            # 使用最终加权F1分数作为超参数优化目标
            draw_model, win_loss_model = self._tune_hyperparameters(
                X_train_scaled, y_train,
                X_test_scaled, y_test,
                X_train_draw_scaled, y_train_draw_binary_balanced,
                X_train_win_loss_scaled, y_train_win_loss_balanced,
                y_test_draw_binary, y_test_win_loss
            )
        else:
            # 训练平/非平模型
            print("\n=== Training Draw/Non-Draw Model (Binary Classification) ===")
            draw_model = self._train_binary_model(
                X_train_draw_scaled, y_train_draw_binary_balanced,
                X_test_scaled, y_test_draw_binary,
                "draw_model",
                use_tuning
            )
            
            # 训练胜负模型（仅用于非平局情况）
            print("\n=== Training Win/Loss Model (Binary Classification for Non-Draw) ===")
            win_loss_model = self._train_binary_model(
                X_train_win_loss_scaled, y_train_win_loss_balanced,
                X_test_non_draw, y_test_win_loss,
                "win_loss_model",
                use_tuning
            )
        
        # 组合预测并评估
        print("\n=== Evaluating Hybrid Model ===")
        y_pred_hybrid, y_pred_proba_hybrid = self._combine_predictions(
            X_test_scaled, draw_model, win_loss_model
        )
        
        # 计算评估指标
        print("Hybrid Model Performance (Default Thresholds):")
        print(classification_report(y_test, y_pred_hybrid, target_names=['Away Win', 'Draw', 'Home Win']))
        
        accuracy = accuracy_score(y_test, y_pred_hybrid)
        weighted_f1 = f1_score(y_test, y_pred_hybrid, average='weighted')
        home_win_f1 = f1_score(y_test, y_pred_hybrid, average=None)[2]
        draw_f1 = f1_score(y_test, y_pred_hybrid, average=None)[1]
        away_win_f1 = f1_score(y_test, y_pred_hybrid, average=None)[0]
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Weighted F1-Score: {weighted_f1:.4f}")
        print(f"Home Win F1-Score: {home_win_f1:.4f}")
        print(f"Draw F1-Score: {draw_f1:.4f}")
        print(f"Away Win F1-Score: {away_win_f1:.4f}")
        
        # 使用阈值调优找到最佳阈值
        best_thresholds, best_metrics = self.tune_thresholds(
            X_test_scaled, y_test, draw_model, win_loss_model
        )
        
        # 使用最佳阈值重新生成预测
        y_pred_best, y_pred_proba_best = self._combine_predictions_with_thresholds(
            X_test_scaled, draw_model, win_loss_model, best_thresholds[0], best_thresholds[1]
        )
        
        # 更新最佳阈值的性能指标
        print("\n=== Hybrid Model Performance (Best Thresholds) ===")
        print(classification_report(y_test, y_pred_best, target_names=['Away Win', 'Draw', 'Home Win']))
        print(f"Best Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"Best Weighted F1-Score: {best_metrics['weighted_f1']:.4f}")
        print(f"Best Home Win F1-Score: {best_metrics['home_win_f1']:.4f}")
        print(f"Best Draw F1-Score: {best_metrics['draw_f1']:.4f}")
        print(f"Best Away Win F1-Score: {best_metrics['away_win_f1']:.4f}")
        
        # 更新模型信息为最佳阈值的性能
        accuracy = best_metrics['accuracy']
        weighted_f1 = best_metrics['weighted_f1']
        home_win_f1 = best_metrics['home_win_f1']
        draw_f1 = best_metrics['draw_f1']
        away_win_f1 = best_metrics['away_win_f1']
        
        # 更新模型的阈值为最佳阈值
        self.draw_threshold = best_thresholds[0]
        self.home_win_threshold = best_thresholds[1]
        
        # 使用最佳阈值的预测结果
        y_pred_hybrid = y_pred_best
        y_pred_proba_hybrid = y_pred_proba_best
        
        # 保存模型、scaler和特征
        season_key = self.version
        
        # 保存平/非平模型
        draw_model_path = os.path.join(self.model_save_dir, f"draw_model_{season_key}.joblib")
        joblib.dump(draw_model, draw_model_path)
        
        # 保存胜负模型
        win_loss_model_path = os.path.join(self.model_save_dir, f"win_loss_model_{season_key}.joblib")
        joblib.dump(win_loss_model, win_loss_model_path)
        
        # 保存scaler和features
        scaler_save_path = os.path.join(self.model_save_dir, f"scaler_{season_key}.joblib")
        joblib.dump(scaler, scaler_save_path)
        
        features_save_path = os.path.join(self.model_save_dir, f"features_{season_key}.joblib")
        joblib.dump(features, features_save_path)
        
        # 保存模型信息
        model_info = {
            "version": self.version,
            "season_key": season_key,
            "seasons": seasons,
            "include_team_state": include_team_state,
            "include_expert": include_expert,
            "use_tuning": use_tuning,
            "draw_threshold": self.draw_threshold,
            "home_win_threshold": getattr(self, 'home_win_threshold', best_thresholds[1]),
            "best_thresholds": {
                "draw_threshold": best_thresholds[0],
                "home_win_threshold": best_thresholds[1]
            },
            "train_shape": X_train.shape,
            "test_shape": X_test.shape,
            "features": features,
            "metrics": {
                "accuracy": accuracy,
                "weighted_f1": weighted_f1,
                "home_win_f1": home_win_f1,
                "draw_f1": draw_f1,
                "away_win_f1": away_win_f1
            },
            "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "draw_model_path": draw_model_path,
            "win_loss_model_path": win_loss_model_path,
            "scaler_path": scaler_save_path,
            "features_path": features_save_path
        }
        
        info_save_path = os.path.join(self.model_save_dir, f"model_info_{season_key}.json")
        with open(info_save_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(model_info, f, ensure_ascii=False, indent=4)
        
        # 输出测试集预测结果
        self._output_predictions(X_test, y_test, y_pred_hybrid, y_pred_proba_hybrid, season_key)
        
        print(f"\nModels saved to: {self.model_save_dir}")
        print(f"Model Info: {info_save_path}")
        return model_info
    
    def _train_binary_model(self, X_train, y_train, X_test, y_test, model_name, use_tuning):
        """训练二元分类模型，使用软投票集成LightGBM、随机森林和XGBoost"""
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
        from sklearn.feature_selection import SelectFromModel
        import xgboost as xgb
        import time
        
        # 为平/非平模型设置专门的参数和类别权重
        is_draw_model = model_name == "draw_model"
        is_win_loss_model = model_name == "win_loss_model"
        
        # 移除特征选择，避免丢失重要信息
        pass
        
        # 训练LightGBM模型
        print(f"\nTraining LightGBM for {model_name}...")
        if use_tuning:
            lgb_model, best_params = self._tune_binary_hyperparameters(X_train, y_train, model_name)
        else:
            # 针对draw_model和win_loss_model使用不同的参数
            if is_draw_model:
                # 针对平局模型的特殊参数，提高召回率
                params = {
                    'objective': 'binary',
                    'random_state': 42,
                    'n_estimators': 3000,
                    'learning_rate': 0.03,
                    'max_depth': 10,
                    'num_leaves': 512,
                    'subsample': 0.7,
                    'colsample_bytree': 0.7,
                    'reg_alpha': 0.05,
                    'reg_lambda': 0.1,
                    'min_split_gain': 0.005,
                    'min_child_samples': 15,
                    'verbosity': -1,
                    'class_weight': 'balanced',
                    'scale_pos_weight': 2.0  # 进一步增加正样本权重
                }
            else:
                # 针对胜负模型的参数
                params = {
                    'objective': 'binary',
                    'random_state': 42,
                    'n_estimators': 2000,
                    'learning_rate': 0.05,
                    'max_depth': 8,
                    'num_leaves': 256,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.2,
                    'min_split_gain': 0.01,
                    'min_child_samples': 20,
                    'verbosity': -1
                }
            
            lgb_model = lgb.LGBMClassifier(**params)
            
            start_time = time.time()
            
            # 选择合适的评估指标
            eval_metric = ['binary_logloss', 'auc']
            if is_draw_model:
                # 对平局模型使用更多的评估指标
                eval_metric = ['binary_logloss', 'auc', 'binary_error']
            
            lgb_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric=eval_metric,
                callbacks=[
                    early_stopping(stopping_rounds=100),  # 增加停止轮数
                    log_evaluation(period=50)
                ]
            )
            train_time = time.time() - start_time
            print(f"{model_name} LightGBM Training Time: {train_time:.2f} seconds")
        
        # 评估LightGBM模型
        lgb_y_pred = lgb_model.predict(X_test)
        print(f"\n{model_name} LightGBM Performance:")
        print(classification_report(y_test, lgb_y_pred))
        print(f"Accuracy: {accuracy_score(y_test, lgb_y_pred):.4f}")
        print(f"F1-Score: {f1_score(y_test, lgb_y_pred):.4f}")
        print(f"Recall: {recall_score(y_test, lgb_y_pred):.4f}")
        
        # 训练随机森林模型
        print(f"\nTraining Random Forest for {model_name}...")
        rf_start_time = time.time()
        
        if is_draw_model:
            # 针对平局模型的随机森林参数
            rf_model = RandomForestClassifier(
                n_estimators=3000,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=1,
                class_weight='balanced',
                criterion='entropy'
            )
        else:
            # 针对胜负模型的随机森林参数
            rf_model = RandomForestClassifier(
                n_estimators=2000,
                max_depth=15,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=1,
                criterion='gini'
            )
        
        rf_model.fit(X_train, y_train)
        
        rf_train_time = time.time() - rf_start_time
        print(f"{model_name} Random Forest Training Time: {rf_train_time:.2f} seconds")
        
        # 评估随机森林模型
        rf_y_pred = rf_model.predict(X_test)
        print(f"\n{model_name} Random Forest Performance:")
        print(classification_report(y_test, rf_y_pred))
        print(f"Accuracy: {accuracy_score(y_test, rf_y_pred):.4f}")
        print(f"F1-Score: {f1_score(y_test, rf_y_pred):.4f}")
        
        # 计算类别权重
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        # 训练XGBoost模型
        print(f"\nTraining XGBoost for {model_name}...")
        xgb_start_time = time.time()
        
        if is_draw_model:
            # 针对平局模型的XGBoost参数
            xgb_model = xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=42,
                n_estimators=3000,
                learning_rate=0.03,
                max_depth=10,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.05,
                reg_lambda=0.1,
                min_child_weight=2,
                gamma=0.05,
                scale_pos_weight=3.0,  # 大幅增加正样本权重
                n_jobs=1
            )
        else:
            # 针对胜负模型的XGBoost参数
            xgb_model = xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=42,
                n_estimators=2000,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.2,
                min_child_weight=3,
                gamma=0.1,
                n_jobs=1
            )
        
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric=['logloss', 'auc'],
            early_stopping_rounds=100,
            verbose=50
        )
        
        xgb_train_time = time.time() - xgb_start_time
        print(f"{model_name} XGBoost Training Time: {xgb_train_time:.2f} seconds")
        
        # 评估XGBoost模型
        xgb_y_pred = xgb_model.predict(X_test)
        print(f"\n{model_name} XGBoost Performance:")
        print(classification_report(y_test, xgb_y_pred))
        print(f"Accuracy: {accuracy_score(y_test, xgb_y_pred):.4f}")
        print(f"F1-Score: {f1_score(y_test, xgb_y_pred):.4f}")
        
        # 构建软投票集成模型，动态调整estimators和权重
        print(f"\nBuilding Enhanced Soft Voting Ensemble for {model_name}...")
        
        # 评估各个模型的性能，用于确定投票权重
        models = {
            'lgb': lgb_model,
            'rf': rf_model,
            'xgb': xgb_model
        }
        
        # 计算各模型的F1分数
        model_scores = {}
        for name, model in models.items():
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            model_scores[name] = f1
            print(f"{name} F1-Score: {f1:.4f}")
        
        # 只对draw_model添加ExtraTrees
        if is_draw_model:
            # 训练ExtraTreesClassifier，专门针对平局模型
            print(f"\nTraining Extra Trees for {model_name}...")
            et_start_time = time.time()
            
            et_model = ExtraTreesClassifier(
                n_estimators=2000,  # 减少树数量，提高效率
                max_depth=15,        # 减少深度，避免过拟合
                min_samples_split=4,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=1,
                class_weight='balanced',
                criterion='entropy'
            )
            et_model.fit(X_train, y_train)
            
            et_train_time = time.time() - et_start_time
            print(f"{model_name} Extra Trees Training Time: {et_train_time:.2f} seconds")
            
            # 评估ExtraTrees模型
            et_y_pred = et_model.predict(X_test)
            print(f"\n{model_name} Extra Trees Performance:")
            print(classification_report(y_test, et_y_pred))
            et_f1 = f1_score(y_test, et_y_pred)
            print(f"Accuracy: {accuracy_score(y_test, et_y_pred):.4f}")
            print(f"F1-Score: {et_f1:.4f}")
            
            models['et'] = et_model
            model_scores['et'] = et_f1
        
        # 计算模型权重，基于F1分数
        total_score = sum(model_scores.values())
        model_weights = {name: score / total_score for name, score in model_scores.items()}
        
        # 创建estimators列表，包含权重
        estimators = [(name, model) for name, model in models.items()]
        
        # 使用加权软投票
        voting_model = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=list(model_weights.values()),
            n_jobs=1
        )
        
        # 训练软投票模型
        voting_start_time = time.time()
        voting_model.fit(X_train, y_train)
        voting_train_time = time.time() - voting_start_time
        print(f"{model_name} Voting Ensemble Training Time: {voting_train_time:.2f} seconds")
        
        # 评估软投票模型
        voting_y_pred = voting_model.predict(X_test)
        print(f"\n{model_name} Soft Voting Ensemble Performance:")
        print(classification_report(y_test, voting_y_pred))
        print(f"Accuracy: {accuracy_score(y_test, voting_y_pred):.4f}")
        print(f"F1-Score: {f1_score(y_test, voting_y_pred):.4f}")
        print(f"Recall: {recall_score(y_test, voting_y_pred):.4f}")
        
        return voting_model
    
    def _combine_predictions(self, X_test_scaled, draw_model, win_loss_model):
        """组合两个子模型的预测结果"""
        # 平/非平模型预测 - 获取平局和非平局的概率
        y_pred_draw_proba_full = draw_model.predict_proba(X_test_scaled)  # 输出: [non_draw_prob, draw_prob]
        
        # 胜负模型预测 - 获取主胜和客胜的概率
        y_pred_win_loss_proba_full = win_loss_model.predict_proba(X_test_scaled)  # 输出: [away_win_prob, home_win_prob]
        
        # 组合预测结果
        y_pred_hybrid = []
        y_pred_proba_hybrid = []
        
        for i in range(len(X_test_scaled)):
            # 获取各概率
            draw_prob = y_pred_draw_proba_full[i, 1]
            non_draw_prob = y_pred_draw_proba_full[i, 0]
            home_win_prob_cond = y_pred_win_loss_proba_full[i, 1]  # 条件概率: 主胜|非平局
            away_win_prob_cond = y_pred_win_loss_proba_full[i, 0]  # 条件概率: 客胜|非平局
            
            # 计算联合概率
            home_win_prob = non_draw_prob * home_win_prob_cond
            away_win_prob = non_draw_prob * away_win_prob_cond
            
            # 归一化概率，确保总和为1
            total_prob = home_win_prob + draw_prob + away_win_prob
            if total_prob > 0:
                home_win_prob /= total_prob
                draw_prob /= total_prob
                away_win_prob /= total_prob
            else:
                # 处理极端情况
                home_win_prob = 1/3
                draw_prob = 1/3
                away_win_prob = 1/3
            
            # 根据概率选择预测结果
            if draw_prob >= self.draw_threshold:
                y_pred_hybrid.append(1)  # 平局
            else:
                if home_win_prob_cond >= self.home_win_threshold:
                    y_pred_hybrid.append(2)  # 主胜
                else:
                    y_pred_hybrid.append(0)  # 客胜
            
            # 保存真实概率分布
            y_pred_proba_hybrid.append([away_win_prob, draw_prob, home_win_prob])
        
        return np.array(y_pred_hybrid), np.array(y_pred_proba_hybrid)
    
    def _output_predictions(self, X_test, y_true, y_pred, y_pred_proba, season_key):
        """输出测试集预测结果，包含概率比较和投资建议"""
        # 获取原始特征数据，包含赔率信息
        from ..data_loader import DataLoader
        
        # 保存预测结果到CSV文件
        pred_df = pd.DataFrame({
            'True_Label': y_true,
            'Predicted_Label': y_pred,
            'Predicted_Proba_AwayWin': y_pred_proba[:, 0],
            'Predicted_Proba_Draw': y_pred_proba[:, 1],
            'Predicted_Proba_HomeWin': y_pred_proba[:, 2]
        })
        
        # 计算赔率隐含概率和模型优势
        if hasattr(self.data_loader, 'X_full'):
            # 获取原始数据，包含赔率信息
            X_full = self.data_loader.X_full
            
            # 确保X_full和X_test的索引对应
            if len(X_full) >= len(X_test):
                # 获取测试集对应的原始数据
                test_indices = X_full.index[-len(X_test):] if len(X_full) > len(X_test) else X_full.index
                X_test_full = X_full.loc[test_indices]
                
                # 计算赔率隐含概率
                if 'closing_win_odds' in X_test_full.columns and 'closing_draw_odds' in X_test_full.columns and 'closing_lose_odds' in X_test_full.columns:
                    # 计算隐含概率（考虑赔付率）
                    win_odds = X_test_full['closing_win_odds'].values
                    draw_odds = X_test_full['closing_draw_odds'].values
                    lose_odds = X_test_full['closing_lose_odds'].values
                    payout_rate = X_test_full['closing_payout_rate'].values if 'closing_payout_rate' in X_test_full.columns else 1.0
                    
                    # 计算隐含概率
                    implied_home_prob = (1 / win_odds) * payout_rate
                    implied_draw_prob = (1 / draw_odds) * payout_rate
                    implied_away_prob = (1 / lose_odds) * payout_rate
                    
                    # 计算模型优势（模型概率 - 隐含概率）
                    pred_df['Implied_Proba_HomeWin'] = implied_home_prob
                    pred_df['Implied_Proba_Draw'] = implied_draw_prob
                    pred_df['Implied_Proba_AwayWin'] = implied_away_prob
                    
                    # 计算模型相对于赔率的优势
                    pred_df['HomeWin_Edge'] = pred_df['Predicted_Proba_HomeWin'] - pred_df['Implied_Proba_HomeWin']
                    pred_df['Draw_Edge'] = pred_df['Predicted_Proba_Draw'] - pred_df['Implied_Proba_Draw']
                    pred_df['AwayWin_Edge'] = pred_df['Predicted_Proba_AwayWin'] - pred_df['Implied_Proba_AwayWin']
                    
                    # 计算置信度（模型概率与次高概率的差值）
                    pred_df['Confidence'] = pred_df[['Predicted_Proba_AwayWin', 'Predicted_Proba_Draw', 'Predicted_Proba_HomeWin']].apply(
                        lambda x: x.max() - x.nlargest(2).min(), axis=1
                    )
                    
                    # 标记有投资价值的比赛（模型优势 > 0.05且置信度 > 0.1）
                    pred_df['Has_Value'] = (pred_df[['HomeWin_Edge', 'Draw_Edge', 'AwayWin_Edge']].max(axis=1) > 0.05) & (pred_df['Confidence'] > 0.1)
                    
                    # 推荐结果
                    def get_recommendation(row):
                        edges = [row['AwayWin_Edge'], row['Draw_Edge'], row['HomeWin_Edge']]
                        max_edge = max(edges)
                        if max_edge > 0.05:
                            pred_probas = [row['Predicted_Proba_AwayWin'], row['Predicted_Proba_Draw'], row['Predicted_Proba_HomeWin']]
                            pred_max = max(pred_probas)
                            if pred_probas[0] == pred_max:
                                return 'Away Win'
                            elif pred_probas[1] == pred_max:
                                return 'Draw'
                            else:
                                return 'Home Win'
                        return 'No Value'
                    
                    pred_df['Recommendation'] = pred_df.apply(get_recommendation, axis=1)
                    
                    # 计算预期收益率
                    def calculate_roi(row):
                        if row['Recommendation'] == 'Home Win':
                            return row['HomeWin_Edge'] * (row['closing_win_odds'] - 1)
                        elif row['Recommendation'] == 'Draw':
                            return row['Draw_Edge'] * (row['closing_draw_odds'] - 1)
                        elif row['Recommendation'] == 'Away Win':
                            return row['AwayWin_Edge'] * (row['closing_lose_odds'] - 1)
                        return 0.0
                    
                    if 'closing_win_odds' in X_test_full.columns:
                        pred_df = pred_df.join(X_test_full[['closing_win_odds', 'closing_draw_odds', 'closing_lose_odds']].reset_index(drop=True))
                        pred_df['Expected_ROI'] = pred_df.apply(calculate_roi, axis=1)
        
        pred_save_path = os.path.join(self.model_save_dir, f"predictions_{season_key}.csv")
        pred_df.to_csv(pred_save_path, index=False)
        
        print(f"\nPredictions saved to: {pred_save_path}")
        print("Sample Predictions:")
        print(pred_df.head(20))
        
        # 输出投资价值统计
        if 'Has_Value' in pred_df.columns:
            value_games = pred_df[pred_df['Has_Value'] == True]
            print(f"\n=== 投资价值统计 ===")
            print(f"总比赛数: {len(pred_df)}")
            print(f"有投资价值的比赛数: {len(value_games)}")
            print(f"占比: {len(value_games) / len(pred_df):.2%}")
            
            if len(value_games) > 0:
                print(f"平均预期收益率: {value_games['Expected_ROI'].mean():.2%}")
                print(f"正预期收益比赛数: {len(value_games[value_games['Expected_ROI'] > 0])}")
                
                # 按推荐类型统计
                print(f"\n推荐类型分布:")
                print(value_games['Recommendation'].value_counts())
                
                # 输出最佳投资机会
                top_roi_games = value_games.nlargest(10, 'Expected_ROI')
                print(f"\n=== 最佳10个投资机会 ===")
                print(top_roi_games[['Recommendation', 'Predicted_Proba_HomeWin', 'Predicted_Proba_Draw', 'Predicted_Proba_AwayWin', 
                                    'Implied_Proba_HomeWin', 'Implied_Proba_Draw', 'Implied_Proba_AwayWin',
                                    'HomeWin_Edge', 'Draw_Edge', 'AwayWin_Edge', 'Confidence', 'Expected_ROI',
                                    'closing_win_odds', 'closing_draw_odds', 'closing_lose_odds']])
        
        return pred_df
    
    def predict(self, X, draw_model, win_loss_model, scaler, features):
        """预测新数据"""
        # Select features
        X_selected = X[features]
        
        # Scale features
        X_scaled = scaler.transform(X_selected)
        
        # 组合预测
        y_pred, y_pred_proba = self._combine_predictions(X_scaled, draw_model, win_loss_model)
        
        return y_pred, y_pred_proba
    
    def tune_thresholds(self, X_test_scaled, y_test, draw_model, win_loss_model, draw_thresholds=None, win_loss_thresholds=None):
        """寻找最佳阈值组合，最大化准确率
        
        Args:
            X_test_scaled: 缩放后的测试特征
            y_test: 测试集真实标签
            draw_model: 平局/非平局模型
            win_loss_model: 主胜/客胜模型
            draw_thresholds: 平局模型的阈值候选列表
            win_loss_thresholds: 主胜模型的阈值候选列表
            
        Returns:
            best_thresholds: 最佳阈值组合
            best_metrics: 最佳性能指标
        """
        # 设置更细粒度的阈值范围，更关注较低的平局阈值以提高召回率
        if draw_thresholds is None:
            draw_thresholds = [round(i * 0.01, 2) for i in range(10, 35)]  # 0.10到0.35，更关注较低的平局概率
        
        if win_loss_thresholds is None:
            win_loss_thresholds = [round(i * 0.02, 2) for i in range(15, 35)]  # 0.30到0.70，更关注主胜概率
        
        best_accuracy = 0
        best_thresholds = (self.draw_threshold, 0.5)  # 默认阈值
        best_metrics = None
        
        print("\n=== 开始阈值调优 ===")
        print(f"平局模型阈值范围: {draw_thresholds}")
        print(f"主胜模型阈值范围: {win_loss_thresholds}")
        print(f"测试样本数: {len(y_test)}")
        
        # 遍历所有阈值组合
        for draw_thresh in draw_thresholds:
            for win_loss_thresh in win_loss_thresholds:
                # 使用当前阈值组合进行预测
                y_pred, _ = self._combine_predictions_with_thresholds(
                    X_test_scaled, draw_model, win_loss_model, draw_thresh, win_loss_thresh
                )
                
                # 计算性能指标
                accuracy = accuracy_score(y_test, y_pred)
                weighted_f1 = f1_score(y_test, y_pred, average='weighted')
                f1_scores = f1_score(y_test, y_pred, average=None)
                home_win_f1 = f1_scores[2]  # 假设标签顺序是0:客胜, 1:平局, 2:主胜
                draw_f1 = f1_scores[1]
                away_win_f1 = f1_scores[0]
                
                # 记录最佳阈值，优先考虑准确率
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_thresholds = (draw_thresh, win_loss_thresh)
                    best_metrics = {
                        'accuracy': accuracy,
                        'weighted_f1': weighted_f1,
                        'home_win_f1': home_win_f1,
                        'draw_f1': draw_f1,
                        'away_win_f1': away_win_f1
                    }
                    
                    # 打印最佳阈值更新
                    print(f"\n找到新的最佳阈值组合:")
                    print(f"平局阈值: {draw_thresh}, 主胜阈值: {win_loss_thresh}")
                    print(f"准确率: {accuracy:.4f}, 加权F1: {weighted_f1:.4f}")
                    print(f"主胜F1: {home_win_f1:.4f}, 平局F1: {draw_f1:.4f}, 客胜F1: {away_win_f1:.4f}")
        
        print("\n=== 阈值调优完成 ===")
        print(f"最佳阈值组合: 平局阈值={best_thresholds[0]}, 主胜阈值={best_thresholds[1]}")
        print(f"最佳性能指标:")
        print(f"  准确率: {best_metrics['accuracy']:.4f}")
        print(f"  加权F1: {best_metrics['weighted_f1']:.4f}")
        print(f"  主胜F1: {best_metrics['home_win_f1']:.4f}")
        print(f"  平局F1: {best_metrics['draw_f1']:.4f}")
        print(f"  客胜F1: {best_metrics['away_win_f1']:.4f}")
        
        return best_thresholds, best_metrics
    
    def _combine_predictions_with_thresholds(self, X_test_scaled, draw_model, win_loss_model, draw_threshold, win_loss_threshold):
        """使用指定阈值组合预测"""
        # 平/非平模型预测 - 获取平局和非平局的概率
        y_pred_draw_proba_full = draw_model.predict_proba(X_test_scaled)  # 输出: [non_draw_prob, draw_prob]
        
        # 胜负模型预测 - 获取主胜和客胜的概率
        y_pred_win_loss_proba_full = win_loss_model.predict_proba(X_test_scaled)  # 输出: [away_win_prob, home_win_prob]
        
        # 组合预测结果
        y_pred_hybrid = []
        y_pred_proba_hybrid = []
        
        for i in range(len(X_test_scaled)):
            # 获取各概率
            draw_prob = y_pred_draw_proba_full[i, 1]
            non_draw_prob = y_pred_draw_proba_full[i, 0]
            home_win_prob_cond = y_pred_win_loss_proba_full[i, 1]  # 条件概率: 主胜|非平局
            away_win_prob_cond = y_pred_win_loss_proba_full[i, 0]  # 条件概率: 客胜|非平局
            
            # 计算联合概率
            home_win_prob = non_draw_prob * home_win_prob_cond
            away_win_prob = non_draw_prob * away_win_prob_cond
            
            # 归一化概率，确保总和为1
            total_prob = home_win_prob + draw_prob + away_win_prob
            if total_prob > 0:
                home_win_prob /= total_prob
                draw_prob /= total_prob
                away_win_prob /= total_prob
            else:
                # 处理极端情况
                home_win_prob = 1/3
                draw_prob = 1/3
                away_win_prob = 1/3
            
            # 根据概率选择预测结果
            if draw_prob >= draw_threshold:
                y_pred_hybrid.append(1)  # 平局
            else:
                if home_win_prob_cond >= win_loss_threshold:
                    y_pred_hybrid.append(2)  # 主胜
                else:
                    y_pred_hybrid.append(0)  # 客胜
            
            # 保存真实概率分布
            y_pred_proba_hybrid.append([away_win_prob, draw_prob, home_win_prob])
        
        return np.array(y_pred_hybrid), np.array(y_pred_proba_hybrid)
    
    def _tune_binary_hyperparameters(self, X_train, y_train, model_name):
        """使用随机搜索进行二元分类模型的超参数调优，特别优化平局模型"""
        print("\n=== Hyperparameter Tuning ===")
        
        is_draw_model = model_name == "draw_model"
        is_win_loss_model = model_name == "win_loss_model"
        
        # 为不同模型定义不同的参数搜索空间
        if is_draw_model:
            # 平局模型参数空间 - 针对平局预测进行优化
            param_dist = {
                'n_estimators': [500, 1000, 1500, 2000],  # 增加树数量
                'learning_rate': [0.01, 0.03, 0.05],  # 降低学习率，增加稳定性
                'max_depth': [6, 8, 10, 12],  # 增加深度，捕捉更多复杂模式
                'num_leaves': [63, 127, 255, 512],  # 增加叶子节点数
                'subsample': [0.6, 0.7, 0.8],
                'colsample_bytree': [0.6, 0.7, 0.8],
                'reg_alpha': [0.01, 0.05, 0.1],  # 减少正则化，允许模型更灵活
                'reg_lambda': [0.01, 0.05, 0.1],
                'min_split_gain': [0.001, 0.01, 0.05],
                'min_child_samples': [10, 20, 30],  # 减少最小样本数，允许更多叶子节点
                'scale_pos_weight': [1.0, 1.5, 2.0, 2.5]  # 添加类别权重参数
            }
            n_iter = 50  # 增加迭代次数，更全面搜索
        elif is_win_loss_model:
            # 胜负模型参数空间 - 基于最佳参数缩小搜索范围
            param_dist = {
                'n_estimators': [800, 1000, 1200],
                'learning_rate': [0.03, 0.04, 0.05],
                'max_depth': [5, 6, 7],
                'num_leaves': [31, 47, 63],
                'subsample': [0.75, 0.8, 0.85],
                'colsample_bytree': [0.75, 0.8, 0.85],
                'reg_alpha': [0.1, 0.15, 0.2],
                'reg_lambda': [0.1, 0.15, 0.2],
                'min_split_gain': [0.01, 0.02, 0.03],
                'min_child_samples': [22, 25, 28]
            }
            n_iter = 40  # 减少迭代次数
        else:
            # 默认参数空间
            param_dist = {
                'n_estimators': [300, 500, 700],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [4, 5, 6],
                'num_leaves': [15, 31, 47],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'reg_alpha': [0.05, 0.1, 0.2],
                'reg_lambda': [0.05, 0.1, 0.2],
            }
            n_iter = 30  # 减少迭代次数
        
        # 创建基础模型
        base_model = lgb.LGBMClassifier(
            objective='binary',
            random_state=42,
            n_jobs=1,
            class_weight='balanced'  # 添加类别权重，特别是对平局模型有帮助
        )
        
        # 使用随机搜索，针对平局模型使用recall作为评分指标，提高召回率
        scoring = 'f1' if not is_draw_model else 'recall'
        
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=n_iter,  # 使用调整后的迭代次数
            scoring=scoring,
            cv=5,  # 增加交叉验证折数，提高结果可靠性
            verbose=2,
            random_state=42,
            n_jobs=1
        )
        
        # 执行搜索
        start_time = time.time()
        random_search.fit(X_train, y_train)
        tuning_time = time.time() - start_time
        
        print(f"\nHyperparameter Tuning Time: {tuning_time:.2f} seconds")
        print("Best Parameters:", random_search.best_params_)
        print("Best F1-Score:", random_search.best_score_)
        
        return random_search.best_estimator_, random_search.best_params_
    
    def _tune_rf_hyperparameters(self, X_train, y_train, model_name):
        """使用随机搜索进行随机森林的超参数调优"""
        print(f"\n=== Random Forest Hyperparameter Tuning for {model_name} ===")
        
        is_draw_model = model_name == "draw_model"
        is_win_loss_model = model_name == "win_loss_model"
        
        # 为不同模型定义不同的参数搜索空间
        param_dist = {
            'n_estimators': [500, 800, 1000, 1200],
            'max_depth': [8, 10, 12, 15],
            'min_samples_split': [4, 6, 8, 10],
            'min_samples_leaf': [2, 4, 6, 8],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'class_weight': ['balanced', None] if is_draw_model else [None]
        }
        
        # 减少迭代次数以提高效率
        n_iter = 20
        
        # 创建基础模型
        base_model = RandomForestClassifier(
            random_state=42,
            n_jobs=1
        )
        
        # 使用随机搜索
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='f1',
            cv=3,
            verbose=2,
            random_state=42,
            n_jobs=1
        )
        
        # 执行搜索
        start_time = time.time()
        random_search.fit(X_train, y_train)
        tuning_time = time.time() - start_time
        
        print(f"\nRandom Forest Hyperparameter Tuning Time: {tuning_time:.2f} seconds")
        print("Best Parameters:", random_search.best_params_)
        print("Best F1-Score:", random_search.best_score_)
        
        return random_search.best_estimator_, random_search.best_params_
    
    def _tune_hyperparameters(self, X_train_scaled, y_train, X_test_scaled, y_test, 
                             X_train_draw_scaled, y_train_draw_binary_balanced,
                             X_train_win_loss_scaled, y_train_win_loss_balanced,
                             y_test_draw_binary, y_test_win_loss):
        """使用最终加权F1分数作为目标的超参数调优，包括LightGBM和随机森林的集成"""
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        
        print(f"\n=== 开始整体超参数调优 ===")
        
        # 定义LightGBM参数搜索空间
        lgb_draw_param_grid = {
            'n_estimators': [300, 500, 700],
            'learning_rate': [0.05, 0.1, 0.12],
            'max_depth': [4, 5, 6],
            'num_leaves': [15, 31, 47],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        lgb_win_loss_param_grid = {
            'n_estimators': [800, 1000, 1200],
            'learning_rate': [0.03, 0.05, 0.07],
            'max_depth': [5, 6, 7],
            'num_leaves': [31, 47, 63],
            'subsample': [0.75, 0.8, 0.85],
            'colsample_bytree': [0.75, 0.8, 0.85]
        }
        
        # 定义随机森林参数搜索空间
        rf_draw_param_grid = {
            'n_estimators': [500, 800, 1000],
            'max_depth': [8, 10, 12],
            'min_samples_split': [4, 6, 8],
            'min_samples_leaf': [2, 4, 6],
            'max_features': ['sqrt', 'log2']
        }
        
        rf_win_loss_param_grid = {
            'n_estimators': [800, 1000, 1200],
            'max_depth': [10, 12, 15],
            'min_samples_split': [6, 8, 10],
            'min_samples_leaf': [3, 4, 5],
            'max_features': ['sqrt', 'log2']
        }
        
        # 初始化最佳参数和最佳分数
        best_weighted_f1 = 0
        best_draw_ensemble = None
        best_win_loss_ensemble = None
        best_params = None
        
        # 随机选择组合数
        import random
        n_trials = 15
        print(f"将进行 {n_trials} 轮超参数搜索")
        
        for trial in range(n_trials):
            print(f"\n=== 第 {trial+1}/{n_trials} 轮超参数搜索 ===")
            
            # 随机选择LightGBM平局模型参数
            lgb_draw_params = {
                'n_estimators': random.choice(lgb_draw_param_grid['n_estimators']),
                'learning_rate': random.choice(lgb_draw_param_grid['learning_rate']),
                'max_depth': random.choice(lgb_draw_param_grid['max_depth']),
                'num_leaves': random.choice(lgb_draw_param_grid['num_leaves']),
                'subsample': random.choice(lgb_draw_param_grid['subsample']),
                'colsample_bytree': random.choice(lgb_draw_param_grid['colsample_bytree']),
                'random_state': 42,
                'n_jobs': 1
            }
            
            # 随机选择LightGBM胜负模型参数
            lgb_win_loss_params = {
                'n_estimators': random.choice(lgb_win_loss_param_grid['n_estimators']),
                'learning_rate': random.choice(lgb_win_loss_param_grid['learning_rate']),
                'max_depth': random.choice(lgb_win_loss_param_grid['max_depth']),
                'num_leaves': random.choice(lgb_win_loss_param_grid['num_leaves']),
                'subsample': random.choice(lgb_win_loss_param_grid['subsample']),
                'colsample_bytree': random.choice(lgb_win_loss_param_grid['colsample_bytree']),
                'random_state': 42,
                'n_jobs': 1
            }
            
            # 随机选择随机森林平局模型参数
            rf_draw_params = {
                'n_estimators': random.choice(rf_draw_param_grid['n_estimators']),
                'max_depth': random.choice(rf_draw_param_grid['max_depth']),
                'min_samples_split': random.choice(rf_draw_param_grid['min_samples_split']),
                'min_samples_leaf': random.choice(rf_draw_param_grid['min_samples_leaf']),
                'max_features': random.choice(rf_draw_param_grid['max_features']),
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': 1
            }
            
            # 随机选择随机森林胜负模型参数
            rf_win_loss_params = {
                'n_estimators': random.choice(rf_win_loss_param_grid['n_estimators']),
                'max_depth': random.choice(rf_win_loss_param_grid['max_depth']),
                'min_samples_split': random.choice(rf_win_loss_param_grid['min_samples_split']),
                'min_samples_leaf': random.choice(rf_win_loss_param_grid['min_samples_leaf']),
                'max_features': random.choice(rf_win_loss_param_grid['max_features']),
                'random_state': 42,
                'n_jobs': 1
            }
            
            try:
                # 训练LightGBM平局模型
                lgb_draw_model = lgb.LGBMClassifier(**lgb_draw_params)
                lgb_draw_model.fit(X_train_draw_scaled, y_train_draw_binary_balanced)
                
                # 训练随机森林平局模型
                rf_draw_model = RandomForestClassifier(**rf_draw_params)
                rf_draw_model.fit(X_train_draw_scaled, y_train_draw_binary_balanced)
                
                # 构建平局模型的软投票集成
                draw_ensemble = VotingClassifier(
                    estimators=[
                        ('lgb', lgb_draw_model),
                        ('rf', rf_draw_model)
                    ],
                    voting='soft',
                    n_jobs=1
                )
                draw_ensemble.fit(X_train_draw_scaled, y_train_draw_binary_balanced)
                
                # 训练LightGBM胜负模型
                lgb_win_loss_model = lgb.LGBMClassifier(**lgb_win_loss_params)
                lgb_win_loss_model.fit(X_train_win_loss_scaled, y_train_win_loss_balanced)
                
                # 训练随机森林胜负模型
                rf_win_loss_model = RandomForestClassifier(**rf_win_loss_params)
                rf_win_loss_model.fit(X_train_win_loss_scaled, y_train_win_loss_balanced)
                
                # 构建胜负模型的软投票集成
                win_loss_ensemble = VotingClassifier(
                    estimators=[
                        ('lgb', lgb_win_loss_model),
                        ('rf', rf_win_loss_model)
                    ],
                    voting='soft',
                    n_jobs=1
                )
                win_loss_ensemble.fit(X_train_win_loss_scaled, y_train_win_loss_balanced)
                
                # 预测并计算加权F1分数
                # 1. 预测平局概率
                draw_proba = draw_ensemble.predict_proba(X_test_scaled)[:, 1]
                
                # 2. 预测胜负概率
                win_loss_proba = win_loss_ensemble.predict_proba(X_test_scaled)[:, 1]
                
                # 3. 组合预测
                y_pred = []
                for dp, wp in zip(draw_proba, win_loss_proba):
                    if dp >= self.draw_threshold:
                        y_pred.append(1)  # 平局
                    else:
                        if wp >= self.home_win_threshold:
                            y_pred.append(2)  # 主胜
                        else:
                            y_pred.append(0)  # 客胜
                
                y_pred = np.array(y_pred)
                
                # 计算加权F1分数
                weighted_f1 = f1_score(y_test, y_pred, average='weighted')
                
                print(f"LightGBM Draw Params: {lgb_draw_params}")
                print(f"Random Forest Draw Params: {rf_draw_params}")
                print(f"LightGBM Win/Loss Params: {lgb_win_loss_params}")
                print(f"Random Forest Win/Loss Params: {rf_win_loss_params}")
                print(f"Weighted F1: {weighted_f1:.4f}")
                
                # 更新最佳参数
                if weighted_f1 > best_weighted_f1:
                    best_weighted_f1 = weighted_f1
                    best_draw_ensemble = draw_ensemble
                    best_win_loss_ensemble = win_loss_ensemble
                    best_params = {
                        'lgb_draw': lgb_draw_params,
                        'rf_draw': rf_draw_params,
                        'lgb_win_loss': lgb_win_loss_params,
                        'rf_win_loss': rf_win_loss_params
                    }
                    print(f"=== 找到更好的参数组合！加权F1: {best_weighted_f1:.4f} ===")
            except Exception as e:
                print(f"训练过程中出现错误: {e}")
                continue
        
        print(f"\n=== 超参数调优完成 ===")
        print(f"最佳加权F1: {best_weighted_f1:.4f}")
        if best_params:
            print(f"最佳LightGBM平局模型参数: {best_params['lgb_draw']}")
            print(f"最佳随机森林平局模型参数: {best_params['rf_draw']}")
            print(f"最佳LightGBM胜负模型参数: {best_params['lgb_win_loss']}")
            print(f"最佳随机森林胜负模型参数: {best_params['rf_win_loss']}")
        
        return best_draw_ensemble, best_win_loss_ensemble
    
    def load_model(self, season_key):
        """加载混合模型"""
        draw_model_path = os.path.join(self.model_save_dir, f"draw_model_{season_key}.joblib")
        win_loss_model_path = os.path.join(self.model_save_dir, f"win_loss_model_{season_key}.joblib")
        scaler_path = os.path.join(self.model_save_dir, f"scaler_{season_key}.joblib")
        features_path = os.path.join(self.model_save_dir, f"features_{season_key}.joblib")
        
        if not all([os.path.exists(p) for p in [draw_model_path, win_loss_model_path, scaler_path, features_path]]):
            raise FileNotFoundError(f"Model files for season {season_key} not found")
        
        draw_model = joblib.load(draw_model_path)
        win_loss_model = joblib.load(win_loss_model_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)
        
        return draw_model, win_loss_model, scaler, features