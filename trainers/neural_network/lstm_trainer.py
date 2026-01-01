import os
import numpy as np
import pandas as pd
import json
import time
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from ..model_trainer import BaseModelTrainer
from ..data_loader import DataLoader

class LSTMTrainer(BaseModelTrainer):
    def __init__(self, data_root, model_dir):
        super().__init__(data_root, model_dir)
        self.version = "4.0.1"
        self.model_save_dir = os.path.join(model_dir, "v4", "4.0.1")
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # 从best_params.json加载最佳配置
        self.best_params_path = os.path.join(os.path.dirname(__file__), "best_params.json")
        if os.path.exists(self.best_params_path):
            with open(self.best_params_path, 'r') as f:
                self.best_params = json.load(f)
        else:
            # 默认参数
            self.best_params = {
                'draw_threshold': 0.25,
                'home_win_threshold': 0.45,
                'lstm_params': {
                    'input_shape': (5, 80),  # (时间步, 特征数)
                    'lstm_units': 64,
                    'dropout_rate': 0.3,
                    'batch_size': 32,
                    'epochs': 100,
                    'learning_rate': 0.001
                }
            }
        
        # 使用best_params中的阈值
        self.draw_threshold = self.best_params['draw_threshold']
        self.home_win_threshold = self.best_params['home_win_threshold']
        
        print(f"Loaded best parameters from {self.best_params_path}")
        print(f"Initial draw threshold: {self.draw_threshold}")
        print(f"Initial home win threshold: {self.home_win_threshold}")
    
    def _build_lstm_model(self, input_shape):
        """构建更适合表格数据的模型，结合注意力机制和全连接层"""
        from keras import Input, Model
        from keras.layers import Layer, Dense, Dropout, BatchNormalization, Reshape, MultiHeadAttention, LayerNormalization
        
        # 使用Transformer-style模型，更适合处理表格序列数据
        inputs = Input(shape=input_shape)
        
        # 展开序列，添加位置编码
        x = Reshape((input_shape[0], input_shape[1]))(inputs)
        
        # 添加注意力层，学习特征间的依赖关系
        attention_output = MultiHeadAttention(
            num_heads=4, 
            key_dim=32,
            dropout=0.2
        )(x, x, x)
        attention_output = LayerNormalization(epsilon=1e-6)(attention_output + x)
        
        # 池化层，减少序列维度
        from keras.layers import GlobalAveragePooling1D
        x = GlobalAveragePooling1D()(attention_output)
        
        # 全连接层
        x = Dense(128, activation='relu', kernel_regularizer='l2')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        
        x = Dense(64, activation='relu', kernel_regularizer='l2')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        
        # 输出层 - 3分类（客胜、平局、主胜）
        outputs = Dense(3, activation='softmax')(x)
        
        # 编译模型
        from keras.optimizers import AdamW
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = AdamW(learning_rate=0.001, weight_decay=1e-5)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_sequence_data(self, X, y, time_steps=5):
        """将数据转换为LSTM需要的序列格式
        
        Args:
            X: 输入特征，形状为 [样本数, 特征数]
            y: 标签，形状为 [样本数]
            time_steps: 时间步长
            
        Returns:
            X_seq: 序列数据，形状为 [样本数 - time_steps + 1, time_steps, 特征数]
            y_seq: 对应的标签，形状为 [样本数 - time_steps + 1]
        """
        X_seq = []
        y_seq = []
        
        # 将y转换为numpy数组，避免索引问题
        y_np = np.array(y)
        
        for i in range(len(X) - time_steps + 1):
            X_seq.append(X[i:i+time_steps])
            y_seq.append(y_np[i+time_steps-1])  # 使用序列最后一个样本的标签
        
        return np.array(X_seq), np.array(y_seq)
    
    def train(self, seasons=None, include_team_state=True, include_expert=True, use_tuning=False, use_llm_expert=False):
        """训练LSTM模型"""
        # 如果没有指定赛季，使用所有可用赛季
        if seasons is None:
            seasons = ['2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020', 
                      '2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
            print(f"Training LSTM model v{self.version} for all seasons: {seasons}")
        else:
            print(f"Training LSTM model v{self.version} for seasons: {seasons}")
        
        # 如果使用LLM专家分析，强制启用专家特征
        if use_llm_expert:
            include_expert = True
            print("✓ Enabling expert features for LLM expert analysis integration")
        
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
        
        # 创建序列数据，调整时间步长为10，增加历史信息
        time_steps = 10  # 每个序列包含10场比赛的历史数据
        print(f"\n=== Creating Sequence Data with {time_steps} time steps ===")
        
        X_train_seq, y_train_seq = self._create_sequence_data(X_train_scaled, y_train, time_steps)
        X_test_seq, y_test_seq = self._create_sequence_data(X_test_scaled, y_test, time_steps)
        
        print(f"Sequence Train Data Shape: {X_train_seq.shape}")
        print(f"Sequence Test Data Shape: {X_test_seq.shape}")
        print(f"Sequence Label Distribution (Train): {np.unique(y_train_seq, return_counts=True)}")
        
        # 转换标签为one-hot编码
        from keras.utils import to_categorical
        y_train_onehot = to_categorical(y_train_seq, num_classes=3)
        y_test_onehot = to_categorical(y_test_seq, num_classes=3)
        
        # 计算类别权重，解决类别不平衡问题
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_seq),
            y=y_train_seq
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        print(f"Class weights: {class_weight_dict}")
        
        # 构建LSTM模型
        print(f"\n=== Building LSTM Model ===")
        input_shape = (time_steps, X_train_seq.shape[2])  # (时间步, 特征数)
        model = self._build_lstm_model(input_shape)
        print(model.summary())
        
        # 训练模型
        print(f"\n=== Training LSTM Model ===")
        
        # 设置早停和模型保存回调
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,  # 增加耐心值，避免过早停止
            restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            filepath=os.path.join(self.model_save_dir, f'lstm_model_{self.version}_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max'
        )
        
        # 动态调整学习率
        from keras.callbacks import ReduceLROnPlateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # 直接使用标准的categorical_crossentropy，通过class_weight参数在fit中处理不平衡
        
        # 训练，添加类别权重
        history = model.fit(
            X_train_seq, y_train_onehot,
            batch_size=64,  # 增加批量大小，提高稳定性
            epochs=200,  # 增加训练轮数
            validation_data=(X_test_seq, y_test_onehot),
            callbacks=[early_stopping, model_checkpoint, reduce_lr],
            class_weight=class_weight_dict,  # 添加类别权重
            verbose=1
        )
        
        # 加载最佳模型
        best_model = load_model(os.path.join(self.model_save_dir, f'lstm_model_{self.version}_best.keras'))
        
        # 评估模型
        print(f"\n=== Evaluating LSTM Model ===")
        
        # 在测试集上预测
        y_pred_proba = best_model.predict(X_test_seq)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 计算评估指标
        print("LSTM Model Performance:")
        print(classification_report(y_test_seq, y_pred, target_names=['Away Win', 'Draw', 'Home Win']))
        
        accuracy = accuracy_score(y_test_seq, y_pred)
        weighted_f1 = f1_score(y_test_seq, y_pred, average='weighted')
        home_win_f1 = f1_score(y_test_seq, y_pred, average=None)[2]
        draw_f1 = f1_score(y_test_seq, y_pred, average=None)[1]
        away_win_f1 = f1_score(y_test_seq, y_pred, average=None)[0]
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Weighted F1-Score: {weighted_f1:.4f}")
        print(f"Home Win F1-Score: {home_win_f1:.4f}")
        print(f"Draw F1-Score: {draw_f1:.4f}")
        print(f"Away Win F1-Score: {away_win_f1:.4f}")
        
        # 保存完整模型
        season_key = self.version
        
        # 保存完整模型，使用Keras本地格式
        model_path = os.path.join(self.model_save_dir, f'lstm_model_{season_key}.keras')
        best_model.save(model_path)
        
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
            "use_llm_expert": use_llm_expert,
            "draw_threshold": self.draw_threshold,
            "home_win_threshold": getattr(self, 'home_win_threshold', self.home_win_threshold),
            "lstm_params": self.best_params['lstm_params'],
            "train_shape": X_train.shape,
            "test_shape": X_test.shape,
            "sequence_train_shape": X_train_seq.shape,
            "sequence_test_shape": X_test_seq.shape,
            "features": features,
            "metrics": {
                "accuracy": accuracy,
                "weighted_f1": weighted_f1,
                "home_win_f1": home_win_f1,
                "draw_f1": draw_f1,
                "away_win_f1": away_win_f1
            },
            "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": model_path,
            "scaler_path": scaler_save_path,
            "features_path": features_save_path
        }
        
        info_save_path = os.path.join(self.model_save_dir, f"model_info_{season_key}.json")
        with open(info_save_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=4)
        
        print(f"\nModels saved to: {self.model_save_dir}")
        print(f"Model Info: {info_save_path}")
        return model_info
    
    def predict(self, X, model, scaler, features):
        """预测新数据"""
        # 选择特征
        X_selected = X[features]
        
        # Scale features
        X_scaled = scaler.transform(X_selected)
        
        # 创建序列数据，使用与训练相同的时间步长
        time_steps = 10
        X_seq = []
        for i in range(len(X_scaled) - time_steps + 1):
            X_seq.append(X_scaled[i:i+time_steps])
        
        if not X_seq:
            return np.array([]), np.array([])
            
        X_seq = np.array(X_seq)
        
        # 预测
        y_pred_proba = model.predict(X_seq)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        return y_pred, y_pred_proba
