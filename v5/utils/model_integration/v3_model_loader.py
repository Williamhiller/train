import joblib
import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Union


class V3ModelLoader:
    """V3模型加载器，用于加载和使用V3版本的训练模型"""
    
    def __init__(self, model_dir: str):
        """初始化V3模型加载器
        
        Args:
            model_dir: V3模型文件所在目录
        """
        self.model_dir = model_dir
        self.win_loss_model = None
        self.draw_model = None
        self.scaler = None
        self.features = None
        
        # 加载模型
        self.load_models()
    
    def load_models(self):
        """加载V3模型和相关组件"""
        try:
            # 检查模型目录结构，支持3.0.4和3.0.7版本
            model_files = os.listdir(self.model_dir)
            
            # 3.0.4版本模型结构（单一模型文件）
            if "model_v3.0.4_lightgbm.joblib" in model_files:
                # 加载单一模型
                self.model = joblib.load(os.path.join(self.model_dir, "model_v3.0.4_lightgbm.joblib"))
                
                # 加载模型信息获取特征列表
                info_path = os.path.join(self.model_dir, "model_v3.0.4_lightgbm_info.json")
                if os.path.exists(info_path):
                    with open(info_path, 'r') as f:
                        model_info = json.load(f)
                    self.features = model_info.get("feature_names", [])
                else:
                    self.features = []
                
                # 3.0.4版本没有单独的scaler和分层模型
                self.scaler = None
                self.win_loss_model = None
                self.draw_model = None
                self.is_v304 = True
                
                print(f"✓ 成功加载V3.0.4模型：")
                print(f"  - 模型文件: model_v3.0.4_lightgbm.joblib")
                print(f"  - 特征列表: {len(self.features)}个特征")
            
            # 3.0.7版本模型结构（分层模型）
            elif "win_loss_model_3.0.7.joblib" in model_files and "draw_model_3.0.7.joblib" in model_files:
                # 加载模型文件
                self.win_loss_model = joblib.load(os.path.join(self.model_dir, "win_loss_model_3.0.7.joblib"))
                self.draw_model = joblib.load(os.path.join(self.model_dir, "draw_model_3.0.7.joblib"))
                self.scaler = joblib.load(os.path.join(self.model_dir, "scaler_3.0.7.joblib"))
                self.features = joblib.load(os.path.join(self.model_dir, "features_3.0.7.joblib"))
                
                self.model = None
                self.is_v304 = False
                
                print(f"✓ 成功加载V3.0.7模型：")
                print(f"  - 胜负模型: win_loss_model_3.0.7.joblib")
                print(f"  - 平局模型: draw_model_3.0.7.joblib")
                print(f"  - 标准化器: scaler_3.0.7.joblib")
                print(f"  - 特征列表: {len(self.features)}个特征")
            
            else:
                raise ValueError(f"不支持的模型版本，目录中没有找到预期的模型文件: {model_files}")
            
        except Exception as e:
            print(f"✗ 加载V3模型失败: {e}")
            raise
    
    def check_features(self, data: pd.DataFrame) -> bool:
        """检查数据是否包含V3模型所需的所有特征
        
        Args:
            data: 输入数据
        
        Returns:
            bool: 是否包含所有特征
        """
        missing_features = [feat for feat in self.features if feat not in data.columns]
        if missing_features:
            print(f"✗ 缺少V3模型所需的特征: {missing_features}")
            return False
        return True
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """准备V3模型所需的特征
        
        Args:
            data: 输入数据
        
        Returns:
            pd.DataFrame: 处理后的特征数据
        """
        # 确保所有特征存在
        if not self.check_features(data):
            # 尝试填充缺失特征
            for feat in self.features:
                if feat not in data.columns:
                    data[feat] = 0
        
        # 选择V3模型所需的特征
        X = data[self.features].copy()
        
        # 处理缺失值
        X = X.fillna(0)
        
        return X
    
    def predict(self, data: Union[pd.DataFrame, Dict]) -> Dict[str, np.ndarray]:
        """使用V3模型进行预测
        
        Args:
            data: 输入数据，可以是DataFrame或字典
        
        Returns:
            Dict: 包含胜/平/负概率的预测结果
        """
        # 转换数据格式
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # 准备特征
        X = self.prepare_features(data)
        
        if self.is_v304:
            # 3.0.4版本：使用单一模型直接预测
            # 3.0.4版本没有scaler，直接使用原始特征
            proba = self.model.predict_proba(X)
            
            # 3.0.4版本的预测结果顺序是：客胜、平局、主胜
            away_win_proba = proba[:, 0]
            draw_proba = proba[:, 1]
            home_win_proba = proba[:, 2]
        else:
            # 3.0.7版本：使用分层模型预测
            # 标准化数据
            X_scaled = self.scaler.transform(X)
            
            # 使用胜负模型预测
            win_loss_proba = self.win_loss_model.predict_proba(X_scaled)
            
            # 使用平局模型预测
            draw_proba = self.draw_model.predict_proba(X_scaled)[:, 1]  # 只取平局概率
            
            # 融合结果
            home_win_proba = win_loss_proba[:, 0] * (1 - draw_proba)
            away_win_proba = win_loss_proba[:, 2] * (1 - draw_proba)
            
            # 确保概率和为1
            total_proba = home_win_proba + draw_proba + away_win_proba
            home_win_proba /= total_proba
            draw_proba /= total_proba
            away_win_proba /= total_proba
        
        return {
            "home_win_proba": home_win_proba,
            "draw_proba": draw_proba,
            "away_win_proba": away_win_proba
        }
    
    def predict_single(self, data: Dict) -> Dict[str, float]:
        """预测单个样本
        
        Args:
            data: 单个样本的数据字典
        
        Returns:
            Dict: 包含胜/平/负概率的预测结果
        """
        results = self.predict(data)
        return {
            "home_win_proba": float(results["home_win_proba"][0]),
            "draw_proba": float(results["draw_proba"][0]),
            "away_win_proba": float(results["away_win_proba"][0])
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取V3模型的特征重要性
        
        Returns:
            Dict: 特征重要性字典
        """
        if hasattr(self.win_loss_model, 'feature_importances_'):
            return dict(zip(self.features, self.win_loss_model.feature_importances_))
        return {}
    
    def get_features(self) -> List[str]:
        """获取V3模型使用的特征列表
        
        Returns:
            List[str]: 特征列表
        """
        return self.features.copy()