import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from datetime import datetime
import json
import os


class FeatureEngineer:
    """特征工程类，用于创建和优化特征"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.feature_importance = {}
        
        # 加载V3版本的特征提取器
        self.odds_feature_path = "/Users/Williamhiler/Documents/my-project/train/utils/feature/odds_feature.py"
        self.team_state_feature_path = "/Users/Williamhiler/Documents/my-project/train/utils/feature/team_state_feature.py"
        
        # 初始化V3版本的特征提取器
        self._init_v3_feature_extractors()
        
    def _init_v3_feature_extractors(self):
        """初始化V3版本的特征提取器"""
        # 动态导入V3版本的特征提取器
        import sys
        sys.path.append("/Users/Williamhiler/Documents/my-project/train/utils/feature")
        
        try:
            from odds_feature import OddsFeatureExtractor
            from team_state_feature import TeamStateFeatureExtractor
            
            # 注意：暂时跳过V3特征提取器的初始化，避免大量数据加载
            self.odds_extractor = None
            self.team_state_extractor = None
        except ImportError as e:
            print(f"警告: 无法加载V3版本特征提取器: {e}")
            self.odds_extractor = None
            self.team_state_extractor = None
    
    def calculate_payout_rate(self, win_odds, draw_odds, lose_odds):
        """计算赔付率"""
        try:
            win_odds = float(win_odds)
            draw_odds = float(draw_odds)
            lose_odds = float(lose_odds)
            
            if win_odds <= 0 or draw_odds <= 0 or lose_odds <= 0:
                return 0.0
            
            payout_rate = 1 / (1/win_odds + 1/draw_odds + 1/lose_odds)
            return round(payout_rate, 4)
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    def calculate_kelly_index(self, odds: float, predicted_prob: float) -> float:
        """计算凯利指数"""
        if odds <= 1:
            return 0.0  # 赔率≤1时不适用
        
        kelly = (odds * predicted_prob - 1) / (odds - 1)
        
        # 确保结果在合理范围内
        kelly = max(kelly, 0.0)  # 负值设为0
        kelly = min(kelly, 1.0)  # 最大值不超过1
        
        return kelly
    
    def calculate_implied_probability(self, payout_rate: float, odds: str or float) -> float:
        """计算隐含概率"""
        try:
            odds = float(odds)
        except (ValueError, TypeError):
            return 0.0
        
        if odds <= 0:
            return 0.0
        return (1 / odds) / payout_rate
    
    def create_interaction_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """创建交互特征
        
        Args:
            features: 原始特征DataFrame
            
        Returns:
            包含交互特征的DataFrame
        """
        enhanced_features = features.copy()
        
        # 主客队状态交互特征
        # 检查必需字段是否存在，如果不存在则创建默认值
        if "home_goal_difference" in features.columns and "away_goal_difference" in features.columns:
            enhanced_features["goal_diff_diff"] = features["home_goal_difference"] - features["away_goal_difference"]
        else:
            enhanced_features["goal_diff_diff"] = 0
            
        if "home_form_points" in features.columns and "away_form_points" in features.columns:
            enhanced_features["form_points_diff"] = features["home_form_points"] - features["away_form_points"]
        else:
            enhanced_features["form_points_diff"] = 0
            
        if "home_goals_scored" in features.columns and "away_goals_scored" in features.columns:
            enhanced_features["goals_scored_ratio"] = features["home_goals_scored"] / (features["away_goals_scored"] + 1e-6)
        else:
            enhanced_features["goals_scored_ratio"] = 1.0
            
        if "home_goals_conceded" in features.columns and "away_goals_conceded" in features.columns:
            enhanced_features["goals_conceded_ratio"] = features["home_goals_conceded"] / (features["away_goals_conceded"] + 1e-6)
        else:
            enhanced_features["goals_conceded_ratio"] = 1.0
        
        # 赔率交互特征
        if "home_win_odds" in features.columns and "away_win_odds" in features.columns:
            enhanced_features["odds_home_away_ratio"] = features["home_win_odds"] / (features["away_win_odds"] + 1e-6)
        else:
            enhanced_features["odds_home_away_ratio"] = 1.0
            
        if "draw_odds" in features.columns and "home_win_odds" in features.columns:
            enhanced_features["odds_draw_home_ratio"] = features["draw_odds"] / (features["home_win_odds"] + 1e-6)
        else:
            enhanced_features["odds_draw_home_ratio"] = 1.0
            
        if "draw_odds" in features.columns and "away_win_odds" in features.columns:
            enhanced_features["odds_draw_away_ratio"] = features["draw_odds"] / (features["away_win_odds"] + 1e-6)
        else:
            enhanced_features["odds_draw_away_ratio"] = 1.0
        
        # 计算赔付率和隐含概率
        if all(col in features.columns for col in ["home_win_odds", "draw_odds", "away_win_odds"]):
            # 计算赔付率
            enhanced_features["payout_rate"] = features.apply(
                lambda x: self.calculate_payout_rate(x["home_win_odds"], x["draw_odds"], x["away_win_odds"]), axis=1
            )
            
            # 计算隐含概率
            enhanced_features["implied_prob_home_win"] = features.apply(
                lambda x: self.calculate_implied_probability(
                    self.calculate_payout_rate(x["home_win_odds"], x["draw_odds"], x["away_win_odds"]), 
                    x["home_win_odds"]
                ), axis=1
            )
            enhanced_features["implied_prob_draw"] = features.apply(
                lambda x: self.calculate_implied_probability(
                    self.calculate_payout_rate(x["home_win_odds"], x["draw_odds"], x["away_win_odds"]), 
                    x["draw_odds"]
                ), axis=1
            )
            enhanced_features["implied_prob_away_win"] = features.apply(
                lambda x: self.calculate_implied_probability(
                    self.calculate_payout_rate(x["home_win_odds"], x["draw_odds"], x["away_win_odds"]), 
                    x["away_win_odds"]
                ), axis=1
            )
        
        # 隐含概率交互特征
        if all(col in enhanced_features.columns for col in ["implied_prob_home_win", "implied_prob_away_win"]):
            enhanced_features["implied_prob_diff"] = enhanced_features["implied_prob_home_win"] - enhanced_features["implied_prob_away_win"]
            enhanced_features["implied_prob_sum"] = enhanced_features["implied_prob_home_win"] + enhanced_features["implied_prob_away_win"]
        else:
            # 使用V5版本原有的隐含概率
            if "implied_home_win" in features.columns and "implied_away_win" in features.columns:
                enhanced_features["implied_prob_diff"] = features["implied_home_win"] - features["implied_away_win"]
                enhanced_features["implied_prob_sum"] = features["implied_home_win"] + features["implied_away_win"]
            else:
                enhanced_features["implied_prob_diff"] = 0
                enhanced_features["implied_prob_sum"] = 1.0
        
        # 状态与赔率交互特征
        if "form_points_diff" in enhanced_features.columns and "odds_home_away_ratio" in enhanced_features.columns:
            enhanced_features["form_odds_interaction"] = enhanced_features["form_points_diff"] * enhanced_features["odds_home_away_ratio"]
        else:
            enhanced_features["form_odds_interaction"] = 0
            
        if "goal_diff_diff" in enhanced_features.columns and "implied_prob_diff" in enhanced_features.columns:
            enhanced_features["goal_diff_odds_interaction"] = enhanced_features["goal_diff_diff"] * enhanced_features["implied_prob_diff"]
        else:
            enhanced_features["goal_diff_odds_interaction"] = 0
        
        # 凯利指数特征（基于预测概率）
        if all(col in enhanced_features.columns for col in ["implied_prob_home_win", "implied_prob_draw", "implied_prob_away_win"]):
            enhanced_features["kelly_index_home"] = features.apply(
                lambda x: self.calculate_kelly_index(x["home_win_odds"], enhanced_features.loc[x.name, "implied_prob_home_win"]), axis=1
            )
            enhanced_features["kelly_index_draw"] = features.apply(
                lambda x: self.calculate_kelly_index(x["draw_odds"], enhanced_features.loc[x.name, "implied_prob_draw"]), axis=1
            )
            enhanced_features["kelly_index_away"] = features.apply(
                lambda x: self.calculate_kelly_index(x["away_win_odds"], enhanced_features.loc[x.name, "implied_prob_away_win"]), axis=1
            )
        
        return enhanced_features
    
    def create_polynomial_features(self, features: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """创建多项式特征
        
        Args:
            features: 原始特征DataFrame
            degree: 多项式度数
            
        Returns:
            包含多项式特征的DataFrame
        """
        poly_features = features.copy()
        
        # 选择数值特征
        numeric_features = features.select_dtypes(include=[np.number]).columns
        
        # 创建二次项
        if degree >= 2:
            for i, feat1 in enumerate(numeric_features):
                # 平方项
                poly_features[f"{feat1}_squared"] = features[feat1] ** 2
                
                # 交叉项
                for feat2 in numeric_features[i+1:]:
                    poly_features[f"{feat1}_x_{feat2}"] = features[feat1] * features[feat2]
        
        # 创建三次项
        if degree >= 3:
            for feat in numeric_features:
                poly_features[f"{feat}_cubed"] = features[feat1] ** 3
        
        return poly_features
    
    def create_rolling_features(self, features: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
        """创建滚动窗口特征
        
        Args:
            features: 原始特征DataFrame（按时间排序）
            window_size: 窗口大小
            
        Returns:
            包含滚动特征的DataFrame
        """
        rolling_features = features.copy()
        
        # 确保数据按时间排序
        if "match_date" in features.columns:
            rolling_features = rolling_features.sort_values("match_date")
        
        # 选择数值特征
        numeric_features = features.select_dtypes(include=[np.number]).columns
        
        # 创建滚动特征
        for feat in numeric_features:
            if feat in ["home_win_odds", "draw_odds", "away_win_odds", "implied_home_win", "implied_draw", "implied_away_win", "payout_rate"]:
                # 赔率相关特征使用滚动平均和标准差
                rolling_features[f"{feat}_rolling_mean"] = features[feat].rolling(window=window_size, min_periods=1).mean()
                rolling_features[f"{feat}_rolling_std"] = features[feat].rolling(window=window_size, min_periods=1).std().fillna(0)
                rolling_features[f"{feat}_rolling_max"] = features[feat].rolling(window=window_size, min_periods=1).max()
                rolling_features[f"{feat}_rolling_min"] = features[feat].rolling(window=window_size, min_periods=1).min()
                rolling_features[f"{feat}_rolling_change"] = features[feat].rolling(window=window_size, min_periods=2).apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
            
            elif feat in ["home_goals_scored", "home_goals_conceded", "away_goals_scored", "away_goals_conceded", "home_goal_difference", "away_goal_difference"]:
                # 进球和净胜球特征使用滚动求和和均值
                rolling_features[f"{feat}_rolling_sum"] = features[feat].rolling(window=window_size, min_periods=1).sum()
                rolling_features[f"{feat}_rolling_mean"] = features[feat].rolling(window=window_size, min_periods=1).mean()
                rolling_features[f"{feat}_rolling_std"] = features[feat].rolling(window=window_size, min_periods=1).std().fillna(0)
                rolling_features[f"{feat}_rolling_max"] = features[feat].rolling(window=window_size, min_periods=1).max()
            
            elif feat in ["home_wins", "home_draws", "home_losses", "away_wins", "away_draws", "away_losses", "home_form_points", "away_form_points"]:
                # 球队状态特征使用滚动求和和均值
                rolling_features[f"{feat}_rolling_sum"] = features[feat].rolling(window=window_size, min_periods=1).sum()
                rolling_features[f"{feat}_rolling_mean"] = features[feat].rolling(window=window_size, min_periods=1).mean()
                rolling_features[f"{feat}_rolling_std"] = features[feat].rolling(window=window_size, min_periods=1).std().fillna(0)
            
            elif feat in ["kelly_index_home", "kelly_index_draw", "kelly_index_away"]:
                # 凯利指数特征使用滚动均值和标准差
                rolling_features[f"{feat}_rolling_mean"] = features[feat].rolling(window=window_size, min_periods=1).mean()
                rolling_features[f"{feat}_rolling_std"] = features[feat].rolling(window=window_size, min_periods=1).std().fillna(0)
                rolling_features[f"{feat}_rolling_max"] = features[feat].rolling(window=window_size, min_periods=1).max()
            
            elif feat in ["h2h_total_matches", "h2h_home_wins", "h2h_away_wins", "h2h_draws"]:
                # 交锋历史特征使用滚动求和
                rolling_features[f"{feat}_rolling_sum"] = features[feat].rolling(window=window_size, min_periods=1).sum()
                rolling_features[f"{feat}_rolling_mean"] = features[feat].rolling(window=window_size, min_periods=1).mean()
        
        return rolling_features
    
    def create_ranking_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """创建排名特征
        
        Args:
            features: 原始特征DataFrame
            
        Returns:
            包含排名特征的DataFrame
        """
        ranking_features = features.copy()
        
        # 按赛季分组（如果有赛季信息）
        if "season" in features.columns:
            groupby_col = "season"
        else:
            groupby_col = None
        
        # 创建状态排名
        if groupby_col:
            ranking_features["home_form_points_rank"] = features.groupby(groupby_col)["home_form_points"].rank(pct=True)
            ranking_features["away_form_points_rank"] = features.groupby(groupby_col)["away_form_points"].rank(pct=True)
            ranking_features["home_goal_diff_rank"] = features.groupby(groupby_col)["home_goal_difference"].rank(pct=True)
            ranking_features["away_goal_diff_rank"] = features.groupby(groupby_col)["away_goal_difference"].rank(pct=True)
        else:
            if "home_form_points" in features.columns:
                ranking_features["home_form_points_rank"] = features["home_form_points"].rank(pct=True)
            if "away_form_points" in features.columns:
                ranking_features["away_form_points_rank"] = features["away_form_points"].rank(pct=True)
            if "home_goal_difference" in features.columns:
                ranking_features["home_goal_diff_rank"] = features["home_goal_difference"].rank(pct=True)
            if "away_goal_difference" in features.columns:
                ranking_features["away_goal_diff_rank"] = features["away_goal_difference"].rank(pct=True)
        
        # 创建排名差异特征
        if "home_form_points_rank" in ranking_features.columns and "away_form_points_rank" in ranking_features.columns:
            ranking_features["form_rank_diff"] = ranking_features["home_form_points_rank"] - ranking_features["away_form_points_rank"]
        else:
            ranking_features["form_rank_diff"] = 0
            
        if "home_goal_diff_rank" in ranking_features.columns and "away_goal_diff_rank" in ranking_features.columns:
            ranking_features["goal_diff_rank_diff"] = ranking_features["home_goal_diff_rank"] - ranking_features["away_goal_diff_rank"]
        else:
            ranking_features["goal_diff_rank_diff"] = 0
        
        return ranking_features
    
    def create_categorical_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """创建分类特征
        
        Args:
            features: 原始特征DataFrame
            
        Returns:
            包含分类特征的DataFrame
        """
        categorical_features = features.copy()
        
        # 赔率范围分类
        if "home_win_odds" in features.columns:
            categorical_features["home_odds_category"] = pd.cut(
                features["home_win_odds"], 
                bins=[0, 1.5, 2.0, 2.5, 3.0, 5.0, float('inf')], 
                labels=["very_low", "low", "medium_low", "medium", "medium_high", "high"]
            )
        
        if "away_win_odds" in features.columns:
            categorical_features["away_odds_category"] = pd.cut(
                features["away_win_odds"], 
                bins=[0, 1.5, 2.0, 2.5, 3.0, 5.0, float('inf')], 
                labels=["very_low", "low", "medium_low", "medium", "medium_high", "high"]
            )
        
        # 状态分类
        if "home_form_points" in features.columns:
            categorical_features["home_form_category"] = pd.cut(
                features["home_form_points"], 
                bins=[-1, 3, 6, 9, 12, 15, float('inf')], 
                labels=["very_poor", "poor", "average", "good", "very_good", "excellent"]
            )
        
        if "away_form_points" in features.columns:
            categorical_features["away_form_category"] = pd.cut(
                features["away_form_points"], 
                bins=[-1, 3, 6, 9, 12, 15, float('inf')], 
                labels=["very_poor", "poor", "average", "good", "very_good", "excellent"]
            )
        
        # 进球差分类
        if "goal_diff_diff" in features.columns:
            categorical_features["goal_diff_category"] = pd.cut(
                features["goal_diff_diff"], 
                bins=[-10, -5, -2, 0, 2, 5, 10], 
                labels=["heavy_loss", "loss", "slight_loss", "draw", "slight_win", "heavy_win"]
            )
        
        return categorical_features
    
    def select_features(self, features: pd.DataFrame, target: pd.Series, method: str = "correlation", k: int = 50) -> List[str]:
        """特征选择
        
        Args:
            features: 特征DataFrame
            target: 目标变量
            method: 选择方法 ("correlation", "mutual_info", "chi2")
            k: 选择的特征数量
            
        Returns:
            选择的特征列表
        """
        from sklearn.feature_selection import (
            SelectKBest, 
            f_classif, 
            mutual_info_classif, 
            chi2
        )
        from sklearn.preprocessing import LabelEncoder
        
        # 只选择数值特征
        numeric_features = features.select_dtypes(include=[np.number]).columns
        X = features[numeric_features]
        
        # 处理目标变量
        if target.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(target)
        else:
            y = target
        
        # 根据方法选择特征
        if method == "correlation":
            # 计算相关性
            correlations = X.apply(lambda x: abs(x.corr(y)))
            selected_features = correlations.nlargest(k).index.tolist()
        elif method == "mutual_info":
            # 互信息
            selector = SelectKBest(mutual_info_classif, k=k)
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
        elif method == "chi2":
            # 卡方检验（需要非负特征）
            X_nonneg = X - X.min() + 1e-8
            selector = SelectKBest(chi2, k=k)
            selector.fit(X_nonneg, y)
            selected_features = X.columns[selector.get_support()].tolist()
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # 保存特征重要性
        if method == "correlation":
            self.feature_importance = dict(correlations[selected_features])
        
        return selected_features
    
    def engineer_features(self, features: pd.DataFrame, target: pd.Series = None) -> Tuple[pd.DataFrame, List[str]]:
        """完整的特征工程流程
        
        Args:
            features: 原始特征DataFrame
            target: 目标变量（用于特征选择）
            
        Returns:
            增强后的特征DataFrame和特征列表
        """
        enhanced_features = features.copy()
        
        # 创建各种特征
        enhanced_features = self.create_interaction_features(enhanced_features)
        enhanced_features = self.create_polynomial_features(enhanced_features, degree=2)
        enhanced_features = self.create_ranking_features(enhanced_features)
        enhanced_features = self.create_categorical_features(enhanced_features)
        
        # 处理分类特征（独热编码）
        categorical_cols = enhanced_features.select_dtypes(include=['category', 'object']).columns
        if len(categorical_cols) > 0:
            enhanced_features = pd.get_dummies(enhanced_features, columns=categorical_cols, drop_first=True)
        
        # 特征选择
        if target is not None:
            selected_features = self.select_features(enhanced_features, target, method="correlation", k=100)
            final_features = enhanced_features[selected_features + [target.name] if target.name in enhanced_features.columns else selected_features]
        else:
            final_features = enhanced_features
            selected_features = enhanced_features.columns.tolist()
        
        return final_features, selected_features