import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class NeuralNetworkDataCleaner:
    def __init__(self, data_root, output_root):
        """初始化神经网络数据清洗器
        
        Args:
            data_root: 原始数据根目录
            output_root: 清洗后数据输出目录
        """
        self.data_root = data_root
        self.output_root = output_root
        self.bookmakers = {'82': 'ladbrokes', '115': 'williamhill'}
        
        # 确保输出目录存在
        os.makedirs(self.output_root, exist_ok=True)
        os.makedirs(os.path.join(self.output_root, 'lstm'), exist_ok=True)
    
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
    
    def calculate_implied_probability(self, payout_rate, odds):
        """计算隐含概率"""
        try:
            odds = float(odds)
        except (ValueError, TypeError):
            return 0.0
        
        if odds <= 0:
            return 0.0
        return (1 / odds) / payout_rate
    
    def _clean_single_match_odds(self, match_details):
        """清洗单场比赛的赔率数据，保留时间序列信息"""
        if not match_details or 'odds' not in match_details:
            return None
        
        odds_data = match_details['odds']
        clean_odds = []
        
        # 遍历所有博彩公司
        for bookmaker_id, bookmaker_odds in odds_data.items():
            if bookmaker_id not in self.bookmakers:
                continue
            
            bookmaker_name = self.bookmakers[bookmaker_id]
            
            # 按时间排序赔率（最早的在前，最新的在后）
            try:
                sorted_odds = sorted(bookmaker_odds, key=lambda x: datetime.strptime(x[6], "%Y-%m-%d %H:%M"))
            except ValueError:
                continue
            
            # 保留最多10个时间点的赔率数据
            sorted_odds = sorted_odds[:10]
            
            for odds in sorted_odds:
                try:
                    # 计算赔付率和隐含概率
                    payout_rate = self.calculate_payout_rate(odds[0], odds[1], odds[2])
                    implied_prob_win = self.calculate_implied_probability(payout_rate, odds[0])
                    implied_prob_draw = self.calculate_implied_probability(payout_rate, odds[1])
                    implied_prob_lose = self.calculate_implied_probability(payout_rate, odds[2])
                    
                    clean_odds.append({
                        'bookmaker': bookmaker_name,
                        'time': odds[6],
                        'win_odds': float(odds[0]),
                        'draw_odds': float(odds[1]),
                        'lose_odds': float(odds[2]),
                        'win_kelly': float(odds[3]) if odds[3] else 0.0,
                        'draw_kelly': float(odds[4]) if odds[4] else 0.0,
                        'lose_kelly': float(odds[5]) if odds[5] else 0.0,
                        'payout_rate': payout_rate,
                        'implied_win_prob': implied_prob_win,
                        'implied_draw_prob': implied_prob_draw,
                        'implied_lose_prob': implied_prob_lose
                    })
                except (ValueError, TypeError):
                    continue
        
        # 按时间排序所有赔率数据
        clean_odds.sort(key=lambda x: x['time'])
        return clean_odds
    
    def clean_season_data_for_lstm(self, season):
        """清洗单个赛季的数据，生成LSTM训练所需的序列数据"""
        print(f"\n=== 清洗赛季 {season} 数据 ===")
        
        season_dir = os.path.join(self.data_root, season)
        if not os.path.exists(season_dir):
            print(f"赛季 {season} 目录不存在")
            return None
        
        # 读取赛季比赛列表
        season_files = [f for f in os.listdir(season_dir) if f.endswith('.json') and not f.startswith('details')]
        if not season_files:
            print(f"赛季 {season} 无比赛数据文件")
            return None
        
        season_data_path = os.path.join(season_dir, season_files[0])
        with open(season_data_path, 'r', encoding='utf-8') as f:
            season_data = json.load(f)
        
        # 构建详细信息目录路径
        details_dir = os.path.join(season_dir, "details")
        if not os.path.exists(details_dir):
            print(f"赛季 {season} 无详细信息目录")
            return None
        
        # 遍历所有比赛
        processed_matches = []
        
        for match_id, match_info in season_data.items():
            # 构建详细信息文件路径
            match_id_str = str(match_id)
            match_details = None
            
            # 查找比赛详细信息文件
            round_dirs = [d for d in os.listdir(details_dir) if os.path.isdir(os.path.join(details_dir, d))]
            for round_dir in round_dirs:
                detail_file = os.path.join(details_dir, round_dir, f"{match_id_str}.json")
                if os.path.exists(detail_file):
                    with open(detail_file, 'r', encoding='utf-8') as f:
                        match_details = json.load(f)
                    break
            
            if not match_details:
                continue
            
            # 清洗赔率数据，保留时间序列
            cleaned_odds = self._clean_single_match_odds(match_details)
            if not cleaned_odds:
                continue
            
            # 提取比赛基本信息
            match_basic = {
                'match_id': match_id_str,
                'season': season,
                'home_team': match_info.get('homeTeamName'),
                'away_team': match_info.get('awayTeamName'),
                'result_code': match_info.get('result'),  # 3=主胜, 1=平局, 0=客胜
                'home_score': match_info.get('homeScore'),
                'away_score': match_info.get('awayScore'),
                'odds_sequence': cleaned_odds
            }
            
            # 转换结果编码为模型预期格式 (2=主胜, 1=平局, 0=客胜)
            if match_basic['result_code'] == 3:
                match_basic['result_code'] = 2
            
            processed_matches.append(match_basic)
        
        print(f"成功处理 {len(processed_matches)} 场比赛")
        
        # 保存清洗后的数据
        output_path = os.path.join(self.output_root, 'lstm', f"{season}_lstm_data.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_matches, f, ensure_ascii=False, indent=2)
        
        return processed_matches
    
    def create_sequence_data(self, season_data, time_steps=5):
        """创建LSTM训练所需的序列数据
        
        Args:
            season_data: 清洗后的赛季数据
            time_steps: 序列长度
            
        Returns:
            X: 序列特征 [样本数, 时间步, 特征数]
            y: 标签 [样本数]
        """
        print(f"\n=== 创建序列数据，时间步: {time_steps} ===")
        
        sequences = []
        labels = []
        
        for match in season_data:
            odds_sequence = match['odds_sequence']
            result = match['result_code']
            
            # 确保有足够的时间点
            if len(odds_sequence) < time_steps:
                continue
            
            # 提取特征序列
            for i in range(len(odds_sequence) - time_steps + 1):
                # 提取连续time_steps个时间点的特征
                seq = odds_sequence[i:i+time_steps]
                
                # 构建特征向量
                features = []
                for odds in seq:
                    # 提取数值特征
                    feature_vector = [
                        odds['win_odds'],
                        odds['draw_odds'],
                        odds['lose_odds'],
                        odds['win_kelly'],
                        odds['draw_kelly'],
                        odds['lose_kelly'],
                        odds['payout_rate'],
                        odds['implied_win_prob'],
                        odds['implied_draw_prob'],
                        odds['implied_lose_prob']
                    ]
                    features.append(feature_vector)
                
                # 添加到序列中
                sequences.append(features)
                labels.append(result)
        
        # 转换为numpy数组
        X = np.array(sequences)
        y = np.array(labels)
        
        print(f"序列数据创建完成: X.shape={X.shape}, y.shape={y.shape}")
        print(f"标签分布: {np.unique(y, return_counts=True)}")
        
        return X, y
    
    def normalize_data(self, X_train, X_test=None):
        """归一化序列数据"""
        print("\n=== 归一化数据 ===")
        
        # 获取特征维度
        n_samples, n_time_steps, n_features = X_train.shape
        
        # 将数据reshape为 [n_samples * n_time_steps, n_features] 以便归一化
        X_train_reshaped = X_train.reshape(-1, n_features)
        
        # 创建并拟合scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_normalized = scaler.fit_transform(X_train_reshaped)
        
        # 恢复原始形状
        X_train_normalized = X_train_normalized.reshape(n_samples, n_time_steps, n_features)
        
        if X_test is not None:
            n_test_samples, n_test_time_steps, n_test_features = X_test.shape
            X_test_reshaped = X_test.reshape(-1, n_test_features)
            X_test_normalized = scaler.transform(X_test_reshaped)
            X_test_normalized = X_test_normalized.reshape(n_test_samples, n_test_time_steps, n_test_features)
            return X_train_normalized, X_test_normalized, scaler
        
        return X_train_normalized, scaler
    
    def clean_all_seasons(self, seasons=None):
        """清洗所有赛季的数据"""
        print("=== 开始清洗所有赛季数据 ===")
        
        # 获取所有赛季目录
        if seasons is None:
            seasons = [d for d in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, d))]
            seasons.sort()
        
        all_processed_data = []
        
        for season in seasons:
            processed_data = self.clean_season_data_for_lstm(season)
            if processed_data:
                all_processed_data.extend(processed_data)
        
        print(f"\n=== 所有赛季数据清洗完成 ===")
        print(f"总共处理 {len(all_processed_data)} 场比赛")
        
        # 保存所有赛季的合并数据
        output_path = os.path.join(self.output_root, 'lstm', f"all_seasons_lstm_data.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_processed_data, f, ensure_ascii=False, indent=2)
        
        return all_processed_data
    
    def prepare_training_data(self, seasons=None, time_steps=5, test_size=0.2):
        """准备完整的训练数据
        
        Args:
            seasons: 要使用的赛季列表
            time_steps: 序列长度
            test_size: 测试集比例
            
        Returns:
            X_train: 训练集特征
            X_test: 测试集特征
            y_train: 训练集标签
            y_test: 测试集标签
            scaler: 归一化器
        """
        # 清洗数据
        processed_data = self.clean_all_seasons(seasons)
        
        # 创建序列数据
        X, y = self.create_sequence_data(processed_data, time_steps)
        
        # 划分训练集和测试集
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"\n数据集划分完成:")
        print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
        print(f"训练集标签分布: {np.unique(y_train, return_counts=True)}")
        print(f"测试集标签分布: {np.unique(y_test, return_counts=True)}")
        
        # 归一化数据
        X_train_normalized, X_test_normalized, scaler = self.normalize_data(X_train, X_test)
        
        return X_train_normalized, X_test_normalized, y_train, y_test, scaler

# 使用示例
if __name__ == "__main__":
    # 配置路径
    DATA_ROOT = "/Users/Williamhiler/Documents/my-project/train/original-data"
    OUTPUT_ROOT = "/Users/Williamhiler/Documents/my-project/train/train-data"
    
    # 创建数据清洗器
    cleaner = NeuralNetworkDataCleaner(DATA_ROOT, OUTPUT_ROOT)
    
    # 清洗所有赛季数据
    cleaner.clean_all_seasons()
    
    # 准备训练数据示例
    # X_train, X_test, y_train, y_test, scaler = cleaner.prepare_training_data(
    #     seasons=['2020-2021', '2021-2022', '2022-2023'],
    #     time_steps=5
    # )
    
    print("\n=== 数据清洗完成 ===")