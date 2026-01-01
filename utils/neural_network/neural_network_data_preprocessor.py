import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

class NeuralNetworkDataPreprocessor:
    """神经网络训练数据预处理器"""
    
    def __init__(self, data_root, output_root):
        """初始化数据预处理器
        
        Args:
            data_root: 原始数据根目录
            output_root: 处理后数据输出目录
        """
        self.data_root = data_root
        self.output_root = output_root
        self.season_mapping = {
            '2015-2016': 1,
            '2016-2017': 2,
            '2017-2018': 3,
            '2018-2019': 4,
            '2019-2020': 5,
            '2020-2021': 6,
            '2021-2022': 7,
            '2022-2023': 8,
            '2023-2024': 9,
            '2024-2025': 10
        }
        
        # 确保输出目录存在
        os.makedirs(self.output_root, exist_ok=True)
        os.makedirs(os.path.join(self.output_root, 'processed'), exist_ok=True)
        os.makedirs(os.path.join(self.output_root, 'sequence'), exist_ok=True)
    
    def load_odds_features(self, season):
        """加载赔率特征"""
        file_path = os.path.join(self.data_root, 'odds', f'{season}_odds_features.json')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"赔率特征文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        matches = []
        for match in data['matches']:
            if 'result' not in match or not match['result']:
                continue
                
            match_id = match['match_id']
            result = match['result']
            
            match_data = {
                'match_id': str(match_id),
                'season': season,
                'season_id': self.season_mapping[season],
                'result_code': 2 if result['result_code'] == 3 else result['result_code'],
                'home_score': result['home_score'],
                'away_score': result['away_score']
            }
            
            if 'ladbrokes' in match['bookmakers']:
                ladbrokes = match['bookmakers']['ladbrokes']
                
                if ladbrokes.get('initial_odds'):
                    initial_odds = ladbrokes['initial_odds']
                    if all(key in initial_odds for key in ['win', 'draw', 'lose', 'payout_rate', 'implied_probability', 'kelly_index']) and \
                       all(key in initial_odds['implied_probability'] for key in ['win', 'draw', 'lose']) and \
                       all(key in initial_odds['kelly_index'] for key in ['win', 'draw', 'lose']):
                        match_data.update({
                            'initial_win_odds': float(initial_odds['win']),
                            'initial_draw_odds': float(initial_odds['draw']),
                            'initial_lose_odds': float(initial_odds['lose']),
                            'initial_payout_rate': initial_odds['payout_rate'],
                            'initial_implied_win_prob': initial_odds['implied_probability']['win'],
                            'initial_implied_draw_prob': initial_odds['implied_probability']['draw'],
                            'initial_implied_lose_prob': initial_odds['implied_probability']['lose'],
                            'initial_win_kelly': float(initial_odds['kelly_index']['win']),
                            'initial_draw_kelly': float(initial_odds['kelly_index']['draw']),
                            'initial_lose_kelly': float(initial_odds['kelly_index']['lose'])
                        })
                
                if ladbrokes.get('closing_odds'):
                    closing_odds = ladbrokes['closing_odds']
                    if all(key in closing_odds for key in ['win', 'draw', 'lose', 'payout_rate', 'implied_probability', 'kelly_index']) and \
                       all(key in closing_odds['implied_probability'] for key in ['win', 'draw', 'lose']) and \
                       all(key in closing_odds['kelly_index'] for key in ['win', 'draw', 'lose']):
                        match_data.update({
                            'closing_win_odds': float(closing_odds['win']),
                            'closing_draw_odds': float(closing_odds['draw']),
                            'closing_lose_odds': float(closing_odds['lose']),
                            'closing_payout_rate': closing_odds['payout_rate'],
                            'closing_implied_win_prob': closing_odds['implied_probability']['win'],
                            'closing_implied_draw_prob': closing_odds['implied_probability']['draw'],
                            'closing_implied_lose_prob': closing_odds['implied_probability']['lose'],
                            'closing_win_kelly': float(closing_odds['kelly_index']['win']),
                            'closing_draw_kelly': float(closing_odds['kelly_index']['draw']),
                            'closing_lose_kelly': float(closing_odds['kelly_index']['lose'])
                        })
                
                if ladbrokes.get('odds_change_rate'):
                    match_data.update({
                        'win_odds_change_rate': ladbrokes['odds_change_rate']['win'],
                        'draw_odds_change_rate': ladbrokes['odds_change_rate']['draw'],
                        'lose_odds_change_rate': ladbrokes['odds_change_rate']['lose']
                    })
                
                if ladbrokes.get('odds_change_frequency') is not None:
                    match_data['odds_change_frequency'] = ladbrokes['odds_change_frequency']
                
                if ladbrokes.get('odds_trend') and ladbrokes['odds_trend'].get('win'):
                    match_data.update({
                        'win_odds_trend': ladbrokes['odds_trend']['win']['trend'],
                        'win_odds_trend_strength': ladbrokes['odds_trend']['win']['strength']
                    })
                
                if ladbrokes.get('odds_trend') and ladbrokes['odds_trend'].get('draw'):
                    match_data.update({
                        'draw_odds_trend': ladbrokes['odds_trend']['draw']['trend'],
                        'draw_odds_trend_strength': ladbrokes['odds_trend']['draw']['strength']
                    })
                
                if ladbrokes.get('odds_trend') and ladbrokes['odds_trend'].get('lose'):
                    match_data.update({
                        'lose_odds_trend': ladbrokes['odds_trend']['lose']['trend'],
                        'lose_odds_trend_strength': ladbrokes['odds_trend']['lose']['strength']
                    })
            
            matches.append(match_data)
        
        return pd.DataFrame(matches)
    
    def load_team_state_features(self, season):
        """加载球队状态特征"""
        file_path = os.path.join(self.data_root, 'team_state', f'{season}_team_state_features.json')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"球队状态特征文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        matches = []
        for match_id, match in data.items():
            match_data = {
                'match_id': str(match_id),
                'season': season
            }
            
            if 'home_team' in match:
                home_team = match['home_team']
                if 'recent_form' in home_team:
                    match_data.update({
                        'home_recent_games': home_team['recent_form']['recent_matches'],
                        'home_recent_wins': home_team['recent_form']['recent_wins'],
                        'home_recent_draws': home_team['recent_form']['recent_draws'],
                        'home_recent_losses': home_team['recent_form']['recent_losses'],
                        'home_recent_win_rate': home_team['recent_form']['win_rate'],
                        'home_recent_avg_goals': home_team['recent_form']['avg_goals_scored'],
                        'home_recent_avg_conceded': home_team['recent_form']['avg_goals_conceded']
                    })
                
                if 'season_data' in home_team:
                    match_data.update({
                        'home_season_wins': home_team['season_data']['season_wins'],
                        'home_season_draws': home_team['season_data']['season_draws'],
                        'home_season_losses': home_team['season_data']['season_losses'],
                        'home_season_goals': home_team['season_data']['season_goals_for'],
                        'home_season_conceded': home_team['season_data']['season_goals_against'],
                        'home_season_points': home_team['season_data']['season_points'],
                        'home_season_rank': home_team['season_data']['season_rank'],
                        'home_season_win_rate': home_team['season_data']['season_win_rate']
                    })
            
            if 'away_team' in match:
                away_team = match['away_team']
                if 'recent_form' in away_team:
                    match_data.update({
                        'away_recent_games': away_team['recent_form']['recent_matches'],
                        'away_recent_wins': away_team['recent_form']['recent_wins'],
                        'away_recent_draws': away_team['recent_form']['recent_draws'],
                        'away_recent_losses': away_team['recent_form']['recent_losses'],
                        'away_recent_win_rate': away_team['recent_form']['win_rate'],
                        'away_recent_avg_goals': away_team['recent_form']['avg_goals_scored'],
                        'away_recent_avg_conceded': away_team['recent_form']['avg_goals_conceded']
                    })
                
                if 'season_data' in away_team:
                    match_data.update({
                        'away_season_wins': away_team['season_data']['season_wins'],
                        'away_season_draws': away_team['season_data']['season_draws'],
                        'away_season_losses': away_team['season_data']['season_losses'],
                        'away_season_goals': away_team['season_data']['season_goals_for'],
                        'away_season_conceded': away_team['season_data']['season_goals_against'],
                        'away_season_points': away_team['season_data']['season_points'],
                        'away_season_rank': away_team['season_data']['season_rank'],
                        'away_season_win_rate': away_team['season_data']['season_win_rate']
                    })
            
            if 'head_to_head' in match:
                head_to_head = match['head_to_head']
                match_data.update({
                    'head_to_head_matches': head_to_head['head_to_head_matches'],
                    'head_to_head_home_wins': head_to_head['home_wins'],
                    'head_to_head_away_wins': head_to_head['away_wins'],
                    'head_to_head_draws': head_to_head['draws'],
                    'head_to_head_home_goals': head_to_head['home_goals'],
                    'head_to_head_away_goals': head_to_head['away_goals'],
                    'head_to_head_home_win_rate': head_to_head['home_win_rate'],
                    'head_to_head_away_win_rate': head_to_head['away_win_rate'],
                    'head_to_head_draw_rate': head_to_head['draw_rate']
                })
            
            matches.append(match_data)
        
        return pd.DataFrame(matches)
    
    def load_expert_features(self, season):
        """加载专家特征"""
        file_path = os.path.join(self.data_root, 'expert', f'{season}_expert_features.json')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"专家特征文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        matches = []
        for match_id, features in data.items():
            match_data = {
                'match_id': str(match_id),
                'season': season,
                'odds_match_degree': features['odds_match_degree'],
                'head_to_head_consistency': features['head_to_head_consistency'],
                'home_away_odds_factor': features['home_away_odds_factor'],
                'recent_form_odds_correlation': features['recent_form_odds_correlation'],
                'expert_confidence_score': features['expert_confidence_score']
            }
            matches.append(match_data)
        
        return pd.DataFrame(matches)
    
    def load_combined_features(self, season, include_team_state=True, include_expert=True):
        """加载组合特征"""
        odds_df = self.load_odds_features(season)
        
        if include_team_state:
            team_state_df = self.load_team_state_features(season)
            odds_df = odds_df.merge(team_state_df, on=['match_id', 'season'], how='inner')
        
        if include_expert:
            expert_df = self.load_expert_features(season)
            odds_df = odds_df.merge(expert_df, on=['match_id', 'season'], how='inner')
        
        return odds_df
    
    def process_season_data(self, season, include_team_state=True, include_expert=True):
        """处理单个赛季的数据"""
        print(f"\n=== 处理赛季 {season} 数据 ===")
        
        try:
            # 加载组合特征
            df = self.load_combined_features(season, include_team_state, include_expert)
            
            # 处理缺失值
            df = df.dropna()
            
            # 不再过滤赔率数据，保留所有比赛
            
            print(f"赛季 {season} 处理完成: {len(df)} 场比赛")
            
            # 保存处理后的数据
            output_path = os.path.join(self.output_root, 'processed', f'{season}_processed.csv')
            df.to_csv(output_path, index=False, encoding='utf-8')
            
            return df
        except Exception as e:
            print(f"处理赛季 {season} 失败: {e}")
            return None
    
    def process_all_seasons(self, seasons, include_team_state=True, include_expert=True):
        """处理所有赛季的数据"""
        print("=== 开始处理所有赛季数据 ===")
        
        all_data = []
        
        for season in seasons:
            processed_df = self.process_season_data(season, include_team_state, include_expert)
            if processed_df is not None:
                all_data.append(processed_df)
        
        if not all_data:
            print("没有处理成功任何赛季数据")
            return None
        
        # 合并所有赛季数据
        combined_df = pd.concat(all_data, ignore_index=True)
        
        print(f"\n=== 所有赛季数据处理完成 ===")
        print(f"总共处理 {len(combined_df)} 场比赛")
        
        # 保存合并后的数据
        combined_output_path = os.path.join(self.output_root, 'processed', 'all_seasons_processed.csv')
        combined_df.to_csv(combined_output_path, index=False, encoding='utf-8')
        
        return combined_df
    
    def encode_categorical_features(self, df):
        """编码类别特征"""
        # 定义需要编码的类别特征
        categorical_features = [
            'win_odds_trend',
            'draw_odds_trend',
            'lose_odds_trend'
        ]
        
        # 趋势映射：down -> -1, stable -> 0, up -> 1
        trend_mapping = {
            'down': -1,
            'stable': 0,
            'up': 1,
            'Down': -1,
            'Stable': 0,
            'Up': 1
        }
        
        df_encoded = df.copy()
        
        for feature in categorical_features:
            if feature in df_encoded.columns:
                # 将字符串趋势转换为数值
                df_encoded[feature] = df_encoded[feature].map(trend_mapping).fillna(0)
        
        return df_encoded
    
    def create_sequence_data(self, df, time_steps=5, test_size=0.2, random_state=42):
        """创建LSTM训练所需的序列数据"""
        print(f"\n=== 创建序列数据，时间步: {time_steps} ===")
        
        # 分离特征和标签
        features = df.drop(['match_id', 'season', 'result_code', 'home_score', 'away_score'], axis=1)
        labels = df['result_code']
        
        # 编码类别特征
        features_encoded = self.encode_categorical_features(features)
        
        # 归一化特征
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_encoded)
        
        # 创建序列数据
        X_seq = []
        y_seq = []
        
        for i in range(len(features_scaled) - time_steps + 1):
            X_seq.append(features_scaled[i:i+time_steps])
            y_seq.append(labels.iloc[i+time_steps-1])
        
        # 转换为numpy数组
        X = np.array(X_seq)
        y = np.array(y_seq)
        
        print(f"序列数据创建完成: X.shape={X.shape}, y.shape={y.shape}")
        print(f"标签分布: {np.unique(y, return_counts=True)}")
        
        # 划分训练集和测试集
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"\n数据集划分完成:")
        print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
        print(f"训练集标签分布: {np.unique(y_train, return_counts=True)}")
        print(f"测试集标签分布: {np.unique(y_test, return_counts=True)}")
        
        # 保存序列数据
        np.save(os.path.join(self.output_root, 'sequence', 'X_train.npy'), X_train)
        np.save(os.path.join(self.output_root, 'sequence', 'X_test.npy'), X_test)
        np.save(os.path.join(self.output_root, 'sequence', 'y_train.npy'), y_train)
        np.save(os.path.join(self.output_root, 'sequence', 'y_test.npy'), y_test)
        
        # 保存scaler
        import joblib
        scaler_path = os.path.join(self.output_root, 'sequence', 'scaler.joblib')
        joblib.dump(scaler, scaler_path)
        
        # 保存特征名称
        feature_names_path = os.path.join(self.output_root, 'sequence', 'feature_names.json')
        with open(feature_names_path, 'w', encoding='utf-8') as f:
            json.dump(features.columns.tolist(), f, ensure_ascii=False, indent=2)
        
        return X_train, X_test, y_train, y_test, scaler
    
    def prepare_training_data(self, seasons, include_team_state=True, include_expert=True, time_steps=5, test_size=0.2, random_state=42):
        """准备完整的训练数据"""
        # 处理所有赛季数据
        processed_df = self.process_all_seasons(seasons, include_team_state, include_expert)
        if processed_df is None:
            return None
        
        # 创建序列数据
        X_train, X_test, y_train, y_test, scaler = self.create_sequence_data(processed_df, time_steps, test_size, random_state)
        
        return X_train, X_test, y_train, y_test, scaler

# 使用示例
if __name__ == "__main__":
    # 配置路径
    DATA_ROOT = "/Users/Williamhiler/Documents/my-project/train/train-data"
    OUTPUT_ROOT = "/Users/Williamhiler/Documents/my-project/train/train-data"
    
    # 赛季列表
    SEASONS = [f"{year}-{year+1}" for year in range(2015, 2025)]  # 2015-2016 to 2024-2025 seasons
    
    # 创建数据预处理器
    preprocessor = NeuralNetworkDataPreprocessor(DATA_ROOT, OUTPUT_ROOT)
    
    # 准备训练数据
    result = preprocessor.prepare_training_data(
        seasons=SEASONS,
        include_team_state=True,
        include_expert=True,
        time_steps=5,
        test_size=0.2
    )
    
    if result is not None:
        print("\n=== 训练数据准备完成 ===")
    else:
        print("\n=== 训练数据准备失败 ===")