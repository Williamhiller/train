import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, data_root):
        self.data_root = data_root
    
    def load_odds_features(self, season):
        """加载赔率特征"""
        file_path = os.path.join(self.data_root, 'odds', f'{season}_odds_features.json')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"赔率特征文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取所有比赛的赔率特征和赛果
        matches = []
        for match in data['matches']:
            if 'result' not in match or not match['result']:
                continue
                
            match_id = match['match_id']
            result = match['result']
            
            # 提取赛果信息
            match_data = {
                'match_id': match_id,
                'season': season,
                # 将result_code从[0, 1, 3]映射到[0, 1, 2]
                'result_code': 2 if result['result_code'] == 3 else result['result_code'],  # 2=主胜, 1=平局, 0=客胜
                'home_score': result['home_score'],
                'away_score': result['away_score']
            }
            
            # 提取ladbrokes的赔率特征
            if 'ladbrokes' in match['bookmakers']:
                ladbrokes = match['bookmakers']['ladbrokes']
                
                # 初始赔率
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
                
                # 终盘赔率
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
                
                # 赔率变化率
                if ladbrokes.get('odds_change_rate'):
                    match_data.update({
                        'win_odds_change_rate': ladbrokes['odds_change_rate']['win'],
                        'draw_odds_change_rate': ladbrokes['odds_change_rate']['draw'],
                        'lose_odds_change_rate': ladbrokes['odds_change_rate']['lose']
                    })
                
                # 其他赔率特征
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
    
    def load_combined_features(self, season, include_team_state=False, include_expert=False, use_llm=False):
        """加载组合特征"""
        # 加载赔率特征（基础特征）
        odds_df = self.load_odds_features(season)
        
        # 合并球队状态特征
        if include_team_state:
            team_state_df = self.load_team_state_features(season)
            # 将match_id转换为字符串进行合并
            odds_df['match_id'] = odds_df['match_id'].astype(str)
            team_state_df['match_id'] = team_state_df['match_id'].astype(str)
            odds_df = odds_df.merge(team_state_df, on=['match_id', 'season'], how='inner')
        
        # 合并专家特征
        if include_expert:
            expert_df = self.load_expert_features(season, use_llm)
            # 将match_id转换为字符串进行合并
            odds_df['match_id'] = odds_df['match_id'].astype(str)
            expert_df['match_id'] = expert_df['match_id'].astype(str)
            odds_df = odds_df.merge(expert_df, on=['match_id', 'season'], how='inner')
        
        return odds_df
    
    def prepare_training_data(self, seasons, include_team_state=False, include_expert=False, use_llm=False):
        """准备训练数据"""
        all_data = []
        
        for season in seasons:
            try:
                season_data = self.load_combined_features(season, include_team_state, include_expert, use_llm)
                all_data.append(season_data)
            except FileNotFoundError as e:
                print(f"跳过赛季 {season}: {e}")
        
        if not all_data:
            raise ValueError("没有找到任何训练数据")
        
        # 合并所有赛季的数据
        df = pd.concat(all_data, ignore_index=True)
        
        # 处理缺失值
        df = df.dropna()
        
        # 不再过滤赔率数据，保留所有比赛
        print(f"数据量: {len(df)}")
        
        # 保存完整特征数据，包含赔率信息
        self.X_full = df.copy()
        
        # 特征和标签分离
        features = df.drop(['match_id', 'season', 'result_code', 'home_score', 'away_score'], axis=1, errors='ignore')
        labels = df['result_code']
        
        # 转换分类特征
        categorical_cols = features.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            features = pd.get_dummies(features, columns=categorical_cols)
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        
        # 保存分割后的索引，以便后续匹配
        self.train_indices = X_train.index
        self.test_indices = X_test.index
        
        return X_train, X_test, y_train, y_test, features.columns.tolist()
    
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
                'match_id': match_id,
                'season': season
            }
            
            # 主队特征
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
            
            # 客队特征
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
            
            # 交手记录特征
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
    
    def load_expert_features(self, season, use_llm=False):
        """加载专家特征"""
        if use_llm:
            file_path = os.path.join(self.data_root, 'expert', f'{season}_expert_features_llm.json')
        else:
            file_path = os.path.join(self.data_root, 'expert', f'{season}_expert_features.json')
        
        if not os.path.exists(file_path):
            # 如果LLM增强特征文件不存在，回退到原始特征
            if use_llm:
                print(f"LLM增强专家特征文件不存在，回退到原始特征: {file_path}")
                file_path = os.path.join(self.data_root, 'expert', f'{season}_expert_features.json')
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"专家特征文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        matches = []
        for match_id, features in data.items():
            match_data = {
                'match_id': match_id,
                'season': season,
                'odds_match_degree': features['odds_match_degree'],
                'head_to_head_consistency': features['head_to_head_consistency'],
                'home_away_odds_factor': features['home_away_odds_factor'],
                'recent_form_odds_correlation': features['recent_form_odds_correlation'],
                'expert_confidence_score': features['expert_confidence_score']
            }
            
            # 添加LLM增强特征（如果存在）
            if 'llm_enhanced' in features and features['llm_enhanced']:
                for key, value in features.items():
                    if key.startswith('llm_'):
                        match_data[key] = value
            
            matches.append(match_data)
        
        return pd.DataFrame(matches)
