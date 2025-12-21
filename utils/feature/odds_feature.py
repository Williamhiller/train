import json
import os
from datetime import datetime

class OddsFeatureExtractor:
    def __init__(self, data_root):
        self.data_root = data_root
        self.bookmakers = {
            '82': 'ladbrokes',
            '115': 'williamhill'
        }

    def calculate_payout_rate(self, win_odds, draw_odds, lose_odds):
        """计算赔付率"""
        try:
            return 1 / (1/float(win_odds) + 1/float(draw_odds) + 1/float(lose_odds))
        except (ValueError, ZeroDivisionError):
            return None

    def calculate_kelly_index(self, odds: float, predicted_prob: float) -> float:
        """计算凯利指数
        
        参数:
            odds: 赔率 (必须 > 1)
            predicted_prob: 预测概率 (0-1之间)
        
        返回:
            凯利指数 (小数形式，如0.15表示15%)
        """
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

    def calculate_odds_variance(self, odds_list):
        """计算赔率离散度"""
        if not odds_list or len(odds_list) < 2:
            return 0.0
        
        try:
            odds_list = [float(odd) for odd in odds_list if odd and float(odd) > 0]
            
            if len(odds_list) < 2:
                return 0.0
            
            mean = sum(odds_list) / len(odds_list)
            variance = sum((x - mean) ** 2 for x in odds_list) / len(odds_list)
            return round(variance, 6)
        except (ValueError, ZeroDivisionError):
            return 0.0

    def calculate_trend(self, odds_list):
        """计算赔率趋势
        
        参数:
            odds_list: 按时间顺序排列的赔率列表
        
        返回:
            trend: 趋势类型 ('up', 'down', 'stable')
            trend_strength: 趋势强度 (0-1之间)
        """
        if not odds_list or len(odds_list) < 2:
            return 'stable', 0.0
        
        try:
            odds_list = [float(odd) for odd in odds_list if odd and float(odd) > 0]
            
            if len(odds_list) < 2:
                return 'stable', 0.0
            
            # 计算简单移动平均斜率
            changes = []
            for i in range(1, len(odds_list)):
                changes.append(odds_list[i] - odds_list[i-1])
            
            avg_change = sum(changes) / len(changes)
            
            # 确定趋势类型
            if avg_change > 0.01:
                trend = 'up'
            elif avg_change < -0.01:
                trend = 'down'
            else:
                trend = 'stable'
            
            # 计算趋势强度 (基于变化的一致性)
            positive_changes = sum(1 for c in changes if c > 0)
            negative_changes = sum(1 for c in changes if c < 0)
            total_changes = len(changes)
            
            if total_changes > 0:
                trend_strength = max(positive_changes, negative_changes) / total_changes
            else:
                trend_strength = 0.0
            
            return trend, round(trend_strength, 4)
        except (ValueError, ZeroDivisionError):
            return 'stable', 0.0

    def extract_match_odds_features(self, season, match_id, season_data=None):
        """提取单场比赛的赔率特征"""
        # 将match_id转换为字符串类型，确保一致性
        match_id = str(match_id)
        
        # 提取赔率特征
        odds_features = {
            'match_id': match_id,
            'season': season,
            'bookmakers': {}
        }
        
        # 添加赛果信息到赔率特征中
        if season_data and match_id in season_data:
            match_basic = season_data[match_id]
            odds_features['result'] = {
                'score': match_basic.get('score'),
                'result_code': match_basic.get('result'),  # 3=主胜, 1=平局, 0=客胜
                'home_score': match_basic.get('homeScore'),
                'away_score': match_basic.get('awayScore')
            }
        
        # 构建详细信息文件路径
        details_path = os.path.join(self.data_root, season, "details")
        
        # 查找比赛详细信息文件
        round_dirs = [d for d in os.listdir(details_path) if os.path.isdir(os.path.join(details_path, d))]
        match_details = None
        
        for round_dir in round_dirs:
            detail_file = os.path.join(details_path, round_dir, f"{match_id}.json")
            if os.path.exists(detail_file):
                with open(detail_file, 'r', encoding='utf-8') as f:
                    match_details = json.load(f)
                break
        
        if not match_details:
            return odds_features  # 即使没有赔率数据，也要返回包含赛果信息的特征
        
        # 检查是否存在赔率数据
        if 'odds' not in match_details:
            return odds_features  # 即使没有赔率数据，也要返回包含赛果信息的特征
        
        odds_data = match_details['odds']
        
        for bookmaker_id, bookmaker_name in self.bookmakers.items():
            if bookmaker_id in odds_data:
                bookmaker_odds = odds_data[bookmaker_id]
                odds_features['bookmakers'][bookmaker_name] = {
                    'initial_odds': None,
                    'closing_odds': None,
                    'odds_changes': [],
                    'average_payout_rate': None,
                    'average_implied_probability': None,
                    'odds_change_frequency': None,
                    'max_odds': None,
                    'min_odds': None,
                    'odds_trend': None,
                    'kelly_index_features': None
                }
                
                # 按时间排序赔率（最早的在前，最新的在后）
                sorted_odds = sorted(bookmaker_odds, key=lambda x: datetime.strptime(x[6], "%Y-%m-%d %H:%M"))
                
                if sorted_odds:
                    # 提取初始赔率和临场赔率
                    initial_odds = sorted_odds[0]
                    closing_odds = sorted_odds[-1]
                    
                    # 计算初始赔率的赔付率和隐含概率
                    payout_rate_initial = self.calculate_payout_rate(initial_odds[0], initial_odds[1], initial_odds[2])
                    # 计算隐含概率
                    implied_prob_win_initial = self.calculate_implied_probability(payout_rate_initial, initial_odds[0])
                    implied_prob_draw_initial = self.calculate_implied_probability(payout_rate_initial, initial_odds[1])
                    implied_prob_lose_initial = self.calculate_implied_probability(payout_rate_initial, initial_odds[2])
                    
                    odds_features['bookmakers'][bookmaker_name]['initial_odds'] = {
                        'win': initial_odds[0],
                        'draw': initial_odds[1],
                        'lose': initial_odds[2],
                        'kelly_index': {
                            'win': initial_odds[3],  # 胜凯利指数
                            'draw': initial_odds[4],  # 平凯利指数
                            'lose': initial_odds[5]   # 负凯利指数
                        },
                        'time': initial_odds[6],
                        'payout_rate': payout_rate_initial,
                        'implied_probability': {
                            'win': implied_prob_win_initial,
                            'draw': implied_prob_draw_initial,
                            'lose': implied_prob_lose_initial
                        },
                        'variance': self.calculate_odds_variance([initial_odds[0], initial_odds[1], initial_odds[2]])
                    }
                    
                    payout_rate_closing = self.calculate_payout_rate(closing_odds[0], closing_odds[1], closing_odds[2])
                    # 计算隐含概率
                    implied_prob_win_closing = self.calculate_implied_probability(payout_rate_closing, closing_odds[0])
                    implied_prob_draw_closing = self.calculate_implied_probability(payout_rate_closing, closing_odds[1])
                    implied_prob_lose_closing = self.calculate_implied_probability(payout_rate_closing, closing_odds[2])
                    
                    odds_features['bookmakers'][bookmaker_name]['closing_odds'] = {
                        'win': closing_odds[0],
                        'draw': closing_odds[1],
                        'lose': closing_odds[2],
                        'kelly_index': {
                            'win': closing_odds[3],  # 胜凯利指数
                            'draw': closing_odds[4],  # 平凯利指数
                            'lose': closing_odds[5]   # 负凯利指数
                        },
                        'time': closing_odds[6],
                        'payout_rate': payout_rate_closing,
                        'implied_probability': {
                            'win': implied_prob_win_closing,
                            'draw': implied_prob_draw_closing,
                            'lose': implied_prob_lose_closing
                        },
                        'variance': self.calculate_odds_variance([closing_odds[0], closing_odds[1], closing_odds[2]])
                    }
                    
                    # 计算赔率变化率（临场vs初赔）
                    try:
                        initial_win = float(initial_odds[0])
                        initial_draw = float(initial_odds[1])
                        initial_lose = float(initial_odds[2])
                        
                        closing_win = float(closing_odds[0])
                        closing_draw = float(closing_odds[1])
                        closing_lose = float(closing_odds[2])
                        
                        odds_features['bookmakers'][bookmaker_name]['odds_change_rate'] = {
                            'win': (closing_win - initial_win) / initial_win if initial_win != 0 else None,
                            'draw': (closing_draw - initial_draw) / initial_draw if initial_draw != 0 else None,
                            'lose': (closing_lose - initial_lose) / initial_lose if initial_lose != 0 else None
                        }
                    except (ValueError, ZeroDivisionError):
                        odds_features['bookmakers'][bookmaker_name]['odds_change_rate'] = None
                    
                    # 提取赔率变化频率
                    odds_features['bookmakers'][bookmaker_name]['odds_change_frequency'] = len(sorted_odds)
                    
                    # 提取最大/最小赔率
                    win_odds_list = [float(odds[0]) for odds in sorted_odds if odds[0] and float(odds[0]) > 0]
                    draw_odds_list = [float(odds[1]) for odds in sorted_odds if odds[1] and float(odds[1]) > 0]
                    lose_odds_list = [float(odds[2]) for odds in sorted_odds if odds[2] and float(odds[2]) > 0]
                    
                    max_odds = {}
                    min_odds = {}
                    
                    if win_odds_list:
                        max_odds['win'] = max(win_odds_list)
                        min_odds['win'] = min(win_odds_list)
                    
                    if draw_odds_list:
                        max_odds['draw'] = max(draw_odds_list)
                        min_odds['draw'] = min(draw_odds_list)
                    
                    if lose_odds_list:
                        max_odds['lose'] = max(lose_odds_list)
                        min_odds['lose'] = min(lose_odds_list)
                    
                    odds_features['bookmakers'][bookmaker_name]['max_odds'] = max_odds
                    odds_features['bookmakers'][bookmaker_name]['min_odds'] = min_odds
                    
                    # 提取趋势特征
                    odds_trend = {}
                    
                    if win_odds_list:
                        win_trend, win_trend_strength = self.calculate_trend(win_odds_list)
                        odds_trend['win'] = {
                            'trend': win_trend,
                            'strength': win_trend_strength
                        }
                    
                    if draw_odds_list:
                        draw_trend, draw_trend_strength = self.calculate_trend(draw_odds_list)
                        odds_trend['draw'] = {
                            'trend': draw_trend,
                            'strength': draw_trend_strength
                        }
                    
                    if lose_odds_list:
                        lose_trend, lose_trend_strength = self.calculate_trend(lose_odds_list)
                        odds_trend['lose'] = {
                            'trend': lose_trend,
                            'strength': lose_trend_strength
                        }
                    
                    odds_features['bookmakers'][bookmaker_name]['odds_trend'] = odds_trend
                    
                    # 提取凯利指数特征
                    kelly_win_list = [float(odds[3]) for odds in sorted_odds if odds[3] and float(odds[3]) >= 0]
                    kelly_draw_list = [float(odds[4]) for odds in sorted_odds if odds[4] and float(odds[4]) >= 0]
                    kelly_lose_list = [float(odds[5]) for odds in sorted_odds if odds[5] and float(odds[5]) >= 0]
                    
                    kelly_index_features = {}
                    
                    if kelly_win_list:
                        kelly_index_features['win'] = {
                            'initial': kelly_win_list[0],
                            'closing': kelly_win_list[-1],
                            'average': sum(kelly_win_list) / len(kelly_win_list),
                            'max': max(kelly_win_list),
                            'min': min(kelly_win_list)
                        }
                    
                    if kelly_draw_list:
                        kelly_index_features['draw'] = {
                            'initial': kelly_draw_list[0],
                            'closing': kelly_draw_list[-1],
                            'average': sum(kelly_draw_list) / len(kelly_draw_list),
                            'max': max(kelly_draw_list),
                            'min': min(kelly_draw_list)
                        }
                    
                    if kelly_lose_list:
                        kelly_index_features['lose'] = {
                            'initial': kelly_lose_list[0],
                            'closing': kelly_lose_list[-1],
                            'average': sum(kelly_lose_list) / len(kelly_lose_list),
                            'max': max(kelly_lose_list),
                            'min': min(kelly_lose_list)
                        }
                    
                    odds_features['bookmakers'][bookmaker_name]['kelly_index_features'] = kelly_index_features
                    
                    # 处理赔率变化记录
                    payout_rates = []
                    implied_probs_win = []
                    implied_probs_draw = []
                    implied_probs_lose = []
                    
                    for odds in sorted_odds:
                        payout_rate = self.calculate_payout_rate(odds[0], odds[1], odds[2])
                        # 计算隐含概率
                        implied_prob_win = self.calculate_implied_probability(payout_rate, odds[0])
                        implied_prob_draw = self.calculate_implied_probability(payout_rate, odds[1])
                        implied_prob_lose = self.calculate_implied_probability(payout_rate, odds[2])
                        
                        odds_features['bookmakers'][bookmaker_name]['odds_changes'].append({
                            'win': odds[0],
                            'draw': odds[1],
                            'lose': odds[2],
                            'kelly_index': {
                                'win': odds[3],  # 胜凯利指数
                                'draw': odds[4],  # 平凯利指数
                                'lose': odds[5]   # 负凯利指数
                            },
                            'time': odds[6],
                            'payout_rate': payout_rate,
                            'implied_probability': {
                                'win': implied_prob_win,
                                'draw': implied_prob_draw,
                                'lose': implied_prob_lose
                            },
                            'variance': self.calculate_odds_variance([odds[0], odds[1], odds[2]])
                        })
                        
                        if payout_rate:
                            payout_rates.append(payout_rate)
                        
                        if implied_prob_win:
                            implied_probs_win.append(implied_prob_win)
                        
                        if implied_prob_draw:
                            implied_probs_draw.append(implied_prob_draw)
                        
                        if implied_prob_lose:
                            implied_probs_lose.append(implied_prob_lose)
                    
                    # 计算平均赔付率和平均隐含概率
                    if payout_rates:
                        odds_features['bookmakers'][bookmaker_name]['average_payout_rate'] = sum(payout_rates) / len(payout_rates)
                    
                    if implied_probs_win or implied_probs_draw or implied_probs_lose:
                        average_implied_prob = {
                            'win': sum(implied_probs_win) / len(implied_probs_win) if implied_probs_win else None,
                            'draw': sum(implied_probs_draw) / len(implied_probs_draw) if implied_probs_draw else None,
                            'lose': sum(implied_probs_lose) / len(implied_probs_lose) if implied_probs_lose else None
                        }
                        odds_features['bookmakers'][bookmaker_name]['average_implied_probability'] = average_implied_prob
        
        # 计算不同博彩公司之间的赔率差异
        bookmaker_names = list(odds_features['bookmakers'].keys())
        if len(bookmaker_names) >= 2:
            odds_features['bookmaker_odds_difference'] = {}
            
            for i in range(len(bookmaker_names)):
                for j in range(i+1, len(bookmaker_names)):
                    bm1 = bookmaker_names[i]
                    bm2 = bookmaker_names[j]
                    
                    if odds_features['bookmakers'][bm1]['closing_odds'] and odds_features['bookmakers'][bm2]['closing_odds']:
                        bm1_odds = odds_features['bookmakers'][bm1]['closing_odds']
                        bm2_odds = odds_features['bookmakers'][bm2]['closing_odds']
                        
                        difference = {}
                        try:
                            difference['win'] = abs(float(bm1_odds['win']) - float(bm2_odds['win']))
                            difference['draw'] = abs(float(bm1_odds['draw']) - float(bm2_odds['draw']))
                            difference['lose'] = abs(float(bm1_odds['lose']) - float(bm2_odds['lose']))
                        except (ValueError, TypeError):
                            pass
                        
                        if difference:
                            odds_features['bookmaker_odds_difference'][f"{bm1}_vs_{bm2}"] = difference
        
        return odds_features

    def extract_season_odds_features(self, season):
        """提取整个赛季的赔率特征"""
        season_dir = os.path.join(self.data_root, season)
        if not os.path.exists(season_dir):
            return None
        
        # 读取赛季比赛列表
        season_files = [f for f in os.listdir(season_dir) if f.endswith('.json') and not f.startswith('details')]
        if not season_files:
            return None
        
        season_features = {
            'season': season,
            'matches': []
        }
        
        for season_file in season_files:
            file_path = os.path.join(season_dir, season_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                season_data = json.load(f)
            
            for match_id, match_info in season_data.items():
                print(f"正在提取比赛 {match_id} 的赔率特征...")
                match_features = self.extract_match_odds_features(season, match_id, season_data)
                if match_features:
                    season_features['matches'].append(match_features)
        
        return season_features

    def save_features(self, features, output_path):
        """保存特征到JSON文件"""
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(features, f, ensure_ascii=False, indent=2)

# 使用示例
if __name__ == "__main__":
    data_root = "/Users/Williamhiler/Documents/my-project/train/original-data"
    train_data_root = "/Users/Williamhiler/Documents/my-project/train/train-data"
    extractor = OddsFeatureExtractor(data_root)
    
    # 获取所有赛季目录
    season_dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    
    for season in season_dirs:
        print(f"正在处理赛季 {season} 的数据...")
        season_features = extractor.extract_season_odds_features(season)
        
        if season_features:
            output_path = os.path.join(train_data_root, "odds", f"{season}_odds_features.json")
            extractor.save_features(season_features, output_path)
            print(f"赛季 {season} 的赔率特征已保存到 {output_path}")
        else:
            print(f"未找到赛季 {season} 的数据")
    
    print("所有赛季数据处理完成！")