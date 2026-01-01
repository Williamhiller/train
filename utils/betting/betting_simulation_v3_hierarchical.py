import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from trainers.hierarchical.hierarchical_trainer import HierarchicalModelTrainer
from trainers.data_loader import DataLoader

class HierarchicalBettingSimulator:
    def __init__(self, data_root, model_dir, model_version="4.0.1", data_season="2025-2026"):
        self.data_root = data_root
        self.model_dir = model_dir
        self.model_version = model_version
        self.data_season = data_season
        self.data_loader = DataLoader(data_root)
        
        # 结果映射：[0, 1, 2] -> ['客胜', '平局', '主胜']
        self.result_mapping = {0: '客胜', 1: '平局', 2: '主胜'}
        
        # 加载模型和配置
        self.draw_model, self.win_loss_model, self.scaler, self.features, self.best_params = self.load_model()
        
        # 加载测试数据
        self.test_data = self.load_test_data()
        
        # 模型阈值
        self.draw_threshold = 0.48
        self.home_win_threshold = 0.56
    
    def load_model(self):
        """加载分层模型的两个子模型"""
        # 根据模型版本选择目录
        if self.model_version.startswith('4'):
            version_dir = os.path.join(self.model_dir, "v4", self.model_version)
        else:
            version_dir = os.path.join(self.model_dir, "v3", self.model_version)
        
        # 加载平局模型
        draw_model_path = os.path.join(version_dir, f"draw_model_{self.model_version}.joblib")
        draw_model = joblib.load(draw_model_path)
        
        # 加载胜负模型
        win_loss_model_path = os.path.join(version_dir, f"win_loss_model_{self.model_version}.joblib")
        win_loss_model = joblib.load(win_loss_model_path)
        
        # 加载scaler
        scaler_path = os.path.join(version_dir, f"scaler_{self.model_version}.joblib")
        scaler = joblib.load(scaler_path)
        
        # 加载features
        features_path = os.path.join(version_dir, f"features_{self.model_version}.joblib")
        features = joblib.load(features_path)
        
        # 加载best_params.json
        best_params_path = os.path.join(os.path.dirname(__file__), '../../trainers/hierarchical/best_params.json')
        with open(best_params_path, 'r') as f:
            best_params = json.load(f)
        
        return draw_model, win_loss_model, scaler, features, best_params
    
    def load_test_data(self):
        """加载2025-2026赛季的测试数据"""
        print(f"\n=== 加载{self.data_season}赛季测试数据 ===")
        
        # 直接加载赛季数据文件
        season_file = os.path.join(self.data_root, self.data_season, f"36_{self.data_season}.json")
        
        if not os.path.exists(season_file):
            raise FileNotFoundError(f"赛季数据文件不存在: {season_file}")
        
        # 加载完整的赛季数据
        with open(season_file, 'r', encoding='utf-8') as f:
            season_data = json.load(f)
        
        # 转换为DataFrame
        test_df = pd.DataFrame.from_dict(season_data, orient='index')
        
        # 重命名列以匹配预期格式
        test_df.rename(columns={
            'matchId': 'match_id',
            'homeTeamName': 'home_team',
            'awayTeamName': 'away_team',
            'result': 'result_code'
        }, inplace=True)
        
        # 转换result_code为与模型预期一致的格式
        # 原始数据中：3=主胜, 1=平局, 0=客胜
        # 模型预期：2=主胜, 1=平局, 0=客胜
        test_df['result_code'] = test_df['result_code'].replace({3: 2})
        
        # 加载赔率数据
        self.load_odds_data(test_df)
        
        # 为模型添加默认特征值
        # 这里我们使用训练时的特征列表，为每个特征添加默认值0
        for feature in self.features:
            if feature not in test_df.columns:
                test_df[feature] = 0.0
        
        print(f"成功加载{len(test_df)}场比赛数据")
        
        return test_df
    
    def load_odds_data(self, test_df):
        """加载详细的赔率数据并添加到测试数据中"""
        # 为所有比赛添加默认赔率列
        test_df['closing_win_odds'] = 1.8  # 主胜赔率
        test_df['closing_draw_odds'] = 3.5  # 平局赔率
        test_df['closing_lose_odds'] = 4.0  # 客胜赔率
        
        details_dir = os.path.join(self.data_root, self.data_season, "details")
        
        # 遍历所有轮次目录
        for round_dir in os.listdir(details_dir):
            round_path = os.path.join(details_dir, round_dir)
            if os.path.isdir(round_path):
                # 遍历该轮次的所有比赛文件
                for match_file in os.listdir(round_path):
                    if match_file.endswith('.json'):
                        match_id = int(match_file.replace('.json', ''))
                        match_path = os.path.join(round_path, match_file)
                        
                        with open(match_path, 'r', encoding='utf-8') as f:
                            match_data = json.load(f)
                        
                        # 提取威廉希尔（oddId 82）的终赔
                        if 'odds' in match_data and '82' in match_data['odds']:
                            # 获取最新的赔率（列表最后一个元素）
                            latest_odds = match_data['odds']['82'][-1]
                            
                            # 更新对应比赛的赔率
                            if match_id in test_df['match_id'].values:
                                win_odds = float(latest_odds[0])
                                draw_odds = float(latest_odds[1])
                                lose_odds = float(latest_odds[2])
                                
                                test_df.loc[test_df['match_id'] == match_id, 'closing_win_odds'] = win_odds
                                test_df.loc[test_df['match_id'] == match_id, 'closing_draw_odds'] = draw_odds
                                test_df.loc[test_df['match_id'] == match_id, 'closing_lose_odds'] = lose_odds
        
        # 为模型需要的所有赔率特征添加值
        # 由于我们只有终盘赔率，所以用终盘赔率作为所有赔率特征的替代值
        for feature in self.features:
            if feature not in test_df.columns:
                # 如果是赔率相关特征，使用终盘赔率作为替代
                if 'win_odds' in feature:
                    test_df[feature] = test_df['closing_win_odds']
                elif 'draw_odds' in feature:
                    test_df[feature] = test_df['closing_draw_odds']
                elif 'lose_odds' in feature:
                    test_df[feature] = test_df['closing_lose_odds']
                elif 'payout_rate' in feature:
                    # 计算赔付率：1 / (1/主胜赔率 + 1/平局赔率 + 1/客胜赔率)
                    test_df[feature] = 1 / (1/test_df['closing_win_odds'] + 1/test_df['closing_draw_odds'] + 1/test_df['closing_lose_odds'])
                elif 'implied' in feature and 'win' in feature:
                    # 计算隐含主胜概率
                    test_df[feature] = 1 / test_df['closing_win_odds']
                elif 'implied' in feature and 'draw' in feature:
                    # 计算隐含平局概率
                    test_df[feature] = 1 / test_df['closing_draw_odds']
                elif 'implied' in feature and 'lose' in feature:
                    # 计算隐含客胜概率
                    test_df[feature] = 1 / test_df['closing_lose_odds']
                elif 'kelly' in feature and 'win' in feature:
                    # 计算主胜凯利指数
                    test_df[feature] = (test_df['closing_win_odds'] - 1) / test_df['closing_win_odds']
                elif 'kelly' in feature and 'draw' in feature:
                    # 计算平局凯利指数
                    test_df[feature] = (test_df['closing_draw_odds'] - 1) / test_df['closing_draw_odds']
                elif 'kelly' in feature and 'lose' in feature:
                    # 计算客胜凯利指数
                    test_df[feature] = (test_df['closing_lose_odds'] - 1) / test_df['closing_lose_odds']
                else:
                    # 其他特征使用默认值0
                    test_df[feature] = 0.0
    
    def predict_match(self, match):
        """预测单场比赛结果，并计算模型优势和置信度"""
        try:
            # 提取特征 - 只选择模型需要的特征
            X = match[self.features]

            # 确保X是正确的格式
            X_values = X.values.reshape(1, -1)
            
            # 使用scaler进行特征缩放
            X_scaled = self.scaler.transform(X_values)
            
            # 预测平局概率
            draw_proba = self.draw_model.predict_proba(X_scaled)[0, 1]
            non_draw_proba = self.draw_model.predict_proba(X_scaled)[0, 0]
            
            # 预测胜负概率
            win_loss_proba = self.win_loss_model.predict_proba(X_scaled)[0, 1]
            home_win_proba = win_loss_proba
            away_win_proba = 1 - win_loss_proba
            
            # 使用模型训练时的最佳阈值
            draw_threshold = self.draw_threshold
            home_win_threshold = self.home_win_threshold
            
            # 计算所有结果的概率
            probabilities = {
                0: away_win_proba * non_draw_proba,  # 客胜概率 = 非平局概率 * 客胜概率
                1: draw_proba,  # 平局概率
                2: home_win_proba * non_draw_proba   # 主胜概率 = 非平局概率 * 主胜概率
            }
            
            # 归一化概率
            total_proba = sum(probabilities.values())
            if total_proba > 0:
                for key in probabilities:
                    probabilities[key] /= total_proba
            
            # 组合预测结果
            predicted_result = max(probabilities, key=probabilities.get)
            predicted_proba = probabilities[predicted_result]
            
            # 计算置信度（最大概率与次大概率的差值）
            sorted_probas = sorted(probabilities.values(), reverse=True)
            if len(sorted_probas) > 1:
                confidence = sorted_probas[0] - sorted_probas[1]
            else:
                confidence = 0
            
            # 计算模型优势（模型概率 - 隐含概率）
            win_odds = match['closing_win_odds']
            draw_odds = match['closing_draw_odds']
            lose_odds = match['closing_lose_odds']
            
            # 计算赔付率
            payout_rate = 1 / (1/win_odds + 1/draw_odds + 1/lose_odds)
            
            # 计算隐含概率
            implied_home_prob = (1 / win_odds) * payout_rate
            implied_draw_prob = (1 / draw_odds) * payout_rate
            implied_away_prob = (1 / lose_odds) * payout_rate
            
            # 计算模型优势
            home_edge = probabilities[2] - implied_home_prob
            draw_edge = probabilities[1] - implied_draw_prob
            away_edge = probabilities[0] - implied_away_prob
            
            return predicted_result, predicted_proba, probabilities[0], probabilities[1], probabilities[2], confidence, home_edge, draw_edge, away_edge
        except Exception as e:
            # 如果特征提取或预测失败，返回随机预测
            return np.random.choice([0, 1, 2]), 0.5, 0.33, 0.33, 0.33, 0.0, 0.0, 0.0, 0.0
    
    def simulate_betting(self, bet_amount=100, use_kelly=True, strategy='kelly'):
        """模拟投注过程，支持固定投注、凯利公式动态投注和direct投注策略"""
        # 支持三种策略：'fixed', 'kelly', 'direct'
        if strategy not in ['fixed', 'kelly', 'direct']:
            strategy = 'kelly'  # 默认策略
        
        print(f"\n=== 开始模拟投注 ===")
        
        initial_match_count = len(self.test_data)
        correct_predictions = 0
        total_bets = 0
        total_profit = 0
        total_winnings = 0
        bet_history = []
        valuable_matches = 0
        
        # 对于direct策略，先过滤出符合条件的比赛
        if strategy == 'direct':
            print(f"赛季: {self.data_season}")
            print(f"模型版本: v{self.model_version}")
            print(f"基准投注金额: ¥10")
            print(f"赔率过滤: 仅投注主胜或客胜赔率在1.8-4.0区间的比赛")
            print(f"投注策略: 基于凯利指数动态调整，最低¥5，最高¥20")
            
            # 过滤出符合条件的比赛：主胜或客胜赔率在1.8-4.0之间
            valid_matches = []
            for _, match in self.test_data.iterrows():
                win_odds = match['closing_win_odds']  # 主胜赔率
                lose_odds = match['closing_lose_odds']  # 客胜赔率
                
                if (1.8 <= win_odds <= 4.0) or (1.8 <= lose_odds <= 4.0):
                    valid_matches.append(match)
            
            print(f"符合赔率条件的比赛: {len(valid_matches)}/{initial_match_count}")
            matches_to_process = valid_matches
        else:
            print(f"赛季: {self.data_season}")
            print(f"模型版本: v{self.model_version}")
            print(f"基础投注金额: ¥{bet_amount}")
            print(f"有价值比赛筛选条件: 模型优势>0.05且置信度>0.1")
            print(f"投注策略: {'凯利公式动态投注' if use_kelly else '对所有符合条件的比赛投注固定金额'}")
            matches_to_process = self.test_data.iterrows()
        
        if strategy == 'direct':
            for i, match in enumerate(matches_to_process):
                # 获取比赛信息
                match_id = match['match_id']
                home_team = match['home_team']
                away_team = match['away_team']
                actual_result = match['result_code']
                
                # 获取赔率
                win_odds = match['closing_win_odds']  # 主胜赔率
                draw_odds = match['closing_draw_odds']  # 平局赔率
                lose_odds = match['closing_lose_odds']  # 客胜赔率
                
                # 预测比赛结果，计算模型优势和置信度
                predicted_result, predicted_proba, away_win_proba, draw_proba, home_win_proba, confidence, home_edge, draw_edge, away_edge = self.predict_match(match)
                
                # 确定当前投注的赔率和概率
                if predicted_result == 0:  # 客胜
                    bet_odds = lose_odds
                    predicted_proba = away_win_proba
                elif predicted_result == 1:  # 平局
                    bet_odds = draw_odds
                    predicted_proba = draw_proba
                else:  # 主胜
                    bet_odds = win_odds
                    predicted_proba = home_win_proba
                
                # 计算凯利比例
                b_kelly = bet_odds - 1  # 净赔率
                p = predicted_proba  # 获胜概率
                q = 1 - p  # 失败概率
                
                # 计算凯利比例
                if b_kelly > 0:
                    kelly_ratio = max(0, (b_kelly * p - q) / b_kelly)  # 确保比例非负
                else:
                    kelly_ratio = 0
                
                # 限制最大凯利比例为1.0
                max_kelly_ratio = 1.0
                kelly_ratio = min(kelly_ratio, max_kelly_ratio)
                
                # 基于凯利指数动态调整投注金额
                # 基准线为10元，根据凯利比例调整，最低5元，最高20元
                base_bet_amount = 10
                adjustment_factor = kelly_ratio / max_kelly_ratio
                dynamic_bet_amount = base_bet_amount + (adjustment_factor * (20 - base_bet_amount))
                
                # 应用最低和最高限制
                dynamic_bet_amount = max(dynamic_bet_amount, 5.0)
                dynamic_bet_amount = min(dynamic_bet_amount, 20.0)
                dynamic_bet_amount = round(dynamic_bet_amount)
                
                # 计算盈利
                b = bet_odds - 1  # 净赔率
                if predicted_result == actual_result:
                    correct_predictions += 1
                    profit = dynamic_bet_amount * b  # 净盈利
                else:
                    profit = -dynamic_bet_amount  # 亏损
                
                # 更新总盈利和总投入
                total_profit += profit
                total_winnings += dynamic_bet_amount
                total_bets += 1
                
                # 记录投注历史
                bet_history.append({
                    'match_id': match_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'actual_result': self.result_mapping[actual_result],
                    'predicted_result': self.result_mapping[predicted_result],
                    'predicted_proba': float(predicted_proba),
                    'away_win_proba': float(away_win_proba),
                    'draw_proba': float(draw_proba),
                    'home_win_proba': float(home_win_proba),
                    'confidence': float(confidence),
                    'home_edge': float(home_edge),
                    'draw_edge': float(draw_edge),
                    'away_edge': float(away_edge),
                    'win_odds': float(win_odds),
                    'draw_odds': float(draw_odds),
                    'lose_odds': float(lose_odds),
                    'bet_odds': float(bet_odds),
                    'kelly_ratio': float(kelly_ratio),
                    'bet_amount': float(dynamic_bet_amount),
                    'profit': float(profit),
                    'total_profit': float(total_profit),
                    'is_correct': bool(predicted_result == actual_result)
                })
                
                # 打印进度
                print(f"\r处理比赛 {i+1}/{len(valid_matches)}...", end="", flush=True)
        else:
            for index, match in matches_to_process:
                # 获取比赛信息
                match_id = match['match_id']
                home_team = match['home_team']
                away_team = match['away_team']
                actual_result = match['result_code']
                
                # 获取赔率
                win_odds = match['closing_win_odds']  # 主胜赔率
                draw_odds = match['closing_draw_odds']  # 平局赔率
                lose_odds = match['closing_lose_odds']  # 客胜赔率
                
                # 预测比赛结果，计算模型优势和置信度
                predicted_result, predicted_proba, away_win_proba, draw_proba, home_win_proba, confidence, home_edge, draw_edge, away_edge = self.predict_match(match)
                
                # 计算所有结果的优势
                edges = {
                    0: away_edge,
                    1: draw_edge,
                    2: home_edge
                }
                
                # 确定当前预测的最大优势
                max_edge = edges[predicted_result]
                
                # 筛选有价值的比赛
                # 条件：模型优势>0.05且置信度>0.1
                if max_edge <= 0.05 or confidence <= 0.1:
                    continue
                
                valuable_matches += 1
                
                # 确定当前投注的赔率
                if predicted_result == 0:  # 客胜
                    bet_odds = lose_odds
                    predicted_proba = away_win_proba
                elif predicted_result == 1:  # 平局
                    bet_odds = draw_odds
                    predicted_proba = draw_proba
                else:  # 主胜
                    bet_odds = win_odds
                    predicted_proba = home_win_proba
                
                # 计算凯利比率
                kelly_ratio = 0.0
                if use_kelly:
                    # 凯利公式：f* = (bp - q) / b，其中b是净赔率，p是获胜概率，q是失败概率
                    b_kelly = bet_odds - 1  # 净赔率
                    p = predicted_proba  # 获胜概率
                    q = 1 - p  # 失败概率
                    
                    # 计算凯利比率
                    if b_kelly > 0:
                        kelly_ratio = (b_kelly * p - q) / b_kelly
                        # 限制凯利比率在0-1之间
                        kelly_ratio = max(0.0, min(0.2, kelly_ratio))  # 最高投注20%
                    else:
                        kelly_ratio = 0.0
                
                # 计算动态投注金额
                if use_kelly:
                    # 使用凯利比率计算动态投注金额
                    dynamic_bet_amount = bet_amount * kelly_ratio
                else:
                    # 使用固定投注金额
                    dynamic_bet_amount = bet_amount
                
                # 确保投注金额至少为1元
                dynamic_bet_amount = max(1.0, dynamic_bet_amount)
                
                # 计算盈利
                b = bet_odds - 1  # 净赔率
                if predicted_result == actual_result:
                    correct_predictions += 1
                    profit = dynamic_bet_amount * b  # 净盈利
                else:
                    profit = -dynamic_bet_amount  # 亏损
                
                # 更新总盈利和总投入
                total_profit += profit
                total_winnings += dynamic_bet_amount
                total_bets += 1
                
                # 记录投注历史
                bet_history.append({
                    'match_id': match_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'actual_result': self.result_mapping[actual_result],
                    'predicted_result': self.result_mapping[predicted_result],
                    'predicted_proba': float(predicted_proba),
                    'away_win_proba': float(away_win_proba),
                    'draw_proba': float(draw_proba),
                    'home_win_proba': float(home_win_proba),
                    'confidence': float(confidence),
                    'home_edge': float(home_edge),
                    'draw_edge': float(draw_edge),
                    'away_edge': float(away_edge),
                    'win_odds': float(win_odds),
                    'draw_odds': float(draw_odds),
                    'lose_odds': float(lose_odds),
                    'bet_odds': float(bet_odds),
                    'kelly_ratio': float(kelly_ratio),
                    'bet_amount': float(dynamic_bet_amount),
                    'profit': float(profit),
                    'total_profit': float(total_profit),
                    'is_correct': bool(predicted_result == actual_result)
                })
                
                # 打印进度
                print(f"\r处理比赛 {total_bets}/{initial_match_count}... 已找到{valuable_matches}场有价值比赛", end="", flush=True)
        
        # 计算准确率
        accuracy = correct_predictions / total_bets if total_bets > 0 else 0
        
        # 计算收益率
        roi = (total_profit / total_winnings) * 100 if total_winnings > 0 else 0
        
        print(f"\r{' ' * 50}\r", end="")  # 清空进度信息
        
        return {
            'initial_match_count': initial_match_count,
            'valuable_matches': valuable_matches,
            'total_bets': total_bets,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'total_winnings': total_winnings,
            'total_profit': total_profit,
            'roi': roi,
            'bet_history': bet_history
        }
    
    def generate_report(self, simulation_result, strategy='kelly'):
        """生成投注报告，支持direct策略"""
        bet_history = simulation_result['bet_history']
        
        # 检测是否使用了凯利公式或direct策略
        use_kelly = any('kelly_ratio' in bet and bet['kelly_ratio'] > 0 for bet in bet_history)
        
        print(f"\n=== 投注模拟报告 ===")
        print(f"赛季: {self.data_season}")
        print(f"模型版本: v{self.model_version}")
        print(f"投注策略: {'direct策略' if strategy == 'direct' else '模型优势>0.05且置信度>0.1'}")
        print(f"投注方式: {'direct动态投注' if strategy == 'direct' else ('凯利公式动态投注' if use_kelly else '固定金额投注')}")
        print(f"\n总比赛数: {simulation_result['initial_match_count']}")
        print(f"有价值比赛数: {simulation_result['valuable_matches']}")
        print(f"实际投注场次: {simulation_result['total_bets']}")
        print(f"总正确率: {simulation_result['accuracy'] * 100:.2f}%")
        print(f"总投入金额: ¥{simulation_result['total_winnings']:.2f}")
        print(f"总盈利: ¥{simulation_result['total_profit']:.2f}")
        print(f"收益率: {simulation_result['roi']:.2f}%")
        
        # 输出详细的投注历史到文件
        output_dir = f"/Users/Williamhiler/Documents/my-project/train/test_data/{self.data_season}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 确定输出文件名（包含策略信息）
        if strategy == 'direct':
            strategy_suffix = "direct"
        elif use_kelly:
            strategy_suffix = "kelly"
        else:
            strategy_suffix = "fixed100"
        
        # 保存为JSON格式
        result_json_path = os.path.join(output_dir, f"betting_result_{self.model_version}_{strategy_suffix}.json")
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(simulation_result, f, ensure_ascii=False, indent=2)
        
        # 保存为CSV格式，方便查看
        bet_df = pd.DataFrame(bet_history)
        result_csv_path = os.path.join(output_dir, f"betting_result_{self.model_version}_{strategy_suffix}.csv")
        bet_df.to_csv(result_csv_path, index=False, encoding='utf-8')
        
        print(f"\n投注结果已保存至:")
        print(f"JSON格式: {result_json_path}")
        print(f"CSV格式: {result_csv_path}")
        
        # 打印前10条投注记录
        print(f"\n前10条投注记录:")
        
        if strategy == 'direct':
            print("-" * 200)
            print(f"{'比赛ID':<10} {'主队':<15} {'客队':<15} {'实际':<8} {'预测':<8} {'赔率':<8} {'模型优势':<10} {'置信度':<10} {'凯利比率':<10} {'投注额':<10} {'盈利':<10} {'累计盈利':<12}")
            print("-" * 200)
            for _, bet in bet_df.head(10).iterrows():
                # 计算当前投注的模型优势
                if bet['predicted_result'] == '客胜':
                    edge = bet['away_edge']
                elif bet['predicted_result'] == '平局':
                    edge = bet['draw_edge']
                else:
                    edge = bet['home_edge']
                
                print(f"{bet['match_id']:<10} {bet['home_team']:<15} {bet['away_team']:<15} {bet['actual_result']:<8} {bet['predicted_result']:<8} {bet['bet_odds']:<8.2f} {edge:<10.2%} {bet['confidence']:<10.2%} {bet['kelly_ratio']:<10.4f} {bet['bet_amount']:<10.2f} {bet['profit']:<10.2f} {bet['total_profit']:<12.2f}")
            print("-" * 200)
        elif use_kelly:
            print("-" * 200)
            print(f"{'比赛ID':<10} {'主队':<15} {'客队':<15} {'实际':<8} {'预测':<8} {'赔率':<8} {'模型优势':<10} {'置信度':<10} {'凯利比率':<10} {'投注额':<10} {'盈利':<10} {'累计盈利':<12}")
            print("-" * 200)
            for _, bet in bet_df.head(10).iterrows():
                # 计算当前投注的模型优势
                if bet['predicted_result'] == '客胜':
                    edge = bet['away_edge']
                elif bet['predicted_result'] == '平局':
                    edge = bet['draw_edge']
                else:
                    edge = bet['home_edge']
                
                print(f"{bet['match_id']:<10} {bet['home_team']:<15} {bet['away_team']:<15} {bet['actual_result']:<8} {bet['predicted_result']:<8} {bet['bet_odds']:<8.2f} {edge:<10.2%} {bet['confidence']:<10.2%} {bet['kelly_ratio']:<10.4f} {bet['bet_amount']:<10.2f} {bet['profit']:<10.2f} {bet['total_profit']:<12.2f}")
            print("-" * 200)
        else:
            print("-" * 160)
            print(f"{'比赛ID':<10} {'主队':<15} {'客队':<15} {'实际':<8} {'预测':<8} {'赔率':<8} {'模型优势':<10} {'置信度':<10} {'投注额':<10} {'盈利':<10} {'累计盈利':<12}")
            print("-" * 160)
            for _, bet in bet_df.head(10).iterrows():
                # 计算当前投注的模型优势
                if bet['predicted_result'] == '客胜':
                    edge = bet['away_edge']
                elif bet['predicted_result'] == '平局':
                    edge = bet['draw_edge']
                else:
                    edge = bet['home_edge']
                
                print(f"{bet['match_id']:<10} {bet['home_team']:<15} {bet['away_team']:<15} {bet['actual_result']:<8} {bet['predicted_result']:<8} {bet['bet_odds']:<8.2f} {edge:<10.2%} {bet['confidence']:<10.2%} {bet['bet_amount']:<10.2f} {bet['profit']:<10.2f} {bet['total_profit']:<12.2f}")
            print("-" * 160)
        
        return result_json_path, result_csv_path
    
    def run_simulation(self, bet_amount=100, use_kelly=True, strategy='kelly'):
        """运行完整的投注模拟，支持固定投注、凯利公式动态投注和direct投注策略"""
        simulation_result = self.simulate_betting(bet_amount, use_kelly, strategy)
        self.generate_report(simulation_result, strategy)
        return simulation_result

import argparse

def main():
    # 配置参数
    data_root = '/Users/Williamhiler/Documents/my-project/train/test_data'
    model_dir = '/Users/Williamhiler/Documents/my-project/models'
    
    # 添加命令行参数支持
    parser = argparse.ArgumentParser(description='分层模型投注模拟')
    parser.add_argument('--model_version', type=str, default='4.0.1', help='模型版本，例如：3.0.7')
    args = parser.parse_args()
    
    # 创建模拟器实例
    simulator = HierarchicalBettingSimulator(
        data_root=data_root,
        model_dir=model_dir,
        model_version=args.model_version,
        data_season="2025-2026"
    )
    
    print("\n" + "="*60)
    print(f"开始运行v{args.model_version}固定金额投注策略")
    print("="*60)
    # 运行固定金额投注策略
    simulator.run_simulation(bet_amount=100, use_kelly=False, strategy='fixed')
    
    print("\n" + "="*60)
    print(f"开始运行v{args.model_version}凯利公式投注策略")
    print("="*60)
    # 运行凯利公式投注策略
    simulator.run_simulation(bet_amount=100, use_kelly=True, strategy='kelly')
    
    print("\n" + "="*60)
    print(f"开始运行v{args.model_version}direct投注策略")
    print("="*60)
    # 运行direct投注策略
    simulator.run_simulation(strategy='direct')

if __name__ == "__main__":
    main()