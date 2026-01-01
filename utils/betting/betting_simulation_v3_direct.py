import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from trainers.model_trainer import ModelTrainerV3
from trainers.data_loader import DataLoader

class DirectBettingSimulator:
    def __init__(self, data_root, model_dir, model_version="3.0.4", data_season="2025-2026", model_name="lightgbm"):
        self.data_root = data_root
        self.model_dir = model_dir
        self.model_version = model_version
        self.data_season = data_season
        self.model_name = model_name
        self.data_loader = DataLoader(data_root)
        
        # 结果映射：[0, 1, 2] -> ['客胜', '平局', '主胜']
        self.result_mapping = {0: '客胜', 1: '平局', 2: '主胜'}
        
        # 加载模型和配置
        self.model, self.features, self.model_info = self.load_model()
        
        # 加载测试数据
        self.test_data = self.load_test_data()
    
    def load_model(self):
        """加载直接多分类模型"""
        version_dir = os.path.join(self.model_dir, "v3", self.model_version)
        
        # 加载模型
        model_filename = f"model_v{self.model_version}_{self.model_name}.joblib"
        model_path = os.path.join(version_dir, model_filename)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        model = joblib.load(model_path)
        
        # 加载模型配置信息
        info_filename = f"model_v{self.model_version}_{self.model_name}_info.json"
        info_path = os.path.join(version_dir, info_filename)
        
        with open(info_path, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        
        # 获取特征列表
        features = model_info['feature_names']
        
        return model, features, model_info
    
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
        
        if not os.path.exists(details_dir):
            print(f"赔率详情目录不存在: {details_dir}")
            return
        
        # 遍历所有轮次目录
        for round_dir in os.listdir(details_dir):
            round_path = os.path.join(details_dir, round_dir)
            if os.path.isdir(round_path):
                # 遍历该轮次的所有比赛文件
                for match_file in os.listdir(round_path):
                    if match_file.endswith('.json'):
                        match_id = int(match_file.replace('.json', ''))
                        match_path = os.path.join(round_path, match_file)
                        
                        try:
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
                        except Exception as e:
                            print(f"处理比赛文件{match_file}时出错: {e}")
        
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
    
    def simulate_betting(self, base_bet_amount=10, use_fixed_amount=False):
        """模拟投注过程"""
        print(f"\n=== 开始模拟投注 ===")
        print(f"赛季: {self.data_season}")
        print(f"模型版本: v{self.model_version}")
        print(f"基准投注金额: ¥{base_bet_amount}")
        print(f"赔率过滤: 仅投注主胜或客胜赔率在1.8-4.0区间的比赛")
        if use_fixed_amount:
            print(f"投注策略: 固定金额投注")
        else:
            print(f"投注策略: 基于凯利指数动态调整，最低¥5，最高¥20")
        
        total_bets = 0
        total_winnings = 0
        total_profit = 0
        correct_predictions = 0
        bet_history = []
        
        # 过滤出符合条件的比赛：主胜或客胜赔率在1.8-4.0之间
        valid_matches = []
        for _, match in self.test_data.iterrows():
            win_odds = match['closing_win_odds']  # 主胜赔率
            lose_odds = match['closing_lose_odds']  # 客胜赔率
            
            if (1.8 <= win_odds <= 4.0) or (1.8 <= lose_odds <= 4.0):
                valid_matches.append(match)
        
        # 初始比赛数
        initial_match_count = len(self.test_data)
        valid_match_count = len(valid_matches)
        
        print(f"符合赔率条件的比赛: {valid_match_count}/{initial_match_count}")
        
        for i, match in enumerate(valid_matches):
            # 获取威廉终赔
            win_odds = match['closing_win_odds']  # 主胜赔率
            draw_odds = match['closing_draw_odds']  # 平局赔率
            lose_odds = match['closing_lose_odds']  # 客胜赔率
            
            total_bets += 1
            
            # 获取比赛信息
            match_id = match['match_id']
            home_team = match['home_team']
            away_team = match['away_team']
            actual_result = match['result_code']
            
            # 预测比赛结果
            predicted_result, predicted_proba = self.predict_match(match)
            
            # 确定当前投注的赔率和概率
            if predicted_result == 0:  # 客胜
                bet_odds = lose_odds
                predicted_prob = predicted_proba[0]
            elif predicted_result == 1:  # 平局
                bet_odds = draw_odds
                predicted_prob = predicted_proba[1]
            else:  # 主胜
                bet_odds = win_odds
                predicted_prob = predicted_proba[2]
            
            # 计算凯利比例
            b = bet_odds - 1  # 净赔率
            p = predicted_prob  # 预测胜率
            q = 1 - p  # 失败率
            
            # 计算凯利比例
            if b > 0:
                kelly_ratio = max(0, (b * p - q) / b)  # 确保比例非负
            else:
                kelly_ratio = 0
            
            # 限制最大凯利比例为1.0，用于调整投注金额
            max_kelly_ratio = 1.0
            kelly_ratio = min(kelly_ratio, max_kelly_ratio)
            
            # 确定投注金额
            if use_fixed_amount:
                # 使用固定金额
                current_bet_amount = base_bet_amount
            else:
                # 基于凯利指数动态调整投注金额
                # 基准线为10元，根据凯利比例调整，最低5元，最高20元
                adjustment_factor = kelly_ratio / max_kelly_ratio
                dynamic_bet_amount = base_bet_amount + (adjustment_factor * (20 - base_bet_amount))
                
                # 应用最低和最高限制
                dynamic_bet_amount = max(dynamic_bet_amount, 5.0)
                dynamic_bet_amount = min(dynamic_bet_amount, 20.0)
                current_bet_amount = round(dynamic_bet_amount)
            
            # 计算盈利
            if predicted_result == actual_result:
                correct_predictions += 1
                profit = current_bet_amount * b  # 净盈利
            else:
                profit = -current_bet_amount  # 亏损
            
            # 更新总盈利和总投入
            total_profit += profit
            total_winnings += current_bet_amount
            
            # 记录投注历史
            bet_history.append({
                'match_id': match_id,
                'home_team': home_team,
                'away_team': away_team,
                'actual_result': self.result_mapping[actual_result],
                'predicted_result': self.result_mapping[predicted_result],
                'predicted_proba': float(predicted_prob),
                'win_odds': float(win_odds),
                'draw_odds': float(draw_odds),
                'lose_odds': float(lose_odds),
                'bet_odds': float(bet_odds),
                'kelly_ratio': float(kelly_ratio),
                'bet_amount': float(current_bet_amount),
                'profit': float(profit),
                'total_profit': float(total_profit),
                'is_correct': bool(predicted_result == actual_result)
            })
            
            # 打印进度
            print(f"\r处理比赛 {total_bets}/{valid_match_count}...", end="", flush=True)
        
        # 计算准确率
        accuracy = correct_predictions / total_bets if total_bets > 0 else 0
        
        # 计算收益率
        roi = (total_profit / total_winnings) * 100 if total_winnings > 0 else 0
        
        print(f"\r{' ' * 50}\r", end="")  # 清空进度信息
        
        return {
            'total_bets': total_bets,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'total_winnings': total_winnings,
            'total_profit': total_profit,
            'roi': roi,
            'bet_history': bet_history
        }
    
    def predict_match(self, match):
        """预测单场比赛结果"""
        try:
            # 提取特征 - 只选择模型需要的特征
            X = match[self.features]

            # 确保X是正确的格式
            X_values = X.values.reshape(1, -1)
            
            # 模型预测
            predicted_result = self.model.predict(X_values)[0]
            predicted_proba = self.model.predict_proba(X_values)[0]
            
            return predicted_result, predicted_proba
        except Exception as e:
            # 如果特征提取或预测失败，返回随机预测
            return np.random.choice([0, 1, 2]), np.array([0.33, 0.33, 0.34])
    
    def generate_report(self, simulation_result):
        """生成投注报告"""
        bet_history = simulation_result['bet_history']
        
        print(f"\n=== 投注模拟报告 ===")
        print(f"赛季: {self.data_season}")
        print(f"模型版本: v{self.model_version}")
        print(f"模型类型: {self.model_name} 直接多分类模型")
        print(f"总场次: {simulation_result['total_bets']}")
        print(f"总正确率: {simulation_result['accuracy'] * 100:.2f}%")
        print(f"总投入金额: ¥{simulation_result['total_winnings']:.2f}")
        print(f"总盈利: ¥{simulation_result['total_profit']:.2f}")
        print(f"收益率: {simulation_result['roi']:.2f}%")
        
        # 输出详细的投注历史到文件
        output_dir = f"/Users/Williamhiler/Documents/my-project/train/test_data/{self.data_season}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为JSON格式
        result_json_path = os.path.join(output_dir, f"betting_result_direct_v{self.model_version}.json")
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(simulation_result, f, ensure_ascii=False, indent=2)
        
        # 保存为CSV格式，方便查看
        bet_df = pd.DataFrame(bet_history)
        result_csv_path = os.path.join(output_dir, f"betting_result_direct_v{self.model_version}.csv")
        bet_df.to_csv(result_csv_path, index=False, encoding='utf-8')
        
        print(f"\n投注结果已保存至:")
        print(f"JSON格式: {result_json_path}")
        print(f"CSV格式: {result_csv_path}")
        
        # 打印前10条投注记录
        print(f"\n前10条投注记录:")
        print("-" * 140)
        print(f"{'比赛ID':<10} {'主队':<15} {'客队':<15} {'实际':<8} {'预测':<8} {'赔率':<8} {'凯利比例':<10} {'投注额':<10} {'盈利':<10} {'累计盈利':<12}")
        print("-" * 140)
        
        for _, bet in bet_df.head(10).iterrows():
            print(f"{bet['match_id']:<10} {bet['home_team']:<15} {bet['away_team']:<15} {bet['actual_result']:<8} {bet['predicted_result']:<8} {bet['bet_odds']:<8.2f} {bet['kelly_ratio']:<10.2%} {bet['bet_amount']:<10.2f} {bet['profit']:<10.2f} {bet['total_profit']:<12.2f}")
        
        print("-" * 140)
        
        return result_json_path, result_csv_path
    
    def run_simulation(self, bet_amount=10, use_fixed_amount=False):
        """运行完整的投注模拟"""
        simulation_result = self.simulate_betting(bet_amount, use_fixed_amount)
        self.generate_report(simulation_result)
        return simulation_result

def main():
    """主函数"""
    data_root = '/Users/Williamhiler/Documents/my-project/train/test_data'
    model_dir = '/Users/Williamhiler/Documents/my-project/train/models'
    
    # 创建投注模拟器
    simulator = DirectBettingSimulator(
        data_root=data_root,
        model_dir=model_dir,
        model_version="3.0.4",
        data_season="2025-2026"
    )
    
    # 只运行固定金额模拟投注
    print("\n=== 固定金额投注模拟 ===")
    simulator.run_simulation(bet_amount=10, use_fixed_amount=True)

if __name__ == "__main__":
    main()