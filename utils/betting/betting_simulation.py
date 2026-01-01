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

class BettingSimulator:
    def __init__(self, data_root, model_dir, model_version=3, model_name='lightgbm', custom_version=None):
        self.data_root = data_root
        self.model_dir = model_dir
        self.model_version = model_version
        self.model_name = model_name
        self.custom_version = custom_version
        self.data_loader = DataLoader(data_root)
        self.scaler = StandardScaler()
        
        # 加载模型和模型信息
        self.model, self.model_info = self.load_model()
        
        # 结果映射：[0, 1, 2] -> ['客胜', '平局', '主胜']
        self.result_mapping = {0: '客胜', 1: '平局', 2: '主胜'}
    
    def load_model(self):
        """加载已训练的模型"""
        # 根据版本号构造不同的文件名和目录
        if self.model_version == 3:
            version_dir = "v3"
            # 使用自定义版本号，如果没有则使用默认
            if self.custom_version:
                version_str = f"v{self.custom_version}"
                # 检查是否存在对应的版本目录
                version_subdir = os.path.join(self.model_dir, version_dir, self.custom_version)
                if not os.path.exists(version_subdir):
                    raise FileNotFoundError(f"版本目录不存在: {version_subdir}")
                model_path = os.path.join(version_subdir, f"model_{version_str}_{self.model_name}.joblib")
                info_path = os.path.join(version_subdir, f"model_{version_str}_{self.model_name}_info.json")
            else:
                version_str = "v3.0.1"
                model_filename = f"model_{version_str}_{self.model_name}.joblib"
                model_path = os.path.join(self.model_dir, version_dir, model_filename)
                info_filename = f"model_{version_str}_{self.model_name}_info.json"
                info_path = os.path.join(self.model_dir, version_dir, info_filename)
        else:
            version_dir = version_str = f"v{self.model_version}"
            model_filename = f"model_{version_str}_{self.model_name}.joblib"
            model_path = os.path.join(self.model_dir, version_dir, model_filename)
            info_filename = f"model_{version_str}_{self.model_name}_info.json"
            info_path = os.path.join(self.model_dir, version_dir, info_filename)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载模型
        model = joblib.load(model_path)
        
        # 加载模型配置信息
        with open(info_path, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        
        return model, model_info
    
    def prepare_test_data(self, seasons=['2024-2025'], include_team_state=True, include_expert=True):
        """准备测试数据"""
        # 加载组合特征
        all_data = []
        for season in seasons:
            try:
                season_data = self.data_loader.load_combined_features(season, include_team_state, include_expert)
                all_data.append(season_data)
            except FileNotFoundError as e:
                print(f"跳过赛季 {season}: {e}")
        
        if not all_data:
            raise ValueError("没有找到任何测试数据")
        
        # 合并所有赛季的数据
        df = pd.concat(all_data, ignore_index=True)
        
        # 处理缺失值
        df = df.dropna()
        
        return df
    
    def simulate_betting(self, test_data, bet_amount=10, max_bets=100, use_fixed_amount=False):
        """模拟投注过程，包含赔率过滤和投注金额选择
        
        Args:
            test_data: 测试数据
            bet_amount: 投注金额（固定金额或基准金额）
            max_bets: 最大投注次数
            use_fixed_amount: 是否使用固定金额投注，False则使用基于凯利指数的动态金额
        """
        # 特征和标签分离
        features = test_data.drop(['match_id', 'season', 'result_code', 'home_score', 'away_score'], axis=1, errors='ignore')
        labels = test_data['result_code']
        
        # 转换分类特征（与训练时保持一致）
        categorical_cols = features.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            features = pd.get_dummies(features, columns=categorical_cols)
        
        # 确保特征顺序与训练时一致
        feature_names = self.model_info['feature_names']
        
        # 添加缺失的特征列（如果有）
        for feature in feature_names:
            if feature not in features.columns:
                features[feature] = 0
        
        # 只保留训练时使用的特征
        features = features[feature_names]
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(features)
        
        # 模型预测
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)
        
        # 模拟投注
        total_bets = 0
        total_winnings = 0
        total_profit = 0
        bet_history = []
        
        # 过滤出符合条件的比赛：主胜或客胜赔率在1.8-4.0之间
        valid_indices = []
        for i in range(len(test_data)):
            match = test_data.iloc[i]
            win_odds = match['closing_win_odds']  # 主胜赔率
            lose_odds = match['closing_lose_odds']  # 客胜赔率
            
            if (1.8 <= win_odds <= 4.0) or (1.8 <= lose_odds <= 4.0):
                valid_indices.append(i)
        
        # 限制最大投注次数
        valid_indices = valid_indices[:max_bets]
        
        for i in valid_indices:
            match = test_data.iloc[i]
            actual_result = match['result_code']
            predicted_result = y_pred[i]
            predicted_proba = y_pred_proba[i]
            
            # 获取赔率信息（使用终盘赔率）
            win_odds = match['closing_win_odds']  # 主胜赔率
            draw_odds = match['closing_draw_odds']  # 平局赔率
            lose_odds = match['closing_lose_odds']  # 客胜赔率
            
            odds = [lose_odds, draw_odds, win_odds]  # 对应客胜、平局、主胜
            
            # 选择预测概率最高的结果进行投注
            bet_result = np.argmax(predicted_proba)
            bet_odds = odds[bet_result]
            predicted_prob = predicted_proba[bet_result]
            
            # 计算凯利指数
            if bet_odds > 0:
                kelly_ratio = max(0, (predicted_prob * bet_odds - 1) / (bet_odds - 1))
            else:
                kelly_ratio = 0
            
            # 决定投注金额
            if use_fixed_amount:
                # 使用固定金额
                current_bet_amount = bet_amount
            else:
                # 动态调整投注金额（基准10元，范围5-20元）
                base_bet_amount = bet_amount
                max_kelly_ratio = 0.5  # 设定最大凯利比例
                
                adjustment_factor = kelly_ratio / max_kelly_ratio if max_kelly_ratio > 0 else 0
                dynamic_bet_amount = base_bet_amount + (adjustment_factor * (20 - base_bet_amount))
                dynamic_bet_amount = max(dynamic_bet_amount, 5.0)
                dynamic_bet_amount = min(dynamic_bet_amount, 20.0)
                current_bet_amount = round(dynamic_bet_amount, 2)
            
            # 计算投注结果
            total_bets += 1
            total_winnings += current_bet_amount
            
            if bet_result == actual_result:
                # 赢了，计算盈利
                profit = current_bet_amount * (bet_odds - 1)
                total_profit += profit
                is_winner = True
            else:
                # 输了，损失本金
                profit = -current_bet_amount
                total_profit += profit
                is_winner = False
            
            # 记录投注历史
            bet_history.append({
                'match_id': match['match_id'],
                'season': match['season'],
                'actual_result': self.result_mapping[actual_result],
                'predicted_result': self.result_mapping[bet_result],
                'predicted_probability': float(predicted_prob),
                'bet_odds': float(bet_odds),
                'kelly_ratio': float(kelly_ratio),
                'bet_amount': float(current_bet_amount),
                'profit': float(profit),
                'total_profit': float(total_profit),
                'is_winner': is_winner
            })
        
        return {
            'total_bets': total_bets,
            'total_winnings': total_winnings,
            'total_profit': total_profit,
            'profit_rate': total_profit / total_winnings if total_winnings > 0 else 0,
            'bet_history': bet_history
        }
    
    def run_simulation(self, seasons=['2024-2025'], bet_amount=10, max_bets=100, use_fixed_amount=False):
        """运行完整的投注模拟
        
        Args:
            seasons: 要测试的赛季
            bet_amount: 投注金额（固定金额或基准金额）
            max_bets: 最大投注次数
            use_fixed_amount: 是否使用固定金额投注，False则使用基于凯利指数的动态金额
        """
        # 准备测试数据
        test_data = self.prepare_test_data(seasons)
        
        # 模拟投注
        simulation_result = self.simulate_betting(test_data, bet_amount, max_bets, use_fixed_amount)
        
        # 输出结果
        self.print_simulation_result(simulation_result, use_fixed_amount)
        
        return simulation_result
    
    def print_simulation_result(self, simulation_result, use_fixed_amount=False):
        """打印模拟结果"""
        print("=" * 60)
        print("模拟投注结果")
        print("=" * 60)
        version_info = f"V{self.model_version}"
        if self.custom_version:
            version_info = f"V{self.custom_version}"
        print(f"模型版本: {version_info} ({self.model_name})")
        print(f"投注策略: {'固定金额投注' if use_fixed_amount else '动态金额投注（基于凯利指数）'}")
        print(f"投注次数: {simulation_result['total_bets']}")
        print(f"每次平均投注金额: ¥{simulation_result['total_winnings'] / simulation_result['total_bets']:.2f}")
        print(f"总投注金额: ¥{simulation_result['total_winnings']:.2f}")
        print(f"总盈利: ¥{simulation_result['total_profit']:.2f}")
        print(f"盈利比例: {simulation_result['profit_rate'] * 100:.2f}%")
        print("=" * 60)
        
        # 打印部分投注历史
        print("\n最近10次投注历史:")
        print("-" * 60)
        print(f"{'比赛ID':<10} {'实际结果':<8} {'预测结果':<8} {'赔率':<6} {'盈利':<8} {'累计盈利':<10}")
        print("-" * 60)
        
        recent_history = simulation_result['bet_history'][-10:]
        for bet in recent_history:
            print(f"{bet['match_id']:<10} {bet['actual_result']:<8} {bet['predicted_result']:<8} "
                  f"{bet['bet_odds']:<6.2f} {bet['profit']:<8.2f} {bet['total_profit']:<10.2f}")
        
        print("=" * 60)

def main():
    """主函数"""
    data_root = '/Users/Williamhiler/Documents/my-project/train/train-data'
    model_dir = '/Users/Williamhiler/Documents/my-project/train/models'
    
    # 使用3.0.4版本模型进行模拟投注
    simulator = BettingSimulator(data_root, model_dir, model_version=3, model_name='lightgbm', custom_version='3.0.4')
    
    # 运行固定金额模拟投注
    print("\n=== 固定金额投注模拟 ===")
    fixed_simulation = simulator.run_simulation(
        seasons=['2024-2025'],
        bet_amount=10,
        max_bets=100,
        use_fixed_amount=True
    )
    
    # 运行动态金额模拟投注
    print("\n=== 动态金额投注模拟（基于凯利指数） ===")
    dynamic_simulation = simulator.run_simulation(
        seasons=['2024-2025'],
        bet_amount=10,
        max_bets=100,
        use_fixed_amount=False
    )
    
    # 保存模拟结果
    result_filename = f"betting_simulation_v3.0.4_lightgbm_fixed.json"
    result_path = os.path.join('/Users/Williamhiler/Documents/my-project/train/output', result_filename)
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(fixed_simulation, f, ensure_ascii=False, indent=2)
    
    result_filename = f"betting_simulation_v3.0.4_lightgbm_dynamic.json"
    result_path = os.path.join('/Users/Williamhiler/Documents/my-project/train/output', result_filename)
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(dynamic_simulation, f, ensure_ascii=False, indent=2)
    
    print(f"模拟结果已保存至: {result_path}")

if __name__ == "__main__":
    main()
