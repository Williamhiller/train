class KellyCalculator:
    @staticmethod
    def calculate_kelly(odds: float, predicted_prob: float) -> float:
        """ 
        计算凯利指数
        
        参数: 
            odds: 赔率 (必须 > 1)
            predicted_prob: 预测概率 (0-1之间)
        
        返回: 
            凯利指数 (小数形式，如0.15表示15%)
        """
        if odds <= 1 or predicted_prob < 0 or predicted_prob > 1:
            return 0.0  # 参数无效时返回0
        
        kelly = (odds * predicted_prob - 1) / (odds - 1)
        
        # 确保结果在合理范围内
        kelly = max(kelly, 0.0)  # 负值设为0
        kelly = min(kelly, 1.0)  # 最大值不超过1
        
        return kelly

    @staticmethod
    def calculate_kelly_for_football_match(odds_dict: dict, prob_dict: dict) -> dict:
        """ 
        计算足球比赛三种结果的凯利指数
        
        参数: 
            odds_dict: {'win': 2.1, 'draw': 3.2, 'lose': 3.5}
            prob_dict: {'win': 0.5, 'draw': 0.3, 'lose': 0.2}
        
        返回: 
            {'win': 0.0455, 'draw': 0.0, 'lose': 0.0}
        """
        kelly_results = {}
        
        for outcome in ['win', 'draw', 'lose']:
            if outcome not in odds_dict or outcome not in prob_dict:
                kelly_results[outcome] = 0.0
                continue
                
            odds = odds_dict[outcome]
            prob = prob_dict[outcome]
            kelly_results[outcome] = KellyCalculator.calculate_kelly(odds, prob)
        
        return kelly_results

# 使用示例
if __name__ == "__main__":
    # 赔率数据
    odds = {'win': 2.10, 'draw': 3.20, 'lose': 3.50}
    
    # 模型预测概率（总和应为1）
    predictions = {'win': 0.55, 'draw': 0.25, 'lose': 0.20}
    
    # 计算凯利指数
    kelly_values = KellyCalculator.calculate_kelly_for_football_match(odds, predictions)
    
    print("凯利指数计算结果：")
    for outcome, value in kelly_values.items():
        print(f"  {outcome}: {value:.2%} (赔率: {odds[outcome]})")
    
    # 找出最有价值投注
    best_outcome = max(kelly_values, key=kelly_values.get)
    if kelly_values[best_outcome] > 0:
        print(f"\n建议投注: {best_outcome}, 凯利指数: {kelly_values[best_outcome]:.2%}")
    else:
        print("\n没有正凯利值的投注选项")
