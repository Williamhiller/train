import json
import os

# 投注结果文件路径
result_path = '/Users/Williamhiler/Documents/my-project/train/test_data/2025-2026/betting_result_3.0.7.json'

# 加载投注结果
with open(result_path, 'r', encoding='utf-8') as f:
    result = json.load(f)

# 定义赔率区间
def get_odds_range(odds):
    if 1.8 <= odds < 2.0:
        return '1.8-2.0'
    elif 2.0 <= odds < 2.5:
        return '2.0-2.5'
    elif 2.5 <= odds < 3.0:
        return '2.5-3.0'
    elif 3.0 <= odds <= 4.0:
        return '3.0-4.0'
    else:
        return 'other'

# 初始化赔率区间统计
def initialize_stats():
    return {
        'count': 0,
        'correct': 0,
        'total_bet': 0,
        'total_profit': 0
    }

# 初始化统计变量，只保留1.8-4.0区间
odds_range_stats = {
    '1.8-2.0': initialize_stats(),
    '2.0-2.5': initialize_stats(),
    '2.5-3.0': initialize_stats(),
    '3.0-4.0': initialize_stats()
}

# 遍历所有投注记录
for bet in result['bet_history']:
    predicted_result = bet['predicted_result']
    actual_result = bet['actual_result']
    bet_amount = bet['bet_amount']
    profit = bet['profit']
    bet_odds = bet['bet_odds']
    
    # 获取赔率区间
    odds_range = get_odds_range(bet_odds)
    
    # 只处理1.8-4.0区间的赔率
    if odds_range == 'other':
        continue
    
    # 更新区间统计
    stats = odds_range_stats[odds_range]
    stats['count'] += 1
    stats['total_bet'] += bet_amount
    stats['total_profit'] += profit
    
    if predicted_result == actual_result:
        stats['correct'] += 1

# 计算盈利率
def calculate_roi(total_bet, total_profit):
    if total_bet == 0:
        return 0
    return (total_profit / total_bet) * 100

# 输出统计结果
print("=== 赔率区间分析 ===")
print()

# 输出每个赔率区间的统计
best_roi = 0
best_range = ''

for odds_range, stats in odds_range_stats.items():
    print(f"{odds_range}赔率区间:")
    print(f"  预测场次: {stats['count']}")
    print(f"  正确场次: {stats['correct']}")
    if stats['count'] > 0:
        accuracy = stats['correct'] / stats['count'] * 100
        print(f"  正确率: {accuracy:.2f}%")
    else:
        print(f"  正确率: 0.00%")
    print(f"  总投注: ¥{stats['total_bet']}")
    print(f"  总盈利: ¥{stats['total_profit']:.2f}")
    roi = calculate_roi(stats['total_bet'], stats['total_profit'])
    print(f"  盈利率: {roi:.2f}%")
    print()
    
    # 记录最佳赔率区间
    if roi > best_roi and stats['count'] > 5:  # 至少5场比赛才有统计意义
        best_roi = roi
        best_range = odds_range

# 输出最佳赔率区间
print(f"=== 最佳赔率区间 ===")
print(f"最佳赔率区间: {best_range}")
print(f"最佳盈利率: {best_roi:.2f}%")
print()

# 输出整体统计
print(f"=== 整体统计 ===")
print(f"总场次: {result['total_bets']}")
print(f"总正确率: {result['accuracy'] * 100:.2f}%")
print(f"总投注: ¥{result['total_winnings']}")
print(f"总盈利: ¥{result['total_profit']:.2f}")
print(f"总盈利率: {result['roi']:.2f}%")
