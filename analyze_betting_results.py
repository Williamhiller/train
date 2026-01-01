import json
import os

# 投注结果文件路径
result_path = '/Users/Williamhiler/Documents/my-project/train/test_data/2025-2026/betting_result_3.0.7.json'

# 加载投注结果
with open(result_path, 'r', encoding='utf-8') as f:
    result = json.load(f)

# 初始化统计变量
home_win_stats = {
    'count': 0,
    'correct': 0,
    'total_bet': 0,
    'total_profit': 0
}

draw_stats = {
    'count': 0,
    'correct': 0,
    'total_bet': 0,
    'total_profit': 0
}

away_win_stats = {
    'count': 0,
    'correct': 0,
    'total_bet': 0,
    'total_profit': 0
}

# 结果映射
result_mapping = {
    '主胜': 'home_win',
    '平局': 'draw',
    '客胜': 'away_win'
}

# 遍历所有投注记录
for bet in result['bet_history']:
    predicted_result = bet['predicted_result']
    actual_result = bet['actual_result']
    bet_amount = bet['bet_amount']
    profit = bet['profit']
    
    # 根据预测结果更新统计
    if predicted_result == '主胜':
        stats = home_win_stats
    elif predicted_result == '平局':
        stats = draw_stats
    else:  # 客胜
        stats = away_win_stats
    
    # 更新统计
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
print("=== 胜平负盈利率统计 ===")
print()

# 主胜预测统计
print(f"主胜预测:")
print(f"  预测场次: {home_win_stats['count']}")
print(f"  正确场次: {home_win_stats['correct']}")
if home_win_stats['count'] > 0:
    print(f"  正确率: {home_win_stats['correct'] / home_win_stats['count'] * 100:.2f}%")
else:
    print(f"  正确率: 0.00%")
print(f"  总投注: ¥{home_win_stats['total_bet']}")
print(f"  总盈利: ¥{home_win_stats['total_profit']:.2f}")
print(f"  盈利率: {calculate_roi(home_win_stats['total_bet'], home_win_stats['total_profit']):.2f}%")
print()

# 平局预测统计
print(f"平局预测:")
print(f"  预测场次: {draw_stats['count']}")
print(f"  正确场次: {draw_stats['correct']}")
if draw_stats['count'] > 0:
    print(f"  正确率: {draw_stats['correct'] / draw_stats['count'] * 100:.2f}%")
else:
    print(f"  正确率: 0.00%")
print(f"  总投注: ¥{draw_stats['total_bet']}")
print(f"  总盈利: ¥{draw_stats['total_profit']:.2f}")
print(f"  盈利率: {calculate_roi(draw_stats['total_bet'], draw_stats['total_profit']):.2f}%")
print()

# 客胜预测统计
print(f"客胜预测:")
print(f"  预测场次: {away_win_stats['count']}")
print(f"  正确场次: {away_win_stats['correct']}")
if away_win_stats['count'] > 0:
    print(f"  正确率: {away_win_stats['correct'] / away_win_stats['count'] * 100:.2f}%")
else:
    print(f"  正确率: 0.00%")
print(f"  总投注: ¥{away_win_stats['total_bet']}")
print(f"  总盈利: ¥{away_win_stats['total_profit']:.2f}")
print(f"  盈利率: {calculate_roi(away_win_stats['total_bet'], away_win_stats['total_profit']):.2f}%")
print()

# 整体统计
print(f"整体统计:")
print(f"  总场次: {result['total_bets']}")
print(f"  总正确率: {result['accuracy'] * 100:.2f}%")
print(f"  总投注: ¥{result['total_winnings']}")
print(f"  总盈利: ¥{result['total_profit']:.2f}")
print(f"  总盈利率: {result['roi']:.2f}%")
