import json
import collections

# 读取模拟投注结果
with open('/Users/Williamhiler/Documents/my-project/train/output/betting_simulation_v3_xgboost.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

bets = data['bet_history']

# 基本统计
print('总投注次数:', len(bets))
print('赢次数:', sum(1 for b in bets if b['is_winner']))
print('输次数:', sum(1 for b in bets if not b['is_winner']))

# 预测结果分布
print('\n预测结果分布:')
predicted = collections.Counter(b['predicted_result'] for b in bets)
for result, count in predicted.items():
    print(f'{result}: {count}次 ({count/len(bets)*100:.1f}%)')

# 实际结果分布
print('\n实际结果分布:')
actual = collections.Counter(b['actual_result'] for b in bets)
for result, count in actual.items():
    print(f'{result}: {count}次 ({count/len(bets)*100:.1f}%)')

# 各预测结果的赢率
print('\n各预测结果的赢率:')
for result in ['客胜', '平局', '主胜']:
    bets_result = [b for b in bets if b['predicted_result'] == result]
    if bets_result:
        win_rate = sum(1 for b in bets_result if b['is_winner']) / len(bets_result)
        print(f'{result}: {win_rate:.2f} ({len(bets_result)}次)')
    else:
        print(f'{result}: 无')

# 模型在各结果上的预测准确率
print('\n模型预测准确率(按实际结果):')
for result in ['客胜', '平局', '主胜']:
    result_bets = [b for b in bets if b['actual_result'] == result]
    if result_bets:
        correct = sum(1 for b in result_bets if b['predicted_result'] == result)
        accuracy = correct / len(result_bets)
        print(f'{result}: {accuracy:.2f} ({len(result_bets)}次)')
    else:
        print(f'{result}: 无')

# 赔率分析
print('\n赔率分析:')
all_odds = [b['bet_odds'] for b in bets]
print(f'平均赔率: {sum(all_odds)/len(all_odds):.2f}')
print(f'最高赔率: {max(all_odds):.2f}')
print(f'最低赔率: {min(all_odds):.2f}')

# 赔率与赢率的关系
print('\n赔率区间与赢率:')
# 按赔率分组
odds_groups = {'低赔率(<2.0)': [], '中赔率(2.0-3.0)': [], '高赔率(>3.0)': []}

for bet in bets:
    odds = bet['bet_odds']
    if odds < 2.0:
        odds_groups['低赔率(<2.0)'].append(bet)
    elif odds < 3.0:
        odds_groups['中赔率(2.0-3.0)'].append(bet)
    else:
        odds_groups['高赔率(>3.0)'].append(bet)

for group, group_bets in odds_groups.items():
    if group_bets:
        win_rate = sum(1 for b in group_bets if b['is_winner']) / len(group_bets)
        print(f'{group}: 赢率 {win_rate:.2f} ({len(group_bets)}次)')
    else:
        print(f'{group}: 无')
