import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_simulation_result(result_path):
    """加载模拟投注结果"""
    with open(result_path, 'r', encoding='utf-8') as f:
        result = json.load(f)
    return result

def plot_profit_curve(bet_history, output_path):
    """绘制盈利曲线"""
    plt.figure(figsize=(12, 6))
    
    # 提取累计盈利数据
    bets = range(1, len(bet_history) + 1)
    total_profits = [bet['total_profit'] for bet in bet_history]
    
    plt.plot(bets, total_profits, marker='o', markersize=3, linewidth=1)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    plt.title('累计盈利曲线', fontsize=16)
    plt.xlabel('投注次数', fontsize=12)
    plt.ylabel('累计盈利 (元)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(output_path, 'profit_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_bet_results(bet_history, output_path):
    """绘制投注结果统计"""
    # 统计赢输次数
    wins = sum(1 for bet in bet_history if bet['is_winner'])
    losses = len(bet_history) - wins
    
    plt.figure(figsize=(8, 6))
    
    labels = ['赢', '输']
    sizes = [wins, losses]
    colors = ['#4CAF50', '#F44336']
    explode = (0.1, 0)
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    
    plt.title('投注结果统计', fontsize=16)
    plt.axis('equal')
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(output_path, 'bet_results.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_result_distribution(bet_history, output_path):
    """绘制预测结果和实际结果分布"""
    plt.figure(figsize=(12, 6))
    
    # 提取预测结果和实际结果
    actual_results = [bet['actual_result'] for bet in bet_history]
    predicted_results = [bet['predicted_result'] for bet in bet_history]
    
    # 创建数据框
    df = pd.DataFrame({
        '实际结果': actual_results,
        '预测结果': predicted_results
    })
    
    # 绘制堆叠柱状图
    cross_tab = pd.crosstab(df['实际结果'], df['预测结果'])
    cross_tab.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis')
    
    plt.title('预测结果与实际结果分布', fontsize=16)
    plt.xlabel('实际结果', fontsize=12)
    plt.ylabel('次数', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='预测结果')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(output_path, 'result_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_odds_analysis(bet_history, output_path):
    """绘制赔率分析"""
    plt.figure(figsize=(12, 6))
    
    # 提取赔率和盈利数据
    odds = [bet['bet_odds'] for bet in bet_history]
    profits = [bet['profit'] for bet in bet_history]
    
    # 绘制散点图
    plt.scatter(odds, profits, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    plt.title('赔率与盈利关系', fontsize=16)
    plt.xlabel('投注赔率', fontsize=12)
    plt.ylabel('盈利 (元)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(output_path, 'odds_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_betting_report(result_path, output_dir):
    """生成投注报告"""
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载模拟结果
    result = load_simulation_result(result_path)
    bet_history = result['bet_history']
    
    # 绘制各种图表
    plot_profit_curve(bet_history, output_dir)
    plot_bet_results(bet_history, output_dir)
    plot_result_distribution(bet_history, output_dir)
    plot_odds_analysis(bet_history, output_dir)
    
    # 生成文本报告
    report_path = os.path.join(output_dir, 'betting_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("模拟投注报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"总投注次数: {result['total_bets']}\n")
        f.write(f"每次投注金额: ¥{result['total_winnings'] / result['total_bets']:.2f}\n")
        f.write(f"总投注金额: ¥{result['total_winnings']:.2f}\n")
        f.write(f"总盈利: ¥{result['total_profit']:.2f}\n")
        f.write(f"盈利比例: {result['profit_rate'] * 100:.2f}%\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("投注历史统计\n")
        f.write("=" * 60 + "\n\n")
        
        # 统计赢输次数
        wins = sum(1 for bet in bet_history if bet['is_winner'])
        losses = len(bet_history) - wins
        f.write(f"赢: {wins} 次 ({wins/len(bet_history)*100:.1f}%)\n")
        f.write(f"输: {losses} 次 ({losses/len(bet_history)*100:.1f}%)\n\n")
        
        # 统计预测结果
        predicted_counts = {}
        for bet in bet_history:
            predicted = bet['predicted_result']
            if predicted not in predicted_counts:
                predicted_counts[predicted] = 0
            predicted_counts[predicted] += 1
        
        f.write("预测结果分布:\n")
        for result_type, count in predicted_counts.items():
            f.write(f"  {result_type}: {count} 次 ({count/len(bet_history)*100:.1f}%)\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("最近10次投注记录\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'比赛ID':<10} {'实际结果':<8} {'预测结果':<8} {'赔率':<6} {'盈利':<8} {'累计盈利':<10}\n")
        f.write("-" * 60 + "\n")
        
        recent_history = bet_history[-10:]
        for bet in recent_history:
            f.write(f"{bet['match_id']:<10} {bet['actual_result']:<8} {bet['predicted_result']:<8} "
                  f"{bet['bet_odds']:<6.2f} {bet['profit']:<8.2f} {bet['total_profit']:<10.2f}\n")
    
    print(f"投注报告已生成至: {report_path}")
    print(f"图表已保存至: {output_dir}")

def main():
    """主函数"""
    # 模拟结果文件路径
    result_path = '/Users/Williamhiler/Documents/my-project/train/output/betting_simulation_v3_xgboost.json'
    
    # 输出目录
    output_dir = '/Users/Williamhiler/Documents/my-project/train/output/betting_visualization'
    
    # 生成报告
    generate_betting_report(result_path, output_dir)

if __name__ == "__main__":
    main()
