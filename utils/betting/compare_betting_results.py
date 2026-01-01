import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_betting_results(result_dir):
    """加载所有投注结果文件"""
    results = []
    
    for filename in os.listdir(result_dir):
        if filename.endswith('.json') and 'betting_result' in filename:
            file_path = os.path.join(result_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            # 提取模型版本和投注策略
            filename_parts = filename.replace('betting_result_', '').replace('.json', '')
            
            # 处理direct版本的文件名
            if 'direct' in filename_parts:
                if filename_parts.startswith('direct_'):
                    # 格式：direct_v3.0.4
                    model_version = filename_parts.replace('direct_', '')
                    strategy = 'direct'
                elif '_direct' in filename_parts:
                    # 格式：v3.0.4_direct
                    model_version = filename_parts.replace('_direct', '')
                    strategy = 'direct'
                else:
                    continue
            else:
                parts = filename_parts.split('_')
                if len(parts) == 1:
                    model_version = parts[0]
                    strategy = 'fixed100'
                else:
                    model_version = parts[0]
                    strategy = parts[1]
            
            # 提取关键指标
            total_bets = result['total_bets']
            accuracy = result['accuracy'] * 100
            total_winnings = result['total_winnings']
            total_profit = result['total_profit']
            roi = result['roi']
            
            results.append({
                'model_version': model_version,
                'strategy': strategy,
                'total_bets': total_bets,
                'accuracy': accuracy,
                'total_winnings': total_winnings,
                'total_profit': total_profit,
                'roi': roi
            })
    
    return results

def generate_comparison_table(results):
    """生成对比表格"""
    df = pd.DataFrame(results)
    
    # 处理重复条目：对于相同的模型版本和策略，只保留投注金额最大的记录
    df = df.sort_values(['model_version', 'strategy', 'total_winnings'], ascending=[True, True, False])
    df = df.drop_duplicates(subset=['model_version', 'strategy'], keep='first')
    
    # 按照模型版本和策略排序
    df.sort_values(['model_version', 'strategy'], inplace=True)
    
    return df

def visualize_comparison(df):
    """可视化对比结果"""
    # 设置图形大小
    plt.figure(figsize=(12, 8))
    
    # 创建子图
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    
    # 1. 总投注场次对比
    pivot_total_bets = df.pivot(index='model_version', columns='strategy', values='total_bets')
    pivot_total_bets.plot(kind='bar', ax=ax1, rot=45)
    ax1.set_title('总投注场次对比')
    ax1.set_ylabel('总投注场次')
    ax1.grid(True, alpha=0.3)
    
    # 2. 准确率对比
    pivot_accuracy = df.pivot(index='model_version', columns='strategy', values='accuracy')
    pivot_accuracy.plot(kind='bar', ax=ax2, rot=45)
    ax2.set_title('准确率对比')
    ax2.set_ylabel('准确率 (%)')
    ax2.grid(True, alpha=0.3)
    
    # 3. 总盈利对比
    pivot_profit = df.pivot(index='model_version', columns='strategy', values='total_profit')
    pivot_profit.plot(kind='bar', ax=ax3, rot=45)
    ax3.set_title('总盈利对比')
    ax3.set_ylabel('总盈利 (元)')
    ax3.grid(True, alpha=0.3)
    
    # 4. 收益率对比
    pivot_roi = df.pivot(index='model_version', columns='strategy', values='roi')
    pivot_roi.plot(kind='bar', ax=ax4, rot=45)
    ax4.set_title('收益率对比')
    ax4.set_ylabel('收益率 (%)')
    ax4.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    plt.savefig('/Users/Williamhiler/Documents/my-project/train/test_data/2025-2026/betting_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 结果目录
    result_dir = '/Users/Williamhiler/Documents/my-project/train/test_data/2025-2026'
    
    # 加载结果
    results = load_betting_results(result_dir)
    
    # 生成对比表格
    df = generate_comparison_table(results)
    
    # 筛选出我们感兴趣的模型版本：2.0.4, 3.0.7
    interested_versions = ['2.0.4', '3.0.7']
    df_filtered = df[df['model_version'].isin(interested_versions)]
    
    # 打印对比表格
    print("=== 投注模拟结果对比 ===")
    print("\n所有模型版本对比:")
    print(df.to_string(index=False))
    
    print("\n\n=== 感兴趣模型版本对比 (2.0.4, 3.0.7) ===")
    print(df_filtered.to_string(index=False))
    
    # 保存对比表格到文件
    df.to_csv(os.path.join(result_dir, 'betting_comparison.csv'), index=False, encoding='utf-8-sig')
    df_filtered.to_csv(os.path.join(result_dir, 'betting_comparison_filtered.csv'), index=False, encoding='utf-8-sig')
    
    # 可视化对比结果
    visualize_comparison(df_filtered)
    
    print("\n\n=== 对比结果已保存 ===")
    print(f"CSV文件: {os.path.join(result_dir, 'betting_comparison.csv')}")
    print(f"筛选后的CSV文件: {os.path.join(result_dir, 'betting_comparison_filtered.csv')}")
    print(f"可视化图表: {os.path.join(result_dir, 'betting_comparison.png')}")

if __name__ == "__main__":
    main()