import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_model_info(model_path):
    """加载模型信息"""
    with open(model_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_models():
    """对比三个版本模型的性能"""
    # 加载三个版本的模型信息
    model_v1 = load_model_info('/Users/Williamhiler/Documents/my-project/train/models/v1/model_v1_xgboost_info.json')
    model_v2 = load_model_info('/Users/Williamhiler/Documents/my-project/train/models/v2/model_v2_xgboost_info.json')
    model_v3 = load_model_info('/Users/Williamhiler/Documents/my-project/train/models/v3/model_v3.0.1_xgboost_info.json')
    
    models = {
        'V1 (仅赔率特征)': model_v1,
        'V2 (赔率+球队状态)': model_v2,
        'V3 (赔率+球队状态+专家，调优)': model_v3
    }
    
    # 创建对比表格数据
    data = {
        '模型版本': [],
        '整体准确率': [],
        '客胜精确率': [],
        '客胜召回率': [],
        '平局精确率': [],
        '平局召回率': [],
        '主胜精确率': [],
        '主胜召回率': []
    }
    
    for model_name, model_info in models.items():
        data['模型版本'].append(model_name)
        data['整体准确率'].append(model_info['accuracy'])
        data['客胜精确率'].append(model_info['classification_report']['客胜']['precision'])
        data['客胜召回率'].append(model_info['classification_report']['客胜']['recall'])
        data['平局精确率'].append(model_info['classification_report']['平局']['precision'])
        data['平局召回率'].append(model_info['classification_report']['平局']['recall'])
        data['主胜精确率'].append(model_info['classification_report']['主胜']['precision'])
        data['主胜召回率'].append(model_info['classification_report']['主胜']['recall'])
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 打印对比表格
    print("三个版本模型性能对比")
    print("=" * 80)
    print(df.to_string(index=False, float_format='%.4f'))
    
    # 生成可视化图表
    generate_visualizations(models, df)
    
    return df

def generate_visualizations(models, df):
    """生成可视化图表"""
    # 创建输出目录
    output_dir = '/Users/Williamhiler/Documents/my-project/train/output/model_comparison'
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. 整体准确率对比图
    plt.figure(figsize=(10, 6))
    model_names = df['模型版本'].tolist()
    accuracies = df['整体准确率'].tolist()
    
    bars = plt.bar(model_names, accuracies, color=['#E53935', '#43A047', '#1E88E5'])
    
    for bar, accuracy in zip(bars, accuracies):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{accuracy:.4f}', ha='center', va='bottom')
    
    plt.title('三个版本模型整体准确率对比', fontsize=16)
    plt.ylabel('准确率', fontsize=12)
    plt.ylim(0, 0.7)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 胜平负精确率对比图
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(model_names))
    width = 0.25
    
    home_away_accuracy = df['客胜精确率'].tolist()
    draw_accuracy = df['平局精确率'].tolist()
    home_win_accuracy = df['主胜精确率'].tolist()
    
    plt.bar(x - width, home_away_accuracy, width, label='客胜精确率', color='#E53935')
    plt.bar(x, draw_accuracy, width, label='平局精确率', color='#43A047')
    plt.bar(x + width, home_win_accuracy, width, label='主胜精确率', color='#1E88E5')
    
    plt.xlabel('模型版本', fontsize=12)
    plt.ylabel('精确率', fontsize=12)
    plt.title('三个版本模型胜平负精确率对比', fontsize=16)
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 胜平负召回率对比图
    plt.figure(figsize=(12, 6))
    
    home_away_recall = df['客胜召回率'].tolist()
    draw_recall = df['平局召回率'].tolist()
    home_win_recall = df['主胜召回率'].tolist()
    
    plt.bar(x - width, home_away_recall, width, label='客胜召回率', color='#E53935')
    plt.bar(x, draw_recall, width, label='平局召回率', color='#43A047')
    plt.bar(x + width, home_win_recall, width, label='主胜召回率', color='#1E88E5')
    
    plt.xlabel('模型版本', fontsize=12)
    plt.ylabel('召回率', fontsize=12)
    plt.title('三个版本模型胜平负召回率对比', fontsize=16)
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recall_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n对比图表已保存至: {output_dir}")

def print_detailed_analysis():
    """打印详细分析"""
    print("\n" + "=" * 80)
    print("模型性能详细分析")
    print("=" * 80)
    print("1. 整体准确率提升趋势:")
    print("   - V1 → V2: 提升了1.35个百分点 (0.5405 → 0.5541)")
    print("   - V2 → V3: 提升了8.11个百分点 (0.5541 → 0.6351)")
    print("   - V1 → V3: 提升了9.46个百分点 (0.5405 → 0.6351)")
    print()
    print("2. 各结果预测性能:")
    print("   - 客胜: V3的精确率最高(0.7000)，召回率与V1相同(0.6667)")
    print("   - 平局: V3的精确率最高(0.5000)，但召回率仍较低(0.2778)")
    print("   - 主胜: V3的精确率(0.6364)和召回率(0.8000)均为最高")
    print()
    print("3. 模型改进关键因素:")
    print("   - 添加球队状态特征(V2)带来的提升有限")
    print("   - 添加专家特征并进行超参数调优(V3)带来了显著提升")
    print("   - 主胜预测性能提升最为明显，从V1的65.71%召回率提升到V3的80.00%")
    print()
    print("4. 仍需改进的地方:")
    print("   - 平局预测的召回率仍然较低，三个版本均不超过34%")
    print("   - 可以考虑调整模型结构或特征选择来提高平局预测准确性")
    print("=" * 80)

if __name__ == "__main__":
    compare_models()
    print_detailed_analysis()
