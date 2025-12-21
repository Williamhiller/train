import json
import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_model_info(model_path):
    """加载模型信息"""
    with open(model_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_v3_versions():
    """对比V3.0.1和V3.0.2版本的模型性能"""
    # 加载两个版本的模型信息
    model_v3_0_1 = load_model_info('/Users/Williamhiler/Documents/my-project/train/models/v3/model_v3.0.1_xgboost_info.json')
    model_v3_0_2 = load_model_info('/Users/Williamhiler/Documents/my-project/train/models/v3/model_v3.0.2_xgboost_info.json')
    
    models = {
        'V3.0.1 (单赛季+调优)': model_v3_0_1,
        'V3.0.2 (全赛季+未调优)': model_v3_0_2
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
    print("V3.0.1和V3.0.2版本模型性能对比")
    print("=" * 80)
    print(df.to_string(index=False, float_format='%.4f'))
    
    # 分析变化情况
    print("\n" + "=" * 80)
    print("准确率变化分析")
    print("=" * 80)
    
    accuracy_change = model_v3_0_2['accuracy'] - model_v3_0_1['accuracy']
    print(f"整体准确率变化: {accuracy_change:.4f} ({accuracy_change*100:.2f}个百分点)")
    
    # 分析各结果预测性能变化
    for result in ['客胜', '平局', '主胜']:
        precision_change = model_v3_0_2['classification_report'][result]['precision'] - model_v3_0_1['classification_report'][result]['precision']
        recall_change = model_v3_0_2['classification_report'][result]['recall'] - model_v3_0_1['classification_report'][result]['recall']
        print(f"{result}精确率变化: {precision_change:.4f}, 召回率变化: {recall_change:.4f}")
    
    # 查看数据规模差异
    print("\n" + "=" * 80)
    print("数据规模对比")
    print("=" * 80)
    print(f"V3.0.1使用的赛季: {model_v3_0_1['seasons']}")
    v3_0_1_support = [model_v3_0_1['classification_report']['客胜']['support'],
                      model_v3_0_1['classification_report']['平局']['support'],
                      model_v3_0_1['classification_report']['主胜']['support']]
    print(f"V3.0.1测试集规模: {sum(v3_0_1_support)}")
    print(f"\nV3.0.2使用的赛季: {model_v3_0_2['seasons']}")
    v3_0_2_support = [model_v3_0_2['classification_report']['客胜']['support'],
                      model_v3_0_2['classification_report']['平局']['support'],
                      model_v3_0_2['classification_report']['主胜']['support']]
    print(f"V3.0.2测试集规模: {sum(v3_0_2_support)}")
    
    # 查看特征数量
    print("\n" + "=" * 80)
    print("特征数量对比")
    print("=" * 80)
    print(f"V3.0.1特征数量: {len(model_v3_0_1['feature_names'])}")
    print(f"V3.0.2特征数量: {len(model_v3_0_2['feature_names'])}")
    
    # 分析可能的原因
    print("\n" + "=" * 80)
    print("性能下降可能原因分析")
    print("=" * 80)
    print("1. 训练数据规模变化:")
    print(f"   - V3.0.1使用了单个赛季的数据")
    print(f"   - V3.0.2使用了所有10个赛季的数据")
    print("   - 数据规模的大幅增加可能导致模型过拟合或需要重新调整超参数")
    
    print("\n2. 超参数调优状态:")
    print(f"   - V3.0.1经过了超参数调优(tune_hyperparams=True)")
    print(f"   - V3.0.2未进行超参数调优(tune_hyperparams=False)")
    print("   - 超参数调优对模型性能有显著影响")
    
    print("\n3. 数据质量和一致性:")
    print("   - 多个赛季的数据可能存在格式不一致或质量差异")
    print("   - 旧赛季的数据可能与当前赛季的比赛模式存在差异")
    
    print("\n4. 特征重要性变化:")
    print("   - 不同赛季中，特征的重要性可能发生变化")
    print("   - 专家特征在不同赛季中的预测能力可能存在差异")
    
    # 生成可视化图表
    generate_visualizations(models, df)
    
    return df

def generate_visualizations(models, df):
    """生成可视化图表"""
    # 创建输出目录
    output_dir = '/Users/Williamhiler/Documents/my-project/train/output/model_comparison'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. 整体准确率对比图
    plt.figure(figsize=(10, 6))
    model_names = df['模型版本'].tolist()
    accuracies = df['整体准确率'].tolist()
    
    bars = plt.bar(model_names, accuracies, color=['#1E88E5', '#FB8C00'])
    
    for bar, accuracy in zip(bars, accuracies):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{accuracy:.4f}', ha='center', va='bottom')
    
    plt.title('V3.0.1和V3.0.2版本整体准确率对比', fontsize=16)
    plt.ylabel('准确率', fontsize=12)
    plt.ylim(0, 0.7)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'v3_versions_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 胜平负精确率对比图
    plt.figure(figsize=(12, 6))
    
    x = range(len(model_names))
    width = 0.25
    
    home_away_accuracy = df['客胜精确率'].tolist()
    draw_accuracy = df['平局精确率'].tolist()
    home_win_accuracy = df['主胜精确率'].tolist()
    
    plt.bar(x[0] - width, home_away_accuracy[0], width, label='客胜精确率', color='#E53935')
    plt.bar(x[0], draw_accuracy[0], width, label='平局精确率', color='#43A047')
    plt.bar(x[0] + width, home_win_accuracy[0], width, label='主胜精确率', color='#1E88E5')
    
    plt.bar(x[1] - width, home_away_accuracy[1], width, color='#E53935')
    plt.bar(x[1], draw_accuracy[1], width, color='#43A047')
    plt.bar(x[1] + width, home_win_accuracy[1], width, color='#1E88E5')
    
    plt.xlabel('模型版本', fontsize=12)
    plt.ylabel('精确率', fontsize=12)
    plt.title('V3.0.1和V3.0.2版本胜平负精确率对比', fontsize=16)
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'v3_versions_precision_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 胜平负召回率对比图
    plt.figure(figsize=(12, 6))
    
    home_away_recall = df['客胜召回率'].tolist()
    draw_recall = df['平局召回率'].tolist()
    home_win_recall = df['主胜召回率'].tolist()
    
    plt.bar(x[0] - width, home_away_recall[0], width, label='客胜召回率', color='#E53935')
    plt.bar(x[0], draw_recall[0], width, label='平局召回率', color='#43A047')
    plt.bar(x[0] + width, home_win_recall[0], width, label='主胜召回率', color='#1E88E5')
    
    plt.bar(x[1] - width, home_away_recall[1], width, color='#E53935')
    plt.bar(x[1], draw_recall[1], width, color='#43A047')
    plt.bar(x[1] + width, home_win_recall[1], width, color='#1E88E5')
    
    plt.xlabel('模型版本', fontsize=12)
    plt.ylabel('召回率', fontsize=12)
    plt.title('V3.0.1和V3.0.2版本胜平负召回率对比', fontsize=16)
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'v3_versions_recall_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n对比图表已保存至: {output_dir}")

if __name__ == "__main__":
    compare_v3_versions()