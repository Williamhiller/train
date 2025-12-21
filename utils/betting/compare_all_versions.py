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

def compare_all_versions():
    """对比所有版本模型的性能"""
    # 加载所有版本的模型信息
    model_v1_old = load_model_info('/Users/Williamhiler/Documents/my-project/train/models/v1/model_v1_xgboost_info.json')
    model_v1_new = load_model_info('/Users/Williamhiler/Documents/my-project/train/models/v1/model_v1.0.2_xgboost_info.json')
    
    model_v2_old = load_model_info('/Users/Williamhiler/Documents/my-project/train/models/v2/model_v2_xgboost_info.json')
    model_v2_new = load_model_info('/Users/Williamhiler/Documents/my-project/train/models/v2/model_v2.0.2_xgboost_info.json')
    
    model_v3_old = load_model_info('/Users/Williamhiler/Documents/my-project/train/models/v3/model_v3.0.1_xgboost_info.json')
    model_v3_new = load_model_info('/Users/Williamhiler/Documents/my-project/train/models/v3/model_v3.0.2_xgboost_info.json')
    
    models = {
        'V1 (旧版本)': model_v1_old,
        'V1.0.2 (新版本)': model_v1_new,
        'V2 (旧版本)': model_v2_old,
        'V2.0.2 (新版本)': model_v2_new,
        'V3.0.1 (旧版本)': model_v3_old,
        'V3.0.2 (新版本)': model_v3_new
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
    print("所有版本模型性能对比")
    print("=" * 80)
    print(df.to_string(index=False, float_format='%.4f'))
    
    # 分析准确率变化
    print("\n" + "=" * 80)
    print("各版本准确率变化分析")
    print("=" * 80)
    
    # 计算版本间的变化
    v1_change = model_v1_new['accuracy'] - model_v1_old['accuracy']
    v2_change = model_v2_new['accuracy'] - model_v2_old['accuracy']
    v3_change = model_v3_new['accuracy'] - model_v3_old['accuracy']
    
    print(f"V1版本变化: {v1_change:.4f} ({v1_change*100:.2f}个百分点)")
    print(f"V2版本变化: {v2_change:.4f} ({v2_change*100:.2f}个百分点)")
    print(f"V3版本变化: {v3_change:.4f} ({v3_change*100:.2f}个百分点)")
    
    print("\n" + "=" * 80)
    print("V3版本性能下降原因分析")
    print("=" * 80)
    print("1. 超参数调优状态:")
    print(f"   - V3.0.1: 进行了超参数调优 (tune_hyperparams=True)")
    print(f"   - V3.0.2: 未进行超参数调优 (tune_hyperparams=False)")
    print("   - 超参数调优对模型性能有显著影响")
    
    print("\n2. 训练数据规模:")
    print(f"   - V3.0.1: 使用单个赛季数据 ['2017-2018']")
    print(f"   - V3.0.2: 使用所有10个赛季数据")
    
    # 计算测试集规模
    v3_0_1_support = sum([model_v3_old['classification_report']['客胜']['support'],
                       model_v3_old['classification_report']['平局']['support'],
                       model_v3_old['classification_report']['主胜']['support']])
    v3_0_2_support = sum([model_v3_new['classification_report']['客胜']['support'],
                       model_v3_new['classification_report']['平局']['support'],
                       model_v3_new['classification_report']['主胜']['support']])
    
    print(f"   - V3.0.1测试集规模: {v3_0_1_support} 场比赛")
    print(f"   - V3.0.2测试集规模: {v3_0_2_support} 场比赛 (增加了 {v3_0_2_support - v3_0_1_support} 场)")
    
    print("\n3. 数据质量和一致性:")
    print("   - 多个赛季的数据可能存在格式不一致或质量差异")
    print("   - 旧赛季的数据可能与当前赛季的比赛模式存在差异")
    
    print("\n4. 特征重要性:")
    print("   - 特征数量相同 (均为80个)")
    print("   - 但不同赛季中特征的重要性可能发生变化")
    
    print("\n5. 模型泛化能力:")
    print("   - V3.0.1可能在单赛季数据上过拟合")
    print("   - V3.0.2使用更多数据但未调优，可能具有更好的泛化能力")
    print("   - 准确率下降可能是由于测试集规模扩大和数据多样性增加导致的")
    
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
    plt.figure(figsize=(12, 6))
    model_names = df['模型版本'].tolist()
    accuracies = df['整体准确率'].tolist()
    
    bars = plt.bar(model_names, accuracies, color=['#E53935', '#FFB300', '#43A047', '#66BB6A', '#1E88E5', '#42A5F5'])
    
    for bar, accuracy in zip(bars, accuracies):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{accuracy:.4f}', ha='center', va='bottom')
    
    plt.title('所有版本模型整体准确率对比', fontsize=16)
    plt.ylabel('准确率', fontsize=12)
    plt.ylim(0, 0.7)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_versions_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 胜平负精确率对比图
    plt.figure(figsize=(14, 6))
    
    x = range(len(model_names))
    width = 0.25
    
    away_win_precision = df['客胜精确率'].tolist()
    draw_precision = df['平局精确率'].tolist()
    home_win_precision = df['主胜精确率'].tolist()
    
    plt.bar([i - width for i in x], away_win_precision, width, label='客胜精确率', color='#E53935')
    plt.bar(x, draw_precision, width, label='平局精确率', color='#43A047')
    plt.bar([i + width for i in x], home_win_precision, width, label='主胜精确率', color='#1E88E5')
    
    plt.xlabel('模型版本', fontsize=12)
    plt.ylabel('精确率', fontsize=12)
    plt.title('所有版本模型胜平负精确率对比', fontsize=16)
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_versions_precision_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 胜平负召回率对比图
    plt.figure(figsize=(14, 6))
    
    away_win_recall = df['客胜召回率'].tolist()
    draw_recall = df['平局召回率'].tolist()
    home_win_recall = df['主胜召回率'].tolist()
    
    plt.bar([i - width for i in x], away_win_recall, width, label='客胜召回率', color='#E53935')
    plt.bar(x, draw_recall, width, label='平局召回率', color='#43A047')
    plt.bar([i + width for i in x], home_win_recall, width, label='主胜召回率', color='#1E88E5')
    
    plt.xlabel('模型版本', fontsize=12)
    plt.ylabel('召回率', fontsize=12)
    plt.title('所有版本模型胜平负召回率对比', fontsize=16)
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_versions_recall_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n对比图表已保存至: {output_dir}")

if __name__ == "__main__":
    compare_all_versions()