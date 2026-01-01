#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用全部赛季数据训练3.0.7版本的LightGBM模型
"""

import os
import sys
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trainers.hierarchical.hierarchical_trainer import HierarchicalModelTrainer

def main():
    """主函数"""
    # 定义数据根目录和模型保存目录
    data_root = '/Users/Williamhiler/Documents/my-project/train/train-data'
    model_dir = '/Users/Williamhiler/Documents/my-project/train/models'
    
    print("=" * 60)
    print(f"训练3.0.7版本模型 (全部赛季数据)")
    print("=" * 60)
    
    # 初始化训练器
    trainer = HierarchicalModelTrainer(data_root, model_dir)
    
    # 训练模型 - 不使用超参调优（快速训练）
    print("\n" + "=" * 60)
    print("训练模型 (默认参数 + Early Stopping)")
    print("=" * 60)
    start_time = time.time()
    
    metrics_default = trainer.train(
        seasons=None,  # 使用所有可用赛季
        include_team_state=True,
        include_expert=True,
        use_tuning=False
    )
    
    training_time = time.time() - start_time
    print(f"\n总训练时间: {training_time:.2f}秒")
    
    print("\n" + "=" * 60)
    print("默认参数模型性能总结:")
    print("=" * 60)
    print(f"准确率: {metrics_default['metrics']['accuracy']:.4f}")
    print(f"加权F1分数: {metrics_default['metrics']['weighted_f1']:.4f}")
    print(f"主队胜F1分数: {metrics_default['metrics']['home_win_f1']:.4f}")
    print(f"平局F1分数: {metrics_default['metrics']['draw_f1']:.4f}")
    print(f"客队胜F1分数: {metrics_default['metrics']['away_win_f1']:.4f}")
    
    # 提示用户是否进行超参调优
    print("\n" + "=" * 60)
    print("注意: 超参调优需要较长时间 (约数小时)")
    print("如果需要进行超参调优，请运行以下命令:")
    print(f"python {os.path.basename(__file__)} --tune")
    print("=" * 60)

if __name__ == "__main__":
    # 检查是否需要超参调优
    use_tuning = len(sys.argv) > 1 and sys.argv[1] == "--tune"
    
    if use_tuning:
        print("""
        #########################################################
        #                                                       #
        #       开始超参调优 - 这将需要较长时间 (约数小时)        #
        #                                                       #
        #########################################################
        """)
        
        data_root = '/Users/Williamhiler/Documents/my-project/train/train-data'
        model_dir = '/Users/Williamhiler/Documents/my-project/train/models'
        
        trainer = HierarchicalModelTrainer(data_root, model_dir)
        
        start_time = time.time()
        metrics_tuned = trainer.train(
            seasons=None,  # 使用所有可用赛季
            include_team_state=True,
            include_expert=True,
            use_tuning=True
        )
        
        tuning_time = time.time() - start_time
        print(f"\n总调优时间: {tuning_time:.2f}秒")
        
        print("\n" + "=" * 60)
        print("超参调优模型性能总结:")
        print("=" * 60)
        print(f"准确率: {metrics_tuned['metrics']['accuracy']:.4f}")
        print(f"加权F1分数: {metrics_tuned['metrics']['weighted_f1']:.4f}")
        print(f"主队胜F1分数: {metrics_tuned['metrics']['home_win_f1']:.4f}")
        print(f"平局F1分数: {metrics_tuned['metrics']['draw_f1']:.4f}")
        print(f"客队胜F1分数: {metrics_tuned['metrics']['away_win_f1']:.4f}")
    else:
        main()