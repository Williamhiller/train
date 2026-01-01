import sys
import os

# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trainers.hierarchical.hierarchical_trainer import HierarchicalModelTrainer

# 配置参数
data_root = '/Users/Williamhiler/Documents/my-project/train/train-data'
model_dir = '/Users/Williamhiler/Documents/my-project/models'

# 选择要训练的赛季 - 使用所有赛季
seasons = ['2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']

print("="*60)
print("开始训练2.0.4版本分层模型（去掉专家数据）")
print("="*60)

# 创建分层训练器实例
trainer = HierarchicalModelTrainer(data_root, model_dir)
trainer.version = "2.0.4"
trainer.model_save_dir = os.path.join(model_dir, "v2", "2.0.4")
os.makedirs(trainer.model_save_dir, exist_ok=True)

# 训练2.0.4版本：使用赔率特征 + team_state特征，去掉专家数据，采用分层策略
trainer.train(
    seasons, 
    include_expert=False,  # 去掉专家数据
    use_tuning=True  # 使用超参优化
)

print("\n" + "="*60)
print("模型训练完成！")
print("="*60)