import sys
import os

# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trainers.model_trainer import ModelTrainerV2

# 配置参数
data_root = '/Users/Williamhiler/Documents/my-project/train/train-data'
model_dir = '/Users/Williamhiler/Documents/my-project/models'

# 选择要训练的赛季 - 使用所有赛季
seasons = ['2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']

# 选择要使用的模型
model_name = 'lightgbm'  # 可以选择 'logistic_regression', 'random_forest' 或 'lightgbm'

print("="*60)
print("开始训练2.0.3版本模型（去掉专家数据） - 超参优化")
print("="*60)

# 训练版本2.0.3：使用赔率特征 + team_state特征，去掉专家数据
trainer_v2 = ModelTrainerV2(data_root, model_dir)
trainer_v2.train(
    seasons, 
    model_name=model_name, 
    tune_hyperparams=True,
    custom_version="2.0.3"
)

print("\n" + "="*60)
print("模型训练完成！")
print("="*60)
