import sys
import os

# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trainers.model_trainer import ModelTrainerV1, ModelTrainerV2, ModelTrainerV3

# 配置参数
data_root = '/Users/Williamhiler/Documents/my-project/train/train-data'
model_dir = '/Users/Williamhiler/Documents/my-project/train/models'

# 选择要训练的赛季 - 使用所有赛季
seasons = ['2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']

# 选择要使用的模型
model_name = 'xgboost'  # 可以选择 'logistic_regression', 'random_forest' 或 'xgboost'

print("="*60)
print("开始训练模型")
print("="*60)

# 训练版本1：仅使用赔率特征
print("\n" + "="*60)
print("训练版本1：仅使用赔率特征")
print("="*60)
trainer_v1 = ModelTrainerV1(data_root, model_dir)
trainer_v1.train(seasons, model_name)

# 训练版本2：使用赔率特征 + team_state特征
print("\n" + "="*60)
print("训练版本2：使用赔率特征 + team_state特征")
print("="*60)
trainer_v2 = ModelTrainerV2(data_root, model_dir)
trainer_v2.train(seasons, model_name)

# 训练版本3：使用赔率特征 + team_state特征 + 专家特征
print("\n" + "="*60)
print("训练版本3：使用赔率特征 + team_state特征 + 专家特征")
print("="*60)
trainer_v3 = ModelTrainerV3(data_root, model_dir)
trainer_v3.train(seasons, model_name)

print("\n" + "="*60)
print("模型训练完成！")
print("="*60)