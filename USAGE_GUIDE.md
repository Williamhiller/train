# 足球比赛预测AI系统使用指南

本文档详细说明如何使用足球比赛预测AI系统进行数据处理、模型训练和比赛预测。

## 目录

1. [系统概述](#系统概述)
2. [环境准备](#环境准备)
3. [数据准备与处理](#数据准备与处理)
4. [模型训练](#模型训练)
5. [比赛预测](#比赛预测)
6. [结果分析](#结果分析)
7. [高级功能](#高级功能)
8. [故障排除](#故障排除)

## 系统概述

本系统基于Llama 3.2 1B模型，通过微调使其能够根据足球比赛的相关数据预测比赛结果。系统主要包含以下模块：

- **数据处理模块**：处理和转换足球比赛数据
- **模型训练模块**：微调Llama模型
- **预测推理模块**：使用训练好的模型进行预测

## 环境准备

### 1. 检查Python版本

```bash
python3 --version
```

确保Python版本为3.9或更高。

### 2. 安装依赖

```bash
# 进入项目目录
cd /Users/Williamhiler/Documents/my-project/train

# 安装所有依赖包
python3 -m pip install -r requirements.txt
```

### 3. 环境检查

运行测试脚本来验证环境是否正确配置：

```bash
python3 test_training.py
```

如果测试成功，说明环境配置正确。

## 数据准备与处理

### 1. 数据格式要求

#### 训练数据

训练数据需要包含以下关键字段：

- **基本信息**：
  - `home_team`: 主队名称
  - `away_team`: 客队名称
  - `league`: 联赛名称
  - `match_date`: 比赛日期 (YYYY-MM-DD格式)

- **赔率信息**：
  - `home_odds`: 主胜赔率
  - `draw_odds`: 平局赔率
  - `away_odds`: 客胜赔率

- **排名信息**：
  - `home_ranking`: 主队排名
  - `away_ranking`: 客队排名

- **近期表现**：
  - `home_recent_results`: 主队最近5场比赛结果 (W=胜, D=平, L=负)
  - `away_recent_results`: 客队最近5场比赛结果

- **攻防数据**（可选）：
  - `home_goals_scored`: 主队进球数
  - `home_goals_conceded`: 主队失球数
  - `away_goals_scored`: 客队进球数
  - `away_goals_conceded`: 客队失球数

- **历史交锋**（可选）：
  - `head_to_head`: 历史交锋记录 (H=主队胜, D=平, A=客队胜)

- **结果标签**（仅训练数据需要）：
  - `result`: 比赛结果 (home_win, draw, away_win)
  - `score`: 比赛比分 (如 "2-1")

参考示例：`examples/sample_training_data.json`

#### 预测数据

预测数据格式与训练数据类似，但不需要包含结果标签 (`result` 和 `score`)。

参考示例：`examples/sample_input.json` 和 `examples/batch_inputs.json`

### 2. 数据预处理

使用 `preprocess.py` 脚本处理原始数据：

```bash
python3 preprocess.py \
    --input_file data/raw/matches.csv \
    --output_dir data/processed \
    --split_ratio 0.8 0.1 0.1 \
    --format json
```

参数说明：
- `--input_file`: 输入数据文件路径 (支持CSV或JSON格式)
- `--output_dir`: 输出目录路径
- `--split_ratio`: 训练集、验证集、测试集的划分比例
- `--format`: 输出格式 (json 或 csv)

处理后的文件将被保存在指定的输出目录中，包括：
- `train.json`: 训练集
- `valid.json`: 验证集
- `test.json`: 测试集

## 模型训练

### 1. 使用配置文件训练

修改 `config/config.yaml` 文件中的配置参数，然后运行：

```bash
python3 train.py --config config/config.yaml
```

### 2. 直接指定参数训练

```bash
python3 train.py \
    --base_model meta-llama/Llama-3.2-1B \
    --train_data data/processed/train.json \
    --valid_data data/processed/valid.json \
    --output_dir models/fine_tuned_model \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --quantization \
    --use_peft \
    --lora_r 8
```

重要参数说明：
- `--base_model`: 基础模型名称
- `--train_data`: 训练数据路径
- `--valid_data`: 验证数据路径
- `--output_dir`: 模型保存路径
- `--num_epochs`: 训练轮数
- `--batch_size`: 批量大小
- `--learning_rate`: 学习率
- `--quantization`: 启用模型量化（节省内存）
- `--use_peft`: 使用参数高效微调
- `--lora_r`: LoRA秩（影响微调效果和内存使用）

### 3. 训练监控

训练过程中会显示以下信息：
- 训练轮数和步数
- 损失值
- 验证准确率
- 训练时间

训练完成后，模型将保存在指定的输出目录中。

## 比赛预测

系统提供三种预测模式：基础预测、交互式预测和批量预测。

### 1. 基础预测

使用 `predict.py` 进行单个比赛预测：

```bash
python3 predict.py \
    --model_path models/fine_tuned_model \
    --input_file examples/sample_input.json \
    --output_file prediction_result.json
```

参数说明：
- `--model_path`: 训练好的模型路径
- `--input_file`: 输入数据文件路径
- `--output_file`: 预测结果输出文件路径

### 2. 交互式预测

使用 `predict_interface.py` 的交互式模式：

```bash
python3 predict_interface.py --mode interactive
```

在交互界面中，按照提示输入比赛相关信息：
- 主队名称
- 客队名称
- 联赛名称
- 赔率信息
- 排名信息

系统会实时生成预测结果并显示，同时提供保存结果的选项。

### 3. 批量预测

对多个比赛进行批量预测：

```bash
python3 predict_interface.py \
    --mode batch \
    --input_file examples/batch_inputs.json \
    --output_file batch_predictions.json
```

批量预测会处理输入文件中的所有比赛数据，并将结果保存在指定的输出文件中。

## 结果分析

### 预测结果格式

预测结果包含以下字段：

- `prediction`: 预测结果 (W=主胜, D=平局, L=客胜)
- `confidence`: 预测置信度 (0-1之间)
- `detailed_probabilities`: 详细的概率分布
  - `home_win`: 主胜概率
  - `draw`: 平局概率
  - `away_win`: 客胜概率
- `score`: 预测比分 (如果模型能够生成)
- `analysis`: 预测分析理由 (如果模型能够生成)
- `home_team`: 主队名称
- `away_team`: 客队名称
- `league`: 联赛名称
- `timestamp`: 预测时间戳

### 结果解读

- **置信度**: 表示模型对预测结果的确信程度，值越高表示预测越可靠
- **概率分布**: 展示三种可能结果的概率分布，可用于风险评估
- **分析理由**: 模型生成的预测依据，帮助理解预测逻辑

## 高级功能

### 1. 模型量化调优

如果遇到内存不足问题，可以调整量化设置：

```bash
python3 train.py \
    --base_model meta-llama/Llama-3.2-1B \
    --quantization \
    --load_in_4bit \
    --bnb_4bit_compute_dtype float16
```

### 2. PEFT参数调整

LoRA参数会影响微调效果和内存使用：

```bash
python3 train.py \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules q_proj k_proj v_proj o_proj
```

参数说明：
- `--lora_r`: LoRA秩，影响微调容量（值越大容量越高但内存占用越多）
- `--lora_alpha`: LoRA alpha参数，与秩共同决定适应能力
- `--lora_dropout`: 防止过拟合的dropout值
- `--target_modules`: 应用LoRA的目标模块

### 3. 自定义预测提示

可以在 `utils/prediction_utils.py` 文件中修改 `prepare_prediction_input` 函数，自定义模型的输入提示格式，以获得更好的预测结果。

## 故障排除

### 常见错误

#### 内存不足

- 错误信息: `CUDA out of memory` 或 `Killed`
- 解决方案:
  - 启用模型量化: 添加 `--quantization` 参数
  - 减小批量大小: 降低 `--batch_size` 的值
  - 使用参数高效微调: 添加 `--use_peft` 参数
  - 在CPU上运行: 添加 `--device cpu` 参数

#### 模型下载失败

- 错误信息: `Model not found` 或网络错误
- 解决方案:
  - 检查网络连接
  - 确保Hugging Face访问权限正确
  - 尝试手动下载模型并指定本地路径

#### 数据格式错误

- 错误信息: `KeyError` 或 `ValueError`
- 解决方案:
  - 检查输入数据是否包含所有必要字段
  - 验证字段名称和格式是否正确
  - 参考示例数据格式 (`examples/` 目录下的文件)

### 性能优化

- **训练速度优化**:
  - 使用GPU加速（如果可用）
  - 调整 `gradient_accumulation_steps` 参数
  - 使用混合精度训练: `--fp16` 或 `--bf16`

- **预测质量优化**:
  - 增加训练数据量和多样性
  - 调整 `temperature` 和 `top_p` 参数
  - 优化数据预处理流程
  - 增加训练轮数或调整学习率

## 常见问题解答

**Q: 模型需要多少训练数据才能达到良好的预测效果？**
A: 建议至少准备1000场比赛的历史数据。数据越多、越多样化，预测效果越好。

**Q: 系统可以在没有GPU的机器上运行吗？**
A: 可以，但需要启用量化和PEFT技术。CPU训练会比较慢，建议在有GPU的环境中进行训练。

**Q: 如何评估模型的预测准确率？**
A: 训练完成后，模型会自动在验证集和测试集上评估，并输出准确率、F1分数等指标。

**Q: 可以预测具体比分吗？**
A: 是的，但比分预测的准确度通常低于胜平负预测。模型会尝试生成可能的比分，但仅供参考。

**Q: 如何更新模型以适应新的比赛数据？**
A: 可以使用新数据继续微调现有模型，或者重新训练一个新模型。继续微调通常需要更少的训练轮数。

## 联系与支持

如有任何问题或建议，请通过项目Issues提交反馈。