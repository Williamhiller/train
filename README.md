# 足球比赛预测AI系统

基于 **Llama 3.2 1B** 的足球比赛预测系统，通过分析赔率、历史战绩、联赛排名等多维数据预测比赛结果。该系统采用先进的大型语言模型微调技术，结合量化和参数高效微调（PEFT）方法，使模型能在普通硬件上运行和微调。

## 🚀 项目功能

- **数据处理模块**：支持多种格式的足球比赛数据导入、清洗和特征工程
- **模型训练模块**：基于 Llama 3.2 1B 的高效微调，支持量化和LoRA技术
- **预测推理模块**：提供多种预测接口（命令行、批处理、交互式）
- **智能分析**：生成比赛结果预测和详细分析理由
- **数据爬虫模块**：自动抓取联赛信息、比赛详情和赔率数据

## 🛠 技术栈

- **Python 3.9+**：核心开发语言
- **PyTorch**：深度学习框架
- **Transformers (Hugging Face)**：提供预训练模型和训练工具
- **PEFT**：参数高效微调技术，减少计算资源需求
- **BitsAndBytes**：模型量化，降低内存占用
- **Pandas/NumPy/Scikit-learn**：数据处理和科学计算
- **YAML/OmegaConf**：配置管理
- **Loguru**：高级日志管理

## 📁 项目结构

```
├── data/                       # 数据目录
│   ├── raw/                    # 原始数据
│   └── processed/              # 预处理后的数据
├── models/                     # 模型存储目录
│   ├── fine_tuned_model/       # 微调后的模型
├── utils/                      # 工具函数库
│   ├── data_processor.py       # 数据处理类和函数
│   ├── model_utils.py          # 模型加载和配置函数
│   └── prediction_utils.py     # 预测相关工具函数
├── crawler/                    # 数据爬虫模块
│   ├── leagues/                # 联赛数据爬虫
│   ├── matches/                # 比赛数据爬虫
│   ├── odds/                   # 赔率数据爬虫
│   ├── utils/                  # 爬虫工具类
│   ├── config/                 # 爬虫配置
│   ├── data/                   # 爬虫数据存储
│   └── README.md               # 爬虫使用文档
├── config/                     # 配置文件
│   └── config.yaml             # 主配置文件
├── examples/                   # 示例文件
│   ├── sample_input.json       # 预测输入示例
│   └── sample_training_data.json # 训练数据示例
├── preprocess.py               # 数据预处理主脚本
├── train.py                    # 模型训练主脚本
├── predict.py                  # 基础预测脚本
├── predict_interface.py        # 增强预测接口（交互式/批量）
├── test_training.py            # 训练流程测试脚本
└── requirements.txt            # 项目依赖
```

## 📋 快速开始

### 1. 环境准备

确保安装了Python 3.9或更高版本：

```bash
python3 --version
```

### 2. 安装依赖

```bash
python3 -m pip install -r requirements.txt
```

### 3. 数据准备

#### 训练数据格式

训练数据可以是CSV或JSON格式，包含以下关键字段：

- **基本信息**：主队名、客队名、联赛、比赛日期
- **赔率信息**：主胜赔率、平局赔率、客胜赔率
- **排名信息**：主队排名、客队排名
- **近期表现**：最近5场比赛结果（W=胜，D=平，L=负）
- **历史交锋**：历史交锋记录（H=主队胜，D=平，A=客队胜）
- **进攻防守**：进球数、失球数
- **结果标签**：实际比赛结果（home_win/draw/away_win）

#### 预测数据格式

与训练数据类似，但不需要结果标签。

### 4. 数据预处理

将原始数据转换为模型可用的格式：

```bash
python3 preprocess.py \
    --input_file data/raw/matches.csv \
    --output_dir data/processed \
    --split_ratio 0.8 0.1 0.1
```

### 5. 模型训练

使用配置文件进行训练：

```bash
python3 train.py --config config/config.yaml
```

或者直接指定参数：

```bash
python3 train.py \
    --base_model meta-llama/Llama-3.2-1B \
    --train_data data/processed/train.json \
    --valid_data data/processed/valid.json \
    --output_dir models/fine_tuned_model \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --quantization
```

### 6. 预测使用

#### 基础预测

```bash
python3 predict.py \
    --model_path models/fine_tuned_model \
    --input_file examples/sample_input.json \
    --output_file prediction_result.json
```

#### 交互式预测

```bash
python3 predict_interface.py --mode interactive
```

#### 批量预测

```bash
python3 predict_interface.py \
    --mode batch \
    --input_file examples/batch_inputs.json \
    --output_file batch_predictions.json
```

## ⚙️ 配置详解

主要配置文件 `config/config.yaml` 包含以下几个部分：

### 模型配置
```yaml
model:
  base_model: "meta-llama/Llama-3.2-1B"  # 基础模型
  quantization: true                    # 是否使用量化
  device_map: "auto"                    # 设备分配
```

### 训练配置
```yaml
training:
  num_epochs: 3                         # 训练轮数
  batch_size: 4                         # 批量大小
  learning_rate: 5e-5                   # 学习率
  gradient_accumulation_steps: 2        # 梯度累积步数
  max_seq_length: 1024                  # 最大序列长度
```

### 数据配置
```yaml
data:
  train_file: "data/processed/train.json"  # 训练数据
  valid_file: "data/processed/valid.json"  # 验证数据
  test_file: "data/processed/test.json"    # 测试数据
```

### PEFT配置
```yaml
peft:
  use_peft: true                        # 是否使用PEFT
  lora_r: 8                             # LoRA秩
  lora_alpha: 32                        # LoRA alpha参数
  lora_dropout: 0.1                     # LoRA dropout
```

## 📊 模型性能评估

训练完成后，模型会自动在验证集上评估，并生成以下指标：

- **准确率**：预测正确的比赛比例
- **F1分数**：综合精确率和召回率的指标
- **混淆矩阵**：展示各类别的预测情况
- **ROC曲线**：展示分类器的性能

## 🚦 常见问题

### 内存不足错误
- 使用 `--quantization` 参数启用模型量化
- 减小 `batch_size` 和 `gradient_accumulation_steps`
- 使用更小的 `max_seq_length`

### 训练速度慢
- 启用梯度累积
- 检查是否使用了GPU加速
- 考虑使用更高效的PEFT配置

### 预测质量不佳
- 增加训练数据量和多样性
- 调整学习率和训练轮数
- 优化数据预处理和特征工程

## 📝 示例输入输出

### 输入示例
```json
{
  "home_team": "曼联",
  "away_team": "利物浦",
  "home_odds": 3.25,
  "draw_odds": 3.50,
  "away_odds": 2.10,
  "home_ranking": 5,
  "away_ranking": 1,
  "league": "英超联赛",
  "home_recent_results": ["W", "D", "W", "L", "W"],
  "away_recent_results": ["W", "W", "W", "W", "D"],
  "head_to_head": ["D", "L", "A", "D", "H"]
}
```

### 输出示例
```json
{
  "prediction": "L",
  "confidence": 0.72,
  "detailed_probabilities": {
    "home_win": 0.18,
    "draw": 0.10,
    "away_win": 0.72
  },
  "score": "1-2",
  "analysis": "根据客队的联赛排名（第1）、近期出色表现（4胜1平）和较低的赔率（2.10），预计客队将获胜。主队近期状态不稳定，且历史交锋处于劣势。",
  "home_team": "曼联",
  "away_team": "利物浦",
  "league": "英超联赛"
}

## 🧪 开发和测试

### 测试训练流程

使用简化的测试脚本验证训练环境：

```bash
python3 test_training.py
```

### 添加新功能

1. 在 `utils/` 目录下添加新的工具函数
2. 更新主脚本以调用新功能
3. 更新配置文件以支持新参数
4. 编写适当的测试

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交问题和拉取请求！

## 📊 数据采集

系统包含完整的数据爬虫模块，用于自动获取足球比赛数据：

- **联赛数据抓取**：获取联赛信息、积分榜、球队列表等
- **比赛数据抓取**：获取比赛详情、进球、统计数据等
- **赔率数据抓取**：获取多家博彩公司的赔率数据
- **按联赛分类存储**：所有数据按联赛和比赛ID进行组织

查看爬虫模块文档：
```bash
cd crawler
cat README.md
```

## 📧 联系方式

如有任何问题，请通过 Issues 与我们联系。