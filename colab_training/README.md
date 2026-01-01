# Colab训练部署说明

## 目录结构

```
train/colab_training/
├── train_lora_local.py               # 第一步训练脚本 - 本地版本
├── train_lora_colab.py               # 第一步训练脚本 - Colab版本
├── train_with_match_data.py          # 第二步训练脚本 - 本地版本
├── train_with_match_data_colab.py    # 第二步训练脚本 - Colab版本
├── test_script_local.py              # 第一步训练测试脚本 - 本地版本
├── test_match_training_local.py      # 第二步训练测试脚本 - 本地版本
└── README.md                         # 部署说明文档
```

## 训练流程

### 第一步：使用专家数据进行初始训练

**功能**：使用指令-回答格式的专家数据对Qwen模型进行初始LoRA微调

**数据**：`/Users/Williamhiler/Documents/my-project/train/v5/data/expert_data/qwen_finetune_data.json`

**输出**：`/Users/Williamhiler/Documents/my-project/train/colab_training/out/qwen_lora_final`

### 第二步：使用比赛数据进行进一步训练

**功能**：基于第一步训练好的LoRA权重，使用英超2015-2016赛季比赛数据进行进一步微调

**数据**：`/Users/Williamhiler/Documents/my-project/train/examples/英超_2015-2016_aggregated.json`

**输出**：`/Users/Williamhiler/Documents/my-project/train/colab_training/out/match_finetune/final_lora`

## Colab部署步骤

### 1. 准备工作

1. 打开Colab：https://colab.research.google.com/
2. 选择"新建笔记本" -> 选择"Python 3"和"T4 GPU"运行时
3. 挂载Google Drive（可选，用于保存训练结果）

### 2. 安装依赖

在Colab中运行以下命令安装所需依赖：

```bash
# 安装基础依赖
pip install -q transformers torch datasets peft trl accelerate bitsandbytes

# 安装Unsloth（可选，用于加速训练）
pip install -q unsloth
```

### 3. 从GitHub拉取项目

```bash
# 克隆项目到Colab
git clone https://github.com/Williamhiller/train.git
cd train/colab_training

# 创建输出目录
mkdir -p out
```

### 4. 运行第一步训练

#### 本地版本
```bash
# 运行第一步训练脚本（本地版本）
python train_lora_local.py
```

#### Colab版本
```bash
# 运行第一步训练脚本（Colab版本）
python train_lora_colab.py
```

**训练参数说明**：
- 模型：Qwen2.5-0.5B-Instruct
- LoRA配置：r=16, alpha=32, dropout=0.05
- 训练批次：
  - 本地版本：batch_size=4, gradient_accumulation_steps=4
  - Colab版本：batch_size=8, gradient_accumulation_steps=2
- 最大训练步数：300
- 学习率：1e-5

**Colab版本优化**：
- 直接从Hugging Face下载模型，不需要本地路径
- 增大批次大小，充分利用GPU内存
- 减少梯度累积步数，加速训练
- 支持自动设备映射
- 启用8位优化器（如果GPU可用）

### 5. 运行第二步训练

#### 本地版本
```bash
# 运行第二步训练脚本（本地版本）
python train_with_match_data.py
```

#### Colab版本
```bash
# 运行第二步训练脚本（Colab版本）
python train_with_match_data_colab.py
```

**训练参数说明**：
- 基于第一步训练的LoRA权重
- 比赛数据：英超2015-2016赛季380场比赛
- 特征：赔率、球队状态、历史战绩等
- 训练批次：
  - 本地版本：batch_size=4, gradient_accumulation_steps=4
  - Colab版本：batch_size=8, gradient_accumulation_steps=2
- 最大训练步数：300
- 学习率：1e-5

**Colab版本优化**：
- 直接从Hugging Face下载模型，不需要本地路径
- 增大批次大小，充分利用GPU内存
- 减少梯度累积步数，加速训练
- 支持自动设备映射
- 启用8位优化器（如果GPU可用）

## 测试脚本使用

### 测试第一步训练

```bash
# 测试第一步训练的核心功能（本地版本）
python test_script_local.py
```

**测试内容**：
- 模型加载
- LoRA配置
- 数据加载
- 权重保存

### 测试第二步训练

```bash
# 测试第二步训练的核心功能（本地版本）
python test_match_training_local.py
```

**测试内容**：
- 比赛数据加载
- 特征提取
- 数据集构建
- 模型和LoRA权重加载
- 模型保存

## 本地测试

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+（可选，用于GPU训练）

### 安装依赖

```bash
pip install transformers torch datasets peft trl accelerate bitsandbytes
```

### 运行测试

```bash
# 切换到colab_training目录
cd /Users/Williamhiler/Documents/my-project/train/colab_training

# 运行测试脚本（本地版本）
python test_script_local.py
python test_match_training_local.py
```

## 训练结果

### 第一步训练结果

- 输出目录：`out/qwen_lora_final`
- 包含文件：
  - `adapter_model.safetensors` - LoRA权重文件
  - `adapter_config.json` - LoRA配置文件
  - `tokenizer.json` - 分词器文件
  - 其他配置文件

### 第二步训练结果

- 输出目录：`out/match_finetune/final_lora`
- 包含文件：
  - `adapter_model.safetensors` - 微调后的LoRA权重
  - `adapter_config.json` - LoRA配置文件
  - `tokenizer.json` - 分词器文件
  - 其他配置文件

## 模型使用

### 加载模型和LoRA权重

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载基础模型
base_model = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)

# 加载LoRA权重
lora_weights = "./out/match_finetune/final_lora"
model = PeftModel.from_pretrained(model, lora_weights)
```

### 生成预测

```python
# 构建输入
input_text = "### 指令：\n请分析以下比赛数据...\n\n### 回答："
inputs = tokenizer(input_text, return_tensors="pt")

# 生成输出
generate_ids = model.generate(
    inputs.input_ids,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

# 解码输出
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)
```

## 常见问题

### 1. CUDA内存不足

**解决方案**：
- 减小`per_device_train_batch_size`
- 增大`gradient_accumulation_steps`
- 使用`load_in_4bit=True`或`load_in_8bit=True`加载模型

### 2. 训练速度慢

**解决方案**：
- 使用GPU训练
- 安装Unsloth加速库
- 减小模型大小
- 减小训练数据量

### 3. 模型生成质量差

**解决方案**：
- 增加训练步数
- 调整学习率
- 增加训练数据量
- 调整LoRA配置（增大r值）

### 4. 无法找到本地模型

**解决方案**：
- 确保模型路径正确
- 使用绝对路径
- 设置`local_files_only=True`

## 性能优化

### GPU训练

- 使用A100或V100 GPU可获得最佳性能
- 确保CUDA版本与PyTorch兼容
- 使用`fp16`或`bf16`精度训练

### 内存优化

- 使用4-bit或8-bit量化
- 启用梯度检查点
- 减小`max_seq_length`

### 训练时长

- 第一步训练：约10-30分钟（取决于GPU）
- 第二步训练：约15-45分钟（取决于GPU）

## 监控训练

### 查看训练日志

训练过程中会输出以下信息：
- 训练步数
- 损失值
- 学习率
- 训练速度

### 保存检查点

训练过程中会定期保存检查点：
- 每50步保存一次
- 最多保存2个检查点
- 保存到`out/`目录下

## 后续步骤

### 模型部署

- 使用Hugging Face Inference API部署
- 使用FastAPI或Flask构建API服务
- 集成到Web应用或移动应用

### 模型评估

- 使用测试数据集评估模型性能
- 计算准确率、F1分数等指标
- 人工评估生成结果质量

### 继续优化

- 收集更多训练数据
- 调整模型参数
- 尝试不同的LoRA配置
- 尝试更大的模型

---

**作者**：Williamhiler  
**日期**：2026-01-01  
**版本**：v1.0