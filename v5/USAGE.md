# 足球预测模型 V5 使用指南

## 1. 项目概述

足球预测模型 V5 是一个融合了 Qwen 大语言模型、赔率数据和专家知识的智能足球比赛预测系统。本指南将帮助您快速上手使用该系统进行数据准备、模型训练和比赛预测。

## 2. 安装要求

### 2.1 环境配置
- Python 3.13
- PyTorch 2.0+
- CUDA 11.8+ (可选，用于GPU加速)

### 2.2 依赖安装

```bash
# 安装基础依赖
pip install numpy pandas scikit-learn torch fastapi uvicorn

# 安装大语言模型依赖
pip install transformers tokenizers
```

## 3. 数据准备

### 3.1 准备原始数据

将原始数据放置在以下目录:
```
/Users/Williamhiler/Documents/my-project/train/original-data/
```

### 3.2 运行数据准备脚本

```bash
python prepare_data.py
```

该脚本将执行以下操作:
- 加载原始数据
- 数据清洗和预处理
- 特征工程
- 生成专家知识特征
- 保存处理后的数据到 `data/processed/` 目录
- 保存增强特征到 `data/enhanced_features.csv`

### 3.3 数据格式

处理后的数据包含以下主要字段:
- 比赛基本信息（球队名称、赛季、比赛日期）
- 赔率数据（主胜、平局、客胜赔率）
- 球队状态数据（近期积分、胜率、交锋历史）
- 专家知识特征（基于专家分析的预测概率）

## 4. 模型训练

### 4.1 分批次训练（推荐）

使用 `train_batch.py` 进行分批次训练，每批次处理32场比赛:

```bash
python train_batch.py
```

### 4.2 训练参数配置

您可以在 `configs/v5_config.yaml` 中调整训练参数:

```yaml
training:
  batch_size: 32          # 每批比赛数量
  num_epochs: 20           # 训练轮数
  learning_rate: 0.001    # 学习率
  
model:
  hidden_size: 128         # 隐藏层大小
  dropout_rate: 0.3        # Dropout率
  
data:
  enhanced_features: true  # 是否使用增强特征
  expert_knowledge: true   # 是否使用专家知识
```

### 4.3 训练过程

训练过程中，系统会:
- 显示每10个批次的训练进度
- 自动保存最佳模型到 `models/` 目录
- 生成训练历史记录
- 输出最终的分类报告

### 4.4 训练结果

训练完成后，您将获得:
- 最佳模型文件: `models/best_model.pth`
- 最终模型文件: `models/final_model.pth`
- 训练历史: `models/training_history.json`

## 5. 比赛预测

### 5.1 对话式交互

使用对话界面进行自然语言预测:

```bash
python run_chat.py
```

**示例交互:**
```
用户: 预测一下曼联 vs 利物浦的比赛结果，曼联近期积分12，利物浦10，赔率2.1/3.4/3.2
系统: 基于Qwen大语言模型和专家分析，预测结果为...
```

### 5.2 预测API

启动预测API服务:

```bash
python run_api.py --port 8000
```

**API端点:**
- `POST /predict`: 接收比赛数据，返回预测结果
- `GET /status`: 查看API状态

**请求示例:**
```json
{
  "home_team": "曼联",
  "away_team": "利物浦",
  "home_win_odds": 2.1,
  "draw_odds": 3.4,
  "away_win_odds": 3.2,
  "home_recent_points": 12,
  "away_recent_points": 10
}
```

**响应示例:**
```json
{
  "prediction": "主胜",
  "probabilities": {
    "home_win": 0.45,
    "draw": 0.30,
    "away_win": 0.25
  },
  "confidence": 0.82,
  "expert_analysis": "根据专家知识和历史数据..."
}
```

## 6. 专家知识管理

### 6.1 查看专家知识

专家知识库位于:
```
data/expert_knowledge/expert_knowledge_base.json
```

### 6.2 编辑专家知识

使用专家知识编辑器调整和优化知识库:

```bash
python utils/expert_knowledge/expert_knowledge_editor.py
```

该编辑器支持:
- 浏览知识单元
- 搜索相关知识
- 编辑知识类型和评分
- 批量修改分类
- 导出/导入CSV格式

### 6.3 专家知识统计

- 总知识单元: 2198个
- 知识分类: 7个
- 关键概念: 213个
- 处理文档: 4份

## 7. 模型架构

### 7.1 核心组件

1. **Qwen适配器**: 将大语言模型的文本理解能力应用于结构化数据
2. **结构化数据编码器**: 处理赔率和球队状态数据
3. **融合层**: 整合文本和结构化信息
4. **预测头**: 输出胜平负预测概率
5. **专家知识推理器**: 基于专家经验调整预测结果

### 7.2 模型文件

- **Qwen适配器**: `models/qwen_adapter/qwen_adapter.py`
- **预测头**: `models/prediction_head/prediction_head.py`
- **融合模型**: `models/fusion_model.py`

## 8. 高级功能

### 8.1 增量训练

支持在现有模型基础上进行增量训练:

```bash
python train_batch.py --resume_from_checkpoint models/best_model.pth
```

### 8.2 模型评估

使用测试集评估模型性能:

```bash
python -c "
from utils.model_utils.evaluation import evaluate_model
model_path = 'models/best_model.pth'
data_path = 'data/enhanced_features.csv'
evaluate_model(model_path, data_path)
"
```

### 8.3 自定义特征工程

您可以通过修改 `utils/feature_engineering/feature_engineer.py` 来自定义特征生成逻辑。

## 9. 故障排除

### 9.1 常见问题

#### 问题1: 内存不足
**解决方法**: 减小批次大小或使用更简单的模型架构

#### 问题2: 训练速度慢
**解决方法**: 使用GPU加速或减小训练轮数

#### 问题3: 预测准确率低
**解决方法**: 
- 增加训练轮数
- 优化特征工程
- 更新专家知识库
- 调整模型参数

#### 问题4: 专家知识特征生成失败
**解决方法**: 检查专家知识库路径是否正确，确保 `expert_knowledge_base.json` 文件存在

### 9.2 日志查看

训练和推理过程中的日志将输出到控制台，您可以查看日志了解系统运行状态。

## 10. 使用示例

### 10.1 完整工作流程

```bash
# 1. 准备数据
python prepare_data.py

# 2. 训练模型
python train_batch.py

# 3. 启动对话界面
python run_chat.py

# 或者启动API服务
python run_api.py
```

### 10.2 预测示例

**输入:**
```
预测曼城 vs 切尔西的比赛结果，曼城近期积分15，切尔西12，赔率1.8/3.4/4.2
```

**输出:**
```
基于Qwen大语言模型和专家知识分析，预测结果如下:
- 主胜概率: 47.4%
- 平局概率: 14.7%
- 客胜概率: 37.9%
- 置信度: 85.0%
- 推荐: 主胜

专家分析: 曼城近期状态良好，赔率支持主胜，结合专家知识分析，主胜可能性较高。
```

## 11. 性能指标

### 11.1 训练指标
- 训练集准确率: ~XX%
- 验证集准确率: ~XX%
- F1分数: ~XX%
- 训练时间: ~XX分钟

### 11.2 推理指标
- 单场预测时间: ~0.5秒
- API响应时间: ~1秒

## 12. 注意事项

1. **模型更新**: 定期更新Qwen模型和专家知识库以获得更好的预测效果
2. **数据更新**: 及时更新比赛数据和赔率信息
3. **风险提示**: 足球比赛结果存在不确定性，预测结果仅供参考
4. **系统监控**: 定期检查系统运行状态，确保服务正常

## 13. 联系与支持

如有任何问题或建议，请联系项目维护团队。

---

**项目版本**: V5.0.1  
**更新日期**: 2025-12-28  
**维护团队**: Trae AI