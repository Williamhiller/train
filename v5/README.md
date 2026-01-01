# 足球预测模型 V5

## 概述

V5版本是一个融合了Qwen大语言模型、赔率数据和球队近况信息的智能足球比赛预测系统。该系统通过改进的上下文匹配机制，能够自动生成全面的比赛上下文，并与2198个专家知识单元进行精确匹配，最终生成高质量的赛果预测。系统支持对话式交互和API调用两种方式。

## 主要特性

1. **多模态融合**: 结合Qwen大语言模型的文本理解能力和结构化数据分析
2. **对话式交互**: 支持自然语言输入球队信息和赔率数据
3. **专家知识整合**: 融合专家分析经验和历史数据，包含2198个专家知识单元
4. **分批次训练**: 支持每批次32场比赛的高效训练，内存占用小
5. **可升级架构**: 支持模型参数保存和增量训练
6. **实时预测**: 快速响应预测请求
7. **实用版增强特征工程**: 集成专家知识的增强特征生成
8. **全面上下文生成**: 自动生成包含比赛ID、详细赔率变化、本赛季球队数据、近6场表现和对战历史的全面上下文
9. **精确上下文匹配**: 基于丰富上下文与专家知识单元进行精确匹配，提高预测准确性

## 目录结构

```
v5/
├── README.md                    # 本文件
├── USAGE.md                     # 使用说明
├── prepare_data.py              # 数据准备脚本
├── train_batch.py               # 分批次训练脚本
├── run_chat.py                  # 对话式交互
├── run_api.py                   # 预测API
├── run_training.py              # 训练主脚本
├── context_generator.py         # 上下文生成器
├── test_context_matching.py     # 上下文匹配测试脚本
├── configs/                     # 配置文件
│   └── v5_config.yaml          # V5模型配置
├── data/                        # 数据目录
│   ├── processed/               # 处理后的数据
│   │   ├── processed_matches.csv  # 3790场比赛数据
│   │   └── processed_matches.json
│   ├── enhanced_features.csv    # 增强特征数据
│   └── expert_knowledge/        # 专家知识数据
│       ├── expert_knowledge_base.json  # 2198个知识单元
│       ├── category_index.json       # 分类索引
│       ├── concept_index.json        # 概念索引
│       ├── expert_knowledge_editable.csv  # 可编辑专家知识
│       └── preprocessing_report.json # 预处理报告
├── models/                      # 模型定义
│   ├── qwen_adapter/            # Qwen适配器
│   │   └── qwen_adapter.py
│   ├── prediction_head/         # 预测头
│   │   └── prediction_head.py
│   └── fusion_model.py          # 融合模型
├── trainers/                    # 训练脚本
│   └── train_v5.py              # V5训练器
├── utils/                       # 工具函数
│   ├── data_processing/         # 数据处理
│   │   ├── data_loader.py
│   │   ├── expert_data_processor.py
│   │   └── match_data_processor.py
│   ├── expert_knowledge/        # 专家知识处理
│   │   ├── expert_knowledge_editor.py
│   │   ├── expert_knowledge_preprocessor.py
│   │   ├── expert_knowledge_reasoner.py
│   │   └── practical_expert_reasoner.py
│   ├── feature_engineering/     # 特征工程
│   │   ├── feature_engineer.py
│   │   └── practical_enhanced_feature_engineer.py
│   └── model_utils/             # 模型工具
│       ├── checkpoint.py
│       └── evaluation.py
└── inference/                   # 推理接口
    ├── chat_interface.py        # 对话接口
    └── prediction_api.py        # 预测API
```

## 核心功能模块

### 1. 数据准备
- **数据加载**: 支持多种数据源的统一加载
- **数据清洗**: 自动处理缺失值和异常值
- **特征工程**: 生成基础特征和增强特征
- **专家知识整合**: 集成2198个专家知识单元

### 2. 模型架构
- **Qwen适配器**: 连接大语言模型
- **结构化数据编码器**: 处理赔率和球队数据
- **融合层**: 整合文本和结构化信息
- **预测头**: 输出比赛结果预测

### 3. 上下文生成
- **全面上下文生成**: 自动生成包含比赛ID、详细赔率变化、本赛季球队数据、近6场表现和对战历史的全面上下文
- **多维度支持**: 支持基本信息、赔率信息、球队表现、近期状态、交锋历史和赛季数据等6个维度
- **菠菜公司细分**: 详细区分威廉、立博等菠菜公司的初赔、终赔和赔率变化
- **近6场表现**: 统计主客队近6场比赛的胜平负数量
- **详细对战历史**: 包含总对战记录和最近3场的具体结果

### 4. 专家知识系统
- **专家知识预处理**: 自动分块和语义标注
- **知识编辑接口**: 支持手动调整专家知识
- **精确上下文匹配**: 基于全面上下文与2198个专家知识单元进行精确匹配
- **智能推理**: 基于赔率模式和球队表现匹配相关专家知识
- **实用版推理**: 每批32场比赛的高效推理

### 5. 训练系统
- **分批次训练**: 每批32场比赛，内存占用小
- **自适应学习率**: 自动调整学习率
- **模型保存**: 自动保存最佳模型
- **训练监控**: 实时显示训练进度和准确率

## 使用方法

### 1. 数据准备

```bash
python prepare_data.py
```

### 2. 训练模型

```bash
python train_batch.py
```

### 3. 对话式交互

```bash
python run_chat.py
```

### 4. 启动预测API

```bash
python run_api.py
```

## 训练配置

```yaml
# v5_config.yaml
training:
  batch_size: 32          # 每批32场比赛
  num_epochs: 20           # 训练20轮
  learning_rate: 0.001    # 学习率
  
model:
  hidden_size: 128         # 隐藏层大小
  dropout_rate: 0.3        # Dropout率
  
data:
  enhanced_features: true  # 使用增强特征
  expert_knowledge: true   # 使用专家知识
```

## 专家知识系统

### 知识单元统计
- **总知识单元**: 2198个
- **知识分类**: 7个
- **关键概念**: 213个
- **处理文档**: 4份PDF文档

### 知识分类
1. **赔率的哲学思维和认知方法**: 400个单元
2. **实战应用和案例分析**: 1305个单元
3. **心理分析和行为模式**: 117个单元
4. **具体的分析技巧和判断标准**: 265个单元
5. **常见的赔率模式和异常情况**: 14个单元
6. **赔率分析的基础理论**: 79个单元
7. **风险控制和资金管理**: 18个单元

## 性能指标

- **预测准确率**: 约XX%
- **胜平负预测F1分数**: 约XX%
- **内存占用**: 每批次约XX MB
- **训练时间**: 约XX分钟

## 技术栈

- **Python 3.13**
- **PyTorch**
- **Pandas & NumPy**
- **Qwen2.5-1.5B**
- **FastAPI** (API服务)
- **scikit-learn** (特征工程)

## 注意事项

1. 确保Qwen模型已正确下载和配置
2. 数据预处理需要按照指定格式
3. 模型训练需要充足的计算资源
4. 定期更新专家知识库
5. 支持GPU加速（如果可用）

## 未来计划

1. **模型优化**: 进一步提升预测准确率
2. **多语言支持**: 支持更多语言的对话交互
3. **实时数据集成**: 接入实时赔率和球队数据
4. **可视化界面**: 增加结果可视化展示
5. **扩展体育类型**: 支持更多体育项目的预测

## 许可证

MIT License

## 更新日志

### v5.0.0 (2025-12-28)
- 初始版本发布
- 融合Qwen大语言模型
- 实现专家知识系统
- 支持分批次训练
- 提供对话式交互
- 支持预测API

### v5.0.1 (2025-12-28)
- 优化专家知识推理
- 改进特征工程
- 修复已知问题
- 清理项目结构
- 更新文档

---

项目维护: Trae AI
更新日期: 2025-12-28