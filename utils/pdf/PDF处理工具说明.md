# PDF处理工具说明

## 1. 概述

本目录包含一套完整的PDF文本提取、清洗、分析和特征提取工具，用于从足球赔率分析相关的PDF文件中提取专家思路和特征，为机器学习模型提供专家知识支持。

## 2. 文件结构

```
utils/pdf/
├── analyze_pdf_expertise.py   # 分析PDF专家思路的脚本
├── clean_pdf_text.py          # 清洗PDF文本的脚本
├── cleaned_pdf_texts.json     # 清洗后的PDF文本数据
├── expert_features_analysis.json  # 专家特征分析结果
├── expertise_analysis.json    # 专家分析思路结果
├── extract_expert_features.py # 提取专家特征的脚本
├── extract_pdf_text.py        # 提取PDF文本的脚本
├── pdf_texts.json             # 原始PDF文本数据
├── process_expert_features.py # 整合处理流程的脚本
└── view_pdf_texts.py          # 查看PDF文本的脚本
```

## 3. 核心功能介绍

### 3.1 文本提取模块

#### `extract_pdf_text.py`
- **功能**：从指定目录下的所有PDF文件中提取文本内容
- **主要函数**：
  - `extract_pdf_text(pdf_path)` - 提取单个PDF文件的文本
  - `extract_all_pdfs(pdf_dir, output_file)` - 提取目录下所有PDF文件的文本并保存
- **使用方法**：
  ```python
  python extract_pdf_text.py
  ```

#### `view_pdf_texts.py`
- **功能**：查看提取的PDF文本内容
- **主要函数**：
  - `view_pdf_texts(json_path, max_chars=2000)` - 查看PDF文本内容（可限制显示字符数）
- **使用方法**：
  ```python
  python view_pdf_texts.py
  ```

### 3.2 文本清洗模块

#### `clean_pdf_text.py`
- **功能**：清洗从PDF提取的原始文本，去除噪声并提高特征提取质量
- **主要类**：
  - `PDFTextCleaner` - 文本清洗类，包含多种清洗方法
- **主要功能**：
  - 移除页码标记、多余空白、多余标点符号
  - 文本标准化（繁转简）
  - 移除噪声信息
- **使用方法**：
  ```python
  python clean_pdf_text.py
  ```

### 3.3 特征提取与分析模块

#### `extract_expert_features.py`
- **功能**：从清洗后的文本中提取结构化的专家分析特征
- **主要类**：
  - `ExpertFeatureExtractor` - 专家特征提取器
- **提取的特征类型**：
  - 赔率相关特征（欧赔、盘口、水位等）
  - 球队表现相关特征（近期战绩、胜率、排名等）
  - 历史对阵相关特征（历史交锋、对战记录等）
  - 分析规则（条件规则、if-then规则等）
- **主要功能**：
  - 特征提取与权重计算
  - 生成机器学习特征建议
  - 保存分析结果
- **使用方法**：
  ```python
  python extract_expert_features.py
  ```

#### `analyze_pdf_expertise.py`
- **功能**：分析PDF文本内容，提取专家的分析思路和方法
- **主要函数**：
  - `analyze_pdf_expertise(json_path)` - 分析PDF文本，提取专家分析思路
- **分析内容**：
  - 关键因素
  - 分析方法
  - 赔率分析相关内容
  - 球队表现分析相关内容
- **使用方法**：
  ```python
  python analyze_pdf_expertise.py
  ```

### 3.4 整合处理模块

#### `process_expert_features.py`
- **功能**：整合的专家特征处理脚本，包含完整流程
- **主要类**：
  - `ExpertFeatureProcessor` - 专家特征处理完整流程类
- **完整流程**：
  1. 从PDF文件中提取文本
  2. 清洗提取的文本
  3. 从清洗后的文本中提取专家特征
  4. 验证处理结果
- **主要功能**：
  - 支持跳过已完成的步骤
  - 结果验证
  - 命令行参数支持
- **使用方法**：
  ```python
  # 运行完整流程
  python process_expert_features.py
  
  # 跳过文本提取（如果已有提取好的文本）
  python process_expert_features.py --skip-text-extraction
  
  # 跳过文本清洗（如果已有清洗好的文本）
  python process_expert_features.py --skip-text-cleaning
  
  # 仅验证结果
  python process_expert_features.py --verify-only
  ```

## 4. 数据文件说明

### 4.1 `pdf_texts.json`
- **内容**：从PDF文件中提取的原始文本数据
- **格式**：JSON对象，键为PDF文件名，值为提取的文本内容

### 4.2 `cleaned_pdf_texts.json`
- **内容**：清洗后的PDF文本数据
- **格式**：JSON对象，键为PDF文件名，值为清洗后的文本内容

### 4.3 `expert_features_analysis.json`
- **内容**：专家特征分析结果
- **主要字段**：
  - `expert_features` - 提取的专家特征
    - `odds_features` - 赔率相关特征
    - `team_performance_features` - 球队表现相关特征
    - `head_to_head_features` - 历史对阵相关特征
    - `analysis_rules` - 分析规则
  - `feature_suggestions` - 机器学习特征建议

### 4.4 `expertise_analysis.json`
- **内容**：专家分析思路结果
- **主要字段**：
  - `analyzed_files` - 分析的PDF文件列表
  - `key_factors` - 关键因素列表
  - `analysis_methods` - 分析方法列表
  - `odds_analysis` - 赔率分析相关内容
  - `team_performance_analysis` - 球队表现分析相关内容

## 5. 使用流程

### 5.1 完整流程

1. **提取PDF文本**：
   ```python
   python extract_pdf_text.py
   ```

2. **清洗文本**：
   ```python
   python clean_pdf_text.py
   ```

3. **提取专家特征**：
   ```python
   python extract_expert_features.py
   ```

4. **（可选）分析专家思路**：
   ```python
   python analyze_pdf_expertise.py
   ```

### 5.2 使用整合脚本（推荐）

```python
# 运行完整流程
python process_expert_features.py

# 或者根据需要跳过某些步骤
python process_expert_features.py --skip-text-extraction --skip-text-cleaning
```

## 6. 机器学习特征建议

根据专家分析，生成了以下几类机器学习特征建议：

### 6.1 赔率特征
- 胜赔、平赔、负赔的初盘值和终盘值
- 赔率变化率（初盘到终盘的变化百分比）
- 赔率组合的合理性
- 主客场赔率差异
- 水位变化率
- 欧赔组合模式识别

### 6.2 球队表现特征
- 近5场、近10场比赛的胜率、平率、负率
- 近5场、近10场比赛的进球数和失球数
- 主场和客场的胜率、进球数、失球数
- 本赛季的积分、排名、进球数、失球数
- 近期状态的变化趋势
- 主客场表现差异

### 6.3 历史对阵特征
- 近5次、近10次对阵的胜负平记录
- 近5次、近10次对阵的进球数和失球数
- 主场和客场对阵的历史战绩
- 上次对阵的结果和赔率
- 历史对阵中的胜率、平率、负率

### 6.4 组合特征
- 球队表现与赔率的匹配度
- 历史对阵结果与当前赔率的一致性
- 主客场因素与赔率的交互作用
- 近期状态与赔率变化的关系
- 专家规则匹配度

## 7. 技术特点

1. **模块化设计**：各功能模块独立，便于维护和扩展
2. **文本清洗技术**：去除噪声、标准化文本，提高特征提取质量
3. **正则表达式优化**：精确匹配不同类型的专家特征
4. **特征权重机制**：基于出现频率计算特征权重
5. **完整流程整合**：提供一键式完整处理流程
6. **结果验证**：验证处理结果的完整性和正确性

## 8. 注意事项

1. 确保PDF文件位于指定目录（默认：`/Users/Williamhiler/Documents/my-project/train/pdf/`）
2. 文本提取和清洗可能需要一定时间，取决于PDF文件的数量和大小
3. 特征提取结果保存在JSON文件中，可直接用于机器学习模型训练
4. 可根据实际需要调整正则表达式模式以提高特征提取准确性
5. 运行整合脚本时，可根据已有文件情况跳过某些步骤以提高效率

## 9. 扩展建议

1. 引入自然语言处理（NLP）技术，提高语义理解能力
2. 增加更多特征类型和模式匹配规则
3. 实现特征可视化功能，便于直观分析专家思路
4. 支持更多PDF格式和布局的适应性处理
5. 与机器学习模型直接集成，实现端到端的专家知识融合