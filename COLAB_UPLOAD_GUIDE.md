# 🚀 V5足球预测模型 - Colab上传指南

## 📦 项目文件说明

### 已创建的文件

1. **Colab训练笔记本**
   - 文件名：`train_v5_colab.ipynb`
   - 路径：`v5/train_v5_colab.ipynb`
   - 说明：完整的Colab训练环境，包含10个步骤

2. **训练脚本**
   - 文件名：`train_two_stage.py`
   - 路径：`v5/trainers/train_two_stage.py`
   - 说明：两阶段训练脚本（阶段1：提取专家知识，阶段2：训练模型）

3. **专家知识提取器**
   - 文件名：`expert_knowledge_extractor.py`
   - 路径：`v5/utils/expert_knowledge/expert_knowledge_extractor.py`
   - 说明：从专家PDF中提取结构化规则

4. **智能规则匹配器**
   - 文件名：`intelligent_rule_matcher.py`
   - 路径：`v5/utils/expert_knowledge/intelligent_rule_matcher.py`
   - 说明：利用Qwen语义理解能力匹配专家规则

5. **数据加载器**
   - 文件名：`data_loader.py`
   - 路径：`v5/utils/data_processing/data_loader.py`
   - 说明：加载训练数据并注入专家知识

---

## 📤 上传到Colab的步骤

### 方法1：直接上传（推荐，快速）

1. **打开Google Colab**
   - 访问：https://colab.research.google.com/

2. **创建新笔记本**
   - 点击"新建笔记本"
   - 命名为：`V5训练`
   - 选择"代码"模式

3. **上传文件**
   - 点击左侧文件面板的"上传文件"图标
   - 依次上传以下文件：
     - `train_v5_colab.ipynb`（主要笔记本）
     - `v5_for_colab.zip`（项目文件，如果已打包）

4. **配置GPU**
   - 点击右上角"运行时"
   - 更改为"GPU"（T4免费）
   - 点击"保存"

5. **运行训练**
   - 点击菜单："运行时" -> "全部运行"
   - 或者逐个点击每个单元格的"运行"按钮

### 方法2：打包后上传（如果zip文件不存在）

1. **打包项目**
   ```bash
   cd /Users/Williamhiler/Documents/my-project/train
   zip -r v5_for_colab.zip v5/
   ```

2. **上传到Colab**
   - 在Colab左侧文件面板
   - 点击"上传文件"
   - 选择`v5_for_colab.zip`
   - 等待上传完成

3. **解压文件**
   ```python
   !unzip v5_for_colab.zip -d /content/
   %cd v5
   ```

4. **打开笔记本**
   - 在Colab中打开`train_v5_colab.ipynb`
   - 运行所有单元格

---

## 🎮 Colab操作提示

### GPU配置
- **推荐**：T4 GPU（免费，16GB显存）
- **设置位置**：右上角"运行时" -> "硬件加速器" -> "GPU"
- **验证**：运行第一个单元格（GPU检测）查看是否成功

### 文件路径
- **项目根目录**：`/content/v5/`
- **配置文件**：`/content/v5/configs/v5_config.yaml`
- **数据目录**：`/content/v5/data/`
- **检查点目录**：`/content/v5/checkpoints/`
- **日志目录**：`/content/v5/logs/`

### 训练时间预估
- **阶段1（提取规则）**：1-3分钟（使用T4 GPU）
- **阶段2（训练模型）**：1-3小时（取决于epoch数）
- **总时间**：1.5-3.5小时

---

## 📊 训练流程

### 阶段1：提取专家知识
1. 安装依赖
2. 下载Qwen模型
3. 加载专家数据
4. 提取专家规则
5. 保存到`data/expert_knowledge/expert_rules.json`

### 阶段2：训练模型
1. 准备训练数据
2. 初始化智能规则匹配器
3. 为每场比赛匹配专家规则
4. 训练V5融合模型
5. 保存最佳模型和检查点

---

## 💡 重要提示

### 1. Colab会话限制
- **免费GPU**：每天最多12小时
- **会话超时**：12小时后自动断开
- **建议**：定期保存检查点到Google Drive

### 2. 文件保存
- **Colab运行时**：文件保存在`/content/`目录
- **会话结束后**：文件会被删除
- **建议**：
  - 定期下载重要文件
  - 或保存到Google Drive
  - 或使用GitHub管理代码

### 3. 训练监控
- **查看日志**：`!tail -f logs/v5_training.log`
- **查看进度**：查看每个单元格的输出
- **TensorBoard**：`%load_ext tensorboard` 然后 `%tensorboard --logdir logs/`

### 4. 断点续传
- **自动恢复**：如果训练中断，重新运行阶段2即可
- **检查点文件**：`checkpoints/checkpoint_epoch_*.pt`
- **最佳模型**：`checkpoints/best_model.pt`

---

## 🎯 快速开始

### 最简单的方式（推荐）

1. **打开Colab**：https://colab.research.google.com/

2. **上传笔记本**
   - 上传`train_v5_colab.ipynb`

3. **配置GPU**
   - 运行时 -> GPU（T4）

4. **运行所有单元格**
   - 点击"运行时" -> "全部运行"

5. **等待完成**
   - 阶段1：1-3分钟
   - 阶段2：1-3小时

---

## 📝 训练完成后

### 查看结果
1. **最佳模型**：`checkpoints/best_model.pt`
2. **训练日志**：`logs/v5_training.log`
3. **专家规则**：`data/expert_knowledge/expert_rules.json`

### 下载模型
1. **在Colab中**
   ```python
   from google.colab import files
   files.download('checkpoints/best_model.pt')
   ```

2. **保存到Drive**
   ```python
   from google.colab import drive
   import shutil
   shutil.copytree('checkpoints', '/content/drive/MyDrive/v5_checkpoints')
   ```

---

## 🚀 立即开始

**您现在可以：**

1. ✅ **打开Colab**：https://colab.research.google.com/
2. ✅ **上传笔记本**：`train_v5_colab.ipynb`
3. ✅ **配置GPU**：运行时 -> GPU（T4）
4. ✅ **运行训练**：点击"全部运行"

**预计时间**：1.5-3.5小时

**完成后**：
- ✅ 专家规则已提取
- ✅ 模型已训练
- ✅ 可以进行预测

---

## 📞 需要帮助？

如果遇到问题，请检查：
1. GPU是否正确配置
2. 文件是否正确上传
3. Colab会话是否超时
4. 训练日志中的错误信息

**祝训练顺利！** 🎉
