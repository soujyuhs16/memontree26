# 基于预训练模型的用户评论多标签智能审核系统

本项目是一个毕业设计项目，实现了基于Hugging Face Transformers的中文评论多标签分类审核系统。

## 项目简介

该系统能够自动识别用户评论中的多种违规内容，包括：
- 🔴 **色情内容** (porn)
- 🔴 **辱骂内容** (abuse)  
- 🔴 **地域歧视** (region)
- 🔴 **性别歧视** (gender)
- 🔴 **种族歧视** (race)
- 🔴 **职业歧视** (occupation)

系统结合了深度学习模型预测和基于规则的检测，提供准确的多标签分类审核能力。

## 功能特点

✅ 多标签分类：支持同时检测多种违规类型  
✅ 规则融合：结合正则表达式规则提升检测准确率  
✅ 交互界面：提供Streamlit可视化审核界面  
✅ 批量处理：支持CSV文件批量审核  
✅ 审计日志：记录所有预测结果便于追溯  
✅ 阈值调节：支持动态调整各标签判定阈值

## 目录结构

```
memontree26/
├── Dataset/                      # 原始数据集目录
│   ├── AbuseSet/
│   │   └── AbuseSet.csv
│   ├── SexHarmset/
│   │   └── SexHarmSet.csv
│   └── BiasSet/
│       ├── Bias_region.csv
│       ├── BiasSet_genden.csv
│       ├── Bias_race.csv
│       └── Bias_occupation.csv
├── data/                         # 生成的数据集
│   └── all_multilabel.csv        # 合并后的多标签数据集
├── outputs/                      # 训练输出
│   ├── best_model/               # 最佳模型
│   └── training_report.txt       # 训练报告
├── logs/                         # 日志目录
│   └── pred_log.csv              # 预测审计日志
├── build_multilabel_dataset.py  # 数据集构建脚本
├── train_multilabel.py          # 模型训练脚本
├── app.py                        # Streamlit演示应用
├── verify_system.py              # 系统验证脚本
├── example_batch.csv             # 批量预测示例文件
├── requirements.txt              # Python依赖
└── README.md                     # 本文件
```

## 快速开始

### 1. 环境准备

**系统要求：**
- Python 3.8+
- 推荐使用GPU（V100或更高）进行训练，CPU也可运行但较慢
- 需要互联网连接以下载预训练模型（首次运行训练时）

**安装依赖：**

```bash
pip install -r requirements.txt
```

主要依赖包括：
- torch: 深度学习框架
- transformers: Hugging Face模型库
- datasets: 数据集处理
- pandas: 数据处理
- streamlit: Web应用框架
- scikit-learn: 评估指标

**验证系统配置：**

在开始之前，建议运行系统验证脚本检查所有依赖和文件：

```bash
python verify_system.py
```

该脚本会检查：Python版本、依赖包、数据集文件、脚本文件、目录结构和GPU支持。

### 2. 数据准备

原始数据集应放置在 `Dataset/` 目录下，包含以下6个CSV文件：
- `Dataset/AbuseSet/AbuseSet.csv`
- `Dataset/SexHarmset/SexHarmSet.csv`
- `Dataset/BiasSet/Bias_region.csv`
- `Dataset/BiasSet/BiasSet_genden.csv`
- `Dataset/BiasSet/Bias_race.csv`
- `Dataset/BiasSet/Bias_occupation.csv`

**注意：** 如果数据集文件较大（如超过100MB），建议添加到 `.gitignore` 中，不提交到仓库。

### 3. 生成多标签数据集

运行数据集构建脚本，将6个原始CSV文件合并为一个多标签数据集：

```bash
python build_multilabel_dataset.py
```

该脚本会：
1. 读取所有原始CSV文件
2. 清洗数据（处理BOM、去除空值、过滤无效文本等）
3. 将每个数据集映射到对应的标签
4. 按文本分组并聚合
5. 生成 `data/all_multilabel.csv`

输出示例：
```
============================================================
数据集统计信息
============================================================
输出文件: data/all_multilabel.csv
总行数: 53509

各标签正样本数:
  porn: 4999 (9.34%)
  abuse: 5493 (10.27%)
  region: 4398 (8.22%)
  gender: 2800 (5.23%)
  race: 5005 (9.35%)
  occupation: 4100 (7.66%)

Null值统计:
  无null值
============================================================
```

### 4. 训练模型

运行训练脚本开始模型训练：

```bash
python train_multilabel.py
```

训练配置：
- **模型**: hfl/chinese-macbert-base (中文预训练模型)
- **最大长度**: 150 tokens
- **批次大小**: 16
- **训练轮数**: 3 epochs
- **学习率**: 2e-5
- **优化器**: AdamW with weight decay
- **混合精度**: FP16 (如果有GPU)

训练过程会：
1. 首次运行时自动从Hugging Face下载预训练模型（约400MB）
2. 加载并分割数据集（训练/验证/测试）
3. 进行多标签分类训练
4. 每个epoch后在验证集上评估
4. 保存最佳模型到 `outputs/best_model/`
5. 在测试集上进行最终评估
6. 生成训练报告 `outputs/training_report.txt`

评估指标：
- Micro F1: 所有标签的微平均F1
- Macro F1: 所有标签的宏平均F1
- Per-label F1: 每个标签的F1分数

**注意：** 完整训练可能需要30分钟到2小时，取决于硬件配置。

### 5. 启动演示系统

训练完成后，使用Streamlit启动Web演示界面：

```bash
streamlit run app.py
```

系统会自动在浏览器中打开（默认地址：http://localhost:8501）

## 使用演示系统

### 单条文本审核

1. 在左侧边栏调整各标签的判定阈值
2. 在"单条预测"标签页输入待审核的文本
3. 点击"开始审核"按钮
4. 查看审核结果，包括：
   - 规则匹配情况（URL、手机号、重复字符等）
   - 模型预测概率
   - 融合后的判定结果
   - 综合判定建议

所有单条预测会自动记录到 `logs/pred_log.csv` 用于审计。

### 批量文本审核

1. 准备包含 `text` 列的CSV文件（可参考 `example_batch.csv`）
2. 在"批量预测"标签页上传文件
3. 点击"开始批量审核"
4. 等待处理完成
5. 查看统计结果和预览
6. 下载完整的审核结果CSV文件

批量结果包含每条文本的：
- 各标签预测概率
- 各标签判定结果（YES/NO）
- 匹配的规则
- 是否存在违规

## 技术细节

### 数据处理

数据清洗流程：
1. 处理UTF-8 BOM编码
2. 删除空值和无效文本
3. 过滤"nan"字符串
4. 按文本分组，取标签最大值
5. 生成0/1二分类标签

### 模型架构

- **基础模型**: hfl/chinese-macbert-base
- **任务类型**: Multi-label classification
- **标签数量**: 6
- **损失函数**: BCEWithLogitsLoss (内置)
- **输出激活**: Sigmoid

### 规则融合

系统实现了以下规则检测：
1. **URL检测**: 识别http/https链接
2. **手机号检测**: 识别中国大陆手机号
3. **重复字符**: 检测灌水行为（5个以上相同字符）
4. **关键词匹配**: 检测色情、辱骂关键词

规则检测结果会提升对应标签的概率，增强判定准确性。

## 性能优化建议

1. **使用GPU**: 训练时使用GPU可显著加速（建议V100或更高）
2. **批次大小**: 根据显存调整batch_size（16-32）
3. **模型选择**: 可尝试更大的模型如chinese-roberta-wwm-ext
4. **数据增强**: 可以通过回译、同义词替换等方式扩充训练数据
5. **阈值调优**: 根据实际业务需求调整各标签的判定阈值

## 常见问题

**Q: 训练时内存不足怎么办？**  
A: 减小batch_size或使用梯度累积，也可以使用更小的模型。

**Q: 如何提高模型准确率？**  
A: 可以增加训练数据、调整超参数、使用更大的预训练模型、或进行数据增强。

**Q: 可以部署到生产环境吗？**  
A: 本项目是演示系统，生产部署需要考虑性能优化、并发处理、安全性等问题。

**Q: 支持其他语言吗？**  
A: 当前针对中文优化，如需支持其他语言，需要更换对应的预训练模型和数据集。

## 开发者

本项目为毕业设计项目，主要技术栈：
- Python 3.8+
- PyTorch
- Hugging Face Transformers
- Streamlit
- Pandas

## 许可证

本项目仅用于学习和研究目的。

## 更新日志

### 2024-02-16
- ✅ 完成数据集构建脚本
- ✅ 完成多标签训练脚本
- ✅ 完成Streamlit演示系统
- ✅ 完成项目文档
