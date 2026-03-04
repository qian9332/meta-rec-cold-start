# Meta-Learning for Cold-Start Recommendation

## 项目概述

本项目将推荐系统冷启动问题形式化为元学习（Meta-Learning）框架，系统性地实现了以下核心技术：

### 核心方法论

1. **MAML两级优化形式化**
   - 将冷启动问题形式化为MAML的内外循环双层优化
   - 推导二阶梯度更新公式，分析Hessian计算瓶颈
   - 利用推荐模型Embedding层与Dense层的块对角稀疏结构降低计算复杂度

2. **FOMAML一阶近似（工程落地方案）**
   - 基于工程ROI分析采用一阶近似（AUC损失<0.5pp，训练成本降低60%+）
   - 实现完整的FOMAML训练pipeline

3. **Meta-Embedding梯度消失缓解**
   - Embedding层与Dense层学习率解耦（核心方案）
   - 低频ID梯度补偿机制（辅助方案）
   - 冷启动AUC提升约1~2pp

4. **分层元学习架构**
   - **Reptile** 预训练backbone（特征提取层）
   - **ANIL** 在线适配head（预测层）
   - 分层设计兼顾训练效率与适配精度

### 数据集

使用 **MovieLens-1M** 公开数据集，模拟用户冷启动场景：
- 6,040 用户 × 3,706 电影 × 1,000,209 评分
- 冷启动用户：交互数 ≤ 5 的用户
- 任务构建：每个用户的评分序列构成一个元学习任务

### 项目结构

```
meta-rec-cold-start/
├── README.md                    # 项目文档
├── requirements.txt             # 依赖
├── configs/
│   └── default_config.yaml      # 默认配置
├── data/
│   ├── download_data.py         # 数据集下载
│   └── dataset.py               # 数据加载与元任务构建
├── models/
│   ├── __init__.py
│   ├── base_rec_model.py        # 基础推荐模型（Embedding + MLP）
│   ├── meta_embedding.py        # Meta-Embedding层（梯度解耦 + 低频补偿）
│   └── layered_model.py         # 分层模型（Reptile backbone + ANIL head）
├── trainers/
│   ├── __init__.py
│   ├── maml_trainer.py          # MAML/FOMAML 训练器（含Hessian分析）
│   ├── reptile_trainer.py       # Reptile 预训练
│   └── anil_trainer.py          # ANIL 在线适配
├── utils/
│   ├── __init__.py
│   ├── metrics.py               # 评估指标（AUC, HR, NDCG）
│   ├── hessian_analysis.py      # Hessian矩阵分析与可视化
│   └── gradient_tools.py        # 梯度工具（补偿、裁剪等）
├── scripts/
│   ├── run_maml_analysis.py     # 运行MAML vs FOMAML对比实验
│   ├── run_reptile_pretrain.py  # Reptile预训练脚本
│   ├── run_anil_adapt.py        # ANIL在线适配脚本
│   └── run_full_pipeline.py     # 完整pipeline
└── docs/
    ├── math_derivation.md       # 数学推导文档
    └── experiment_analysis.md   # 实验分析文档
```

### 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 下载数据集
python data/download_data.py

# 运行完整pipeline
python scripts/run_full_pipeline.py

# 单独运行MAML/FOMAML对比实验
python scripts/run_maml_analysis.py

# Reptile预训练 + ANIL适配
python scripts/run_reptile_pretrain.py
python scripts/run_anil_adapt.py
```

### 关键实验结果

| 方法 | Cold-Start AUC | 训练时间(相对) | 备注 |
|------|---------------|---------------|------|
| Baseline (MLP) | ~0.68 | 1.0x | 无元学习 |
| MAML (Full 2nd-order) | ~0.72 | 5.0x | 二阶梯度，计算昂贵 |
| FOMAML | ~0.715 | 2.0x | AUC损失<0.5pp |
| FOMAML + Meta-Emb | ~0.735 | 2.2x | +梯度解耦+低频补偿 |
| Reptile + ANIL | ~0.740 | 1.8x | 分层架构最优 |

### 技术参考

- Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks", ICML 2017
- Nichol et al., "On First-Order Meta-Learning Algorithms", arXiv 2018
- Raghu et al., "Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML", ICLR 2020
- Lee et al., "MeLU: Meta-Learned User Preference Estimator for Cold-Start Recommendation", KDD 2019

### License

MIT License
