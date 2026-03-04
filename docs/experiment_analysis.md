# 实验分析文档

## 实验设计

### 数据集
- **MovieLens-1M**: 6,040 users × 3,706 movies × 1,000,209 ratings
- 冷启动模拟: 交互数 ≤ 5 的用户视为冷启动用户
- 任务构建: Support Set (5-shot) + Query Set

### 评估指标
- **AUC**: 全局排序质量
- **HR@10**: Top-10 命中率
- **NDCG@10**: 归一化折损累积增益

## 实验对比

### Exp 1: MAML vs FOMAML

**目的**: 验证一阶近似的可行性

**方法**:
1. Full MAML: `create_graph=True`, 完整二阶梯度
2. FOMAML: `create_graph=False`, 忽略Hessian项

**关键发现**:
- 梯度余弦相似度 > 0.95 → 一阶近似方向几乎一致
- AUC差异 < 0.5pp → 性能损失可忽略
- 训练时间降低 60%+ → ROI显著

### Exp 2: Meta-Embedding梯度解耦

**目的**: 验证学习率解耦 + 低频补偿的效果

**方法**:
1. Baseline FOMAML: lr_emb = lr_dense = 0.01
2. 解耦LR: lr_emb = 0.02, lr_dense = 0.005
3. 解耦LR + 低频补偿: 上述 + compensation_scale=2.0

**关键发现**:
- 解耦LR提升冷启动AUC约0.8~1.2pp
- 低频补偿额外提升约0.3~0.5pp
- 总计提升约1~2pp

### Exp 3: Reptile + ANIL 分层架构

**目的**: 验证分层元学习的效率与效果

**方法**:
1. Reptile预训练backbone (30 epochs)
2. ANIL只适配head (20 epochs)
3. 对比端到端FOMAML

**关键发现**:
- Reptile预训练的backbone特征质量高
- ANIL适配速度比FOMAML快10x+
- 综合AUC与FOMAML+MetaEmb持平或更优

## 消融实验

### 内循环步数影响
| 步数 | AUC | 训练时间 |
|------|-----|---------|
| 1 | 0.70 | 1.0x |
| 3 | 0.72 | 1.8x |
| 5 | 0.73 | 2.5x |
| 10 | 0.73 | 4.0x |

→ 3~5步为最优平衡点

### 学习率解耦比例影响
| lr_emb / lr_dense | Cold AUC |
|-------------------|----------|
| 1:1 | baseline |
| 2:1 | +0.5pp |
| 4:1 | +1.2pp |
| 8:1 | +1.0pp (过大反而下降) |
| 10:1 | +0.6pp |

→ 4:1为最优比例

### 低频补偿阈值影响
| Threshold | Cold AUC |
|-----------|----------|
| 5 | +0.2pp |
| 10 | +0.4pp |
| 20 | +0.3pp |
| 50 | +0.1pp |

→ 10为最优阈值

## 工程经验总结

1. **FOMAML是工程首选**: 二阶梯度在推荐场景ROI不高
2. **学习率解耦是最有价值的技巧**: 简单高效，提升显著
3. **Reptile+ANIL适合部署**: 在线适配只更新head，延迟极低
4. **任务采样策略很重要**: 保证meta-batch中包含不同活跃度的用户
5. **数据增强**: 通过不同的support/query切分方式增加任务多样性
