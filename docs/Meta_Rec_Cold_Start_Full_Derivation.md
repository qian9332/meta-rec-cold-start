# Meta-Learning for Cold-Start Recommendation

## 完整数学推导与技术文档

> GitHub: [qian9332/meta-rec-cold-start](https://github.com/qian9332/meta-rec-cold-start)  
> 核心技术：MAML / FOMAML / Reptile / ANIL / Meta-Embedding

---

## 目录

- [一、问题定义与形式化](#一问题定义与形式化)
- [二、MAML两级优化推导](#二maml两级优化推导)
- [三、Hessian矩阵分析与计算瓶颈](#三hessian矩阵分析与计算瓶颈)
- [四、FOMAML一阶近似与工程ROI](#四fomaml一阶近似与工程roi)
- [五、Meta-Embedding梯度消失缓解](#五meta-embedding梯度消失缓解)
- [六、Reptile + ANIL 分层架构](#六reptile--anil-分层架构)
- [七、核心代码实现](#七核心代码实现)
- [八、实验结果汇总](#八实验结果汇总)
- [九、参考文献](#九参考文献)

---

## 一、问题定义与形式化

### 1.1 冷启动问题

推荐系统的冷启动问题是指：当新用户或新物品进入系统时，由于缺乏历史交互数据，传统协同过滤方法无法为其生成准确的推荐。

设推荐系统有 **N** 个用户，每个用户 **u** 的交互数据为：

$$\mathcal{D}_u = \{(x_i, y_i)\}_{i=1}^{n_u}$$

其中：
- $x_i = (\text{user\_id}, \text{item\_id})$ 表示用户-物品对
- $y_i \in \{0, 1\}$ 为交互标签（点击/未点击）

**冷启动用户定义：** 交互数 $n_u \leq K$ 的用户，其中 K 为阈值（如 K=5）。

---

### 1.2 元学习形式化

将冷启动问题形式化为元学习框架，核心思想是：**学习一个良好的初始化参数 θ\***，使得对任意冷启动用户 u，只需在其少量数据上做几步梯度更新，即可获得良好的推荐效果。

#### 元学习任务构建

每个用户的数据构成一个**任务** $\mathcal{T}_u$：

| 数据集 | 定义 | 说明 |
|--------|------|------|
| **Support Set** | $\mathcal{S}_u = \{(x_i, y_i)\}_{i=1}^K$ | 前K条交互，模拟少量已知行为 |
| **Query Set** | $\mathcal{Q}_u = \{(x_j, y_j)\}_{j=K+1}^{K+Q}$ | 后续交互，评估适配效果 |

**任务分布：**

$$p(\mathcal{T}) = \text{Uniform}(\{\mathcal{T}_u\}_{u=1}^N)$$

---

## 二、MAML两级优化推导

### 2.1 内循环（任务适配）

对于任务 $\mathcal{T}_u$，在 Support Set 上做 **M** 步梯度下降，从初始参数 θ 适配到任务特定参数 $\theta'_u$：

**初始化：**

$$\theta_u^{(0)} = \theta$$

**迭代更新：**

$$\theta_u^{(m+1)} = \theta_u^{(m)} - \alpha \nabla_{\theta_u^{(m)}} \mathcal{L}_{\text{support}}(f_{\theta_u^{(m)}}, \mathcal{S}_u)$$

其中 α 为内循环学习率。

**简化为一步更新（常用设置）：**

$$\theta'_u = \theta - \alpha \nabla_\theta \mathcal{L}_{\text{support}}(f_\theta, \mathcal{S}_u)$$

> **关键点：** 一步适配后的参数 $\theta'_u$ 是初始参数 θ 的函数。

---

### 2.2 外循环（元优化）

外循环的目标是优化初始参数 θ，使得在所有任务上适配后的平均损失最小：

**目标函数：**

$$\min_\theta \mathbb{E}_{\mathcal{T} \sim p(\mathcal{T})} \left[ \mathcal{L}_{\text{query}}(f_{\theta'_\mathcal{T}}, \mathcal{Q}_\mathcal{T}) \right]$$

**梯度更新：**

$$\theta \leftarrow \theta - \beta \nabla_\theta \sum_{u \sim p(\mathcal{T})} \mathcal{L}_{\text{query}}(f_{\theta'_u}, \mathcal{Q}_u)$$

其中 β 为外循环学习率。

> **关键挑战：** 如何计算 $\nabla_\theta \mathcal{L}_{\text{query}}(f_{\theta'_u})$？

---

### 2.3 二阶梯度完整推导

由于 $\theta'_u = \theta - \alpha\nabla_\theta \mathcal{L}_{\text{support}}$ 是 θ 的函数，需要使用**链式法则**：

**第一步：链式法则展开**

$$\nabla_\theta \mathcal{L}_{\text{query}}(f_{\theta'_u}) = \nabla_{\theta'} \mathcal{L}_{\text{query}} \cdot \frac{\partial \theta'}{\partial \theta}$$

**第二步：计算雅可比矩阵**

$$\theta'_u = \theta - \alpha \nabla_\theta \mathcal{L}_{\text{support}}(f_\theta)$$

$$\frac{\partial \theta'_u}{\partial \theta} = I - \alpha \nabla^2_\theta \mathcal{L}_{\text{support}}(f_\theta) = I - \alpha H_{\text{support}}$$

其中 $H_{\text{support}} = \nabla^2_\theta \mathcal{L}_{\text{support}}$ 是 **Hessian 矩阵**（二阶导数矩阵）。

**第三步：完整二阶梯度公式**

$$\boxed{\nabla_\theta \mathcal{L}_{\text{query}}(f_{\theta'_u}) = \nabla_{\theta'} \mathcal{L}_{\text{query}} \cdot (I - \alpha H_{\text{support}})}$$

**展开后：**

$$\nabla_\theta \mathcal{L}_{\text{query}}(f_{\theta'_u}) = \underbrace{\nabla_{\theta'} \mathcal{L}_{\text{query}}}_{\text{一阶项 (FOMAML)}} - \alpha \underbrace{\nabla_{\theta'} \mathcal{L}_{\text{query}} \cdot H_{\text{support}}}_{\text{二阶项 (Hessian-vector product)}}$$

| 项 | 说明 |
|----|------|
| **一阶项** | FOMAML 保留的部分 |
| **二阶项** | 涉及 Hessian 计算，计算代价高 |

---

## 三、Hessian矩阵分析与计算瓶颈

### 3.1 推荐模型的参数分块

推荐模型的参数 $\theta = (\theta_{\text{emb}}, \theta_{\text{dense}})$ 具有特殊的块结构：

| 参数层 | 说明 | 维度 |
|--------|------|------|
| $\theta_{\text{emb}}$ | Embedding层参数（用户+物品） | $(N_u + N_i) \times d$ |
| $\theta_{\text{dense}}$ | MLP层参数 | 远小于Embedding |

---

### 3.2 Hessian的块结构

$$H = \begin{pmatrix} H_{\text{emb,emb}} & H_{\text{emb,dense}} \\ H_{\text{dense,emb}} & H_{\text{dense,dense}} \end{pmatrix}$$

#### 关键观察：块对角稀疏性

**1. $H_{\text{emb,emb}}$ 极度稀疏**

Embedding 查找操作 $e = W[\text{id}]$ 只涉及一行，每次前向传播只有 batch 中的 B 个 ID 有梯度。

- 非零元素：$O(B \times d)$
- 总参数：$(N_u + N_i) \times d$
- 稀疏率：$B \ll N_u + N_i$

**2. $H_{\text{emb,dense}}$ 近似为零**

Embedding 层和 Dense 层的参数交互通过前向传播间接关联。对于大部分未被查询的 Embedding 行，交叉二阶导为 0。

**3. $H_{\text{dense,dense}}$ 密集但小**

Dense 层参数量远小于 Embedding 层，这一块的计算量可接受。

---

### 3.3 计算复杂度对比

| 方法 | 存储复杂度 | 计算复杂度 |
|------|-----------|-----------|
| Full Hessian | $O(\|\theta\|^2)$ | $O(\|\theta\|^3)$ |
| Hessian-vector product | $O(\|\theta\|)$ | $O(\|\theta\|)$ per product |
| **FOMAML (无Hessian)** | $O(\|\theta\|)$ | $O(\|\theta\|)$ |

> **结论：** 对于推荐模型，$\|\theta_{\text{emb}}\| \gg \|\theta_{\text{dense}}\|$（通常100x以上），且 $H_{\text{emb,emb}}$ 的有效非零率 < 0.1%。**Hessian 的有效信息量远小于参数空间的二次方，忽略二阶项的代价很小。**

---

## 四、FOMAML一阶近似与工程ROI

### 4.1 一阶近似公式

$$\nabla_\theta \mathcal{L}_{\text{query}}(f_{\theta'_u}) \approx \nabla_{\theta'} \mathcal{L}_{\text{query}}(f_{\theta'_u})$$

即忽略 $-\alpha \cdot \nabla_{\theta'} \mathcal{L}_{\text{query}} \cdot H_{\text{support}}$ 项。

**实现上：** 只需在计算内循环梯度时设置 `create_graph=False`，PyTorch 不会保留梯度计算图，从而避免二阶导数计算。

---

### 4.2 工程ROI分析

| 指标 | MAML (Full) | FOMAML |
|------|-------------|--------|
| **AUC** | 基准 | 基准 - 0.3~0.5pp |
| **训练时间** | 基准 | 基准 × 0.35~0.4 |
| **内存占用** | 基准 | 基准 × 0.5 |
| **实现复杂度** | 高 | 低 |

> **结论：** AUC 损失 < 0.5pp，训练成本降低 60%+，**FOMAML 是工程最优选择**。

---

## 五、Meta-Embedding梯度消失缓解

### 5.1 问题分析

在 MAML 内循环中，Embedding 层面临梯度消失问题：

$$\nabla_{\theta_{\text{emb}}} \mathcal{L} = \frac{\partial \mathcal{L}}{\partial e} \cdot \frac{\partial e}{\partial \theta_{\text{emb}}}$$

其中 $\frac{\partial e}{\partial \theta_{\text{emb}}}$ 是选择矩阵（one-hot），**只有被查询 ID 的那一行有梯度**。

**问题：** 对于低频 ID，在 meta-train 的多个任务中很少被采样，Embedding 几乎得不到有效更新 → 冷启动时适配效果差。

---

### 5.2 解决方案A：学习率解耦（核心）

内循环更新改为对不同参数层使用不同学习率：

**Embedding 层（较大学习率）：**

$$\theta'_{\text{emb}} = \theta_{\text{emb}} - \alpha_{\text{emb}} \cdot \nabla_{\theta_{\text{emb}}} \mathcal{L}_{\text{support}}$$

**Dense 层（较小学习率）：**

$$\theta'_{\text{dense}} = \theta_{\text{dense}} - \alpha_{\text{dense}} \cdot \nabla_{\theta_{\text{dense}}} \mathcal{L}_{\text{support}}$$

**典型设置：** $\alpha_{\text{emb}} = 0.02$, $\alpha_{\text{dense}} = 0.005$

> **原理：** Embedding 梯度天然稀疏，需要更大步长补偿更新不足。

---

### 5.3 解决方案B：低频ID梯度补偿（辅助）

对于频率低于阈值 τ 的 ID，梯度乘以补偿系数：

$$g'_{\text{emb}}[\text{id}] = \begin{cases} \gamma \cdot g_{\text{emb}}[\text{id}] & \text{if } \text{freq}(\text{id}) < \tau \\ g_{\text{emb}}[\text{id}] & \text{otherwise} \end{cases}$$

其中：

$$\gamma = s \cdot \frac{\tau}{\max(\text{freq}(\text{id}), 1)}$$

s 为基础补偿系数。

> **效果：** 冷启动 AUC 提升约 **1~2pp**

---

## 六、Reptile + ANIL 分层架构

### 6.1 Reptile预训练backbone

Reptile 是一种简化的一阶元学习算法：

**更新规则：**

$$\theta \leftarrow \theta + \epsilon(\theta' - \theta)$$

其中 $\theta' = \text{SGD}(\theta, \mathcal{T}, k \text{ steps})$，ε 为外循环学习率。

**等价优化目标：**

$$\min_\theta \mathbb{E}_{\mathcal{T}} \left[ \mathcal{L}(\theta') + \frac{1}{2\alpha k}\|\theta' - \theta\|^2 \right]$$

即在任务适配后的性能 + 参数不要偏离初始值太远（隐式正则化）之间权衡。

**Reptile优势：**

| 优势 | 说明 |
|------|------|
| 无需二阶梯度 | 计算高效 |
| 隐式正则化 | 防止过拟合 |
| 泛化性好 | 学到的表示具有更好的迁移性 |

---

### 6.2 ANIL在线适配head

基于 Raghu et al. 2020 的发现：**MAML的成功主要来自特征复用而非快速学习**。

**核心思想：** 内循环只更新 head（最后一层），冻结 backbone。

**适配公式：**

$$\theta'_{\text{head}} = \theta_{\text{head}} - \alpha \nabla_{\theta_{\text{head}}} \mathcal{L}_{\text{support}}(f_{\theta_{\text{backbone}}, \theta_{\text{head}}})$$

**Backbone 冻结：** $\theta_{\text{backbone}}$ 不变

---

### 6.3 分层架构优势

| 指标 | ANIL | MAML |
|------|------|------|
| 适配参数量 | ~100 | ~800K |
| 内循环速度 | 快 100x+ | 基准 |
| 效果 | 接近MAML | 基准 |

**推荐系统中效果接近MAML的原因：**

1. 推荐模型的特征提取层（Embedding + MLP低层）学习的是用户-物品的通用表示，具有良好的迁移性
2. Head层（预测层）参数量小（如 64×1=65），适配速度快
3. 用户偏好变化主要体现在预测层而非特征层

---

## 七、核心代码实现

### 7.1 内循环实现（MAMLTrainer）

```python
def inner_loop(self, support_users, support_items, support_labels):
    """内循环：在support set上适配参数
    
    θ' = θ - α ∇_θ L_support(f_θ)
    """
    # 获取当前参数的克隆
    params = {name: param.clone() 
               for name, param in self.model.named_parameters()}
    
    for step in range(self.inner_steps):
        # 计算 support loss
        loss = self.model.compute_loss(
            support_users, support_items, support_labels,
            params=params
        )
        
        # 计算梯度
        grads = torch.autograd.grad(
            loss, params.values(),
            create_graph=not self.first_order  # MAML需要保留计算图
        )
        
        # 解耦学习率更新
        if self.use_decoupled_lr:
            params = decoupled_inner_update(
                params, grads, 
                lr_emb=self.lr_emb,      # Embedding层大学习率
                lr_dense=self.lr_dense   # Dense层小学习率
            )
    
    return params
```

---

### 7.2 元学习任务数据集

```python
class MetaTaskDataset(Dataset):
    """元学习任务数据集
    
    每个任务对应一个用户的冷启动场景：
    - Support Set: 用户的前K条交互
    - Query Set: 用户的后续交互
    """
    
    def __getitem__(self, index):
        uid = self.task_users[index]
        interactions = self.user_interactions[uid]
        
        # Support: 前n_support条
        support_ints = interactions[:self.support_size]
        # Query: 之后的n_query条  
        query_ints = interactions[self.support_size:self.support_size + self.query_size]
        
        # 构建support set (包含正负样本)
        for it in support_ints:
            support_users.append(uid)
            support_items.append(it["item_idx"])
            support_labels.append(it["label"])
            # 负采样
            for neg_item in self._sample_negatives(uid, self.neg_ratio):
                support_items.append(neg_item)
                support_labels.append(0.0)
        
        return {
            "support_users": torch.LongTensor(support_users),
            "support_items": torch.LongTensor(support_items),
            "support_labels": torch.FloatTensor(support_labels),
            "query_users": torch.LongTensor(query_users),
            # ...
        }
```

---

### 7.3 ANIL适配实现

```python
def inner_loop_head_only(self, support_users, support_items, support_labels):
    """ANIL内循环：只适配head参数
    
    frozen backbone提取特征 → 可学习head做预测
    """
    # 获取head参数
    head_params = {}
    for name, param in self.model.named_parameters():
        if "head" in name:
            head_params[name] = param.clone().detach().requires_grad_(True)
    
    # 内循环: 只更新head
    for step in range(self.inner_steps):
        loss = self.model.compute_loss(
            support_users, support_items, support_labels,
            head_params=head_params
        )
        
        # 只对head参数求梯度
        grads = torch.autograd.grad(
            loss, head_params.values(),
            create_graph=False  # ANIL不需要二阶梯度
        )
        
        # 更新head参数
        new_head_params = {}
        for (name, param), grad in zip(head_params.items(), grads):
            new_head_params[name] = param - self.inner_lr * grad
        head_params = new_head_params
    
    return head_params
```

---

## 八、实验结果汇总

| 方法 | Cold-Start AUC | 训练时间(相对) | 备注 |
|------|---------------|---------------|------|
| Baseline (MLP) | ~0.68 | 1.0x | 无元学习 |
| MAML (Full 2nd-order) | ~0.72 | 5.0x | 二阶梯度，计算昂贵 |
| FOMAML | ~0.715 | 2.0x | AUC损失<0.5pp |
| FOMAML + Meta-Emb | ~0.735 | 2.2x | +梯度解耦+低频补偿 |
| **Reptile + ANIL** | **~0.740** | **1.8x** | **分层架构最优** |

---

## 九、参考文献

1. Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks", ICML 2017

2. Nichol et al., "On First-Order Meta-Learning Algorithms", arXiv 2018

3. Raghu et al., "Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML", ICLR 2020

4. Lee et al., "MeLU: Meta-Learned User Preference Estimator for Cold-Start Recommendation", KDD 2019

---

<p align="center">
  <a href="https://github.com/qian9332/meta-rec-cold-start">
    <b>GitHub: qian9332/meta-rec-cold-start</b>
  </a>
</p>
