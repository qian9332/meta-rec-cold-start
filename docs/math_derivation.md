# 数学推导：MAML在冷启动推荐中的形式化

## 1. 冷启动问题的元学习形式化

### 问题定义

设推荐系统有 $N$ 个用户，每个用户 $u$ 的交互数据为 $\mathcal{D}_u = \{(x_i, y_i)\}_{i=1}^{n_u}$，
其中 $x_i = (\text{user\_id}, \text{item\_id})$，$y_i \in \{0, 1\}$ 为交互标签。

**冷启动用户**：$n_u \leq K$（交互数极少）

**目标**：学习初始参数 $\theta^*$，使得对任意冷启动用户 $u$，
只需在其少量数据 $\mathcal{D}_u^{support}$ 上做几步梯度更新，
即可在 $\mathcal{D}_u^{query}$ 上获得良好推荐效果。

### 元学习任务构建

每个用户的数据构成一个**任务** $\mathcal{T}_u$：
- **Support Set**: $\mathcal{S}_u = \{(x_i, y_i)\}_{i=1}^K$（前K条交互）
- **Query Set**: $\mathcal{Q}_u = \{(x_j, y_j)\}_{j=K+1}^{K+Q}$（后续交互）

任务分布：$p(\mathcal{T}) = \text{Uniform}(\{\mathcal{T}_u\}_{u=1}^N)$

---

## 2. MAML两级优化

### 内循环（任务适配）

对于任务 $\mathcal{T}_u$，在 support set 上做 $M$ 步梯度下降：

$$\theta_u^{(0)} = \theta$$

$$\theta_u^{(m+1)} = \theta_u^{(m)} - \alpha \nabla_{\theta_u^{(m)}} \mathcal{L}_{support}(f_{\theta_u^{(m)}}, \mathcal{S}_u)$$

简化为一步更新：

$$\theta'_u = \theta - \alpha \nabla_\theta \mathcal{L}_{support}(f_\theta, \mathcal{S}_u)$$

### 外循环（元优化）

$$\theta \leftarrow \theta - \beta \nabla_\theta \sum_{u \sim p(\mathcal{T})} \mathcal{L}_{query}(f_{\theta'_u}, \mathcal{Q}_u)$$

---

## 3. 二阶梯度更新公式推导

### 外循环梯度的展开

关键：$\theta'_u$ 是 $\theta$ 的函数，需要用链式法则：

$$\nabla_\theta \mathcal{L}_{query}(f_{\theta'_u}) = \nabla_{\theta'_u} \mathcal{L}_{query} \cdot \frac{\partial \theta'_u}{\partial \theta}$$

计算 $\frac{\partial \theta'_u}{\partial \theta}$：

$$\theta'_u = \theta - \alpha \nabla_\theta \mathcal{L}_{support}(f_\theta)$$

$$\frac{\partial \theta'_u}{\partial \theta} = I - \alpha \nabla^2_\theta \mathcal{L}_{support}(f_\theta) = I - \alpha H_{support}$$

其中 $H_{support} = \nabla^2_\theta \mathcal{L}_{support}$ 是 **Hessian矩阵**。

### 完整二阶梯度

$$\nabla_\theta \mathcal{L}_{query}(f_{\theta'_u}) = \nabla_{\theta'_u} \mathcal{L}_{query} \cdot (I - \alpha H_{support})$$

$$= \underbrace{\nabla_{\theta'_u} \mathcal{L}_{query}}_{\text{一阶项 (FOMAML)}} - \alpha \underbrace{\nabla_{\theta'_u} \mathcal{L}_{query} \cdot H_{support}}_{\text{二阶项 (Hessian-vector product)}}$$

---

## 4. Hessian计算瓶颈分析

### 推荐模型的参数分块

设模型参数 $\theta = (\theta_{emb}, \theta_{dense})$：
- $\theta_{emb}$: Embedding层参数（用户+物品），维度 $(N_u + N_i) \times d$
- $\theta_{dense}$: MLP层参数，维度远小于Embedding

### Hessian的块结构

$$H = \begin{pmatrix} H_{emb,emb} & H_{emb,dense} \\ H_{dense,emb} & H_{dense,dense} \end{pmatrix}$$

#### 关键观察：块对角稀疏性

1. **$H_{emb,emb}$ 极度稀疏**：
   - Embedding查找操作 $e = W[id]$ 只涉及一行
   - 每次前向只有 batch 中的 $B$ 个ID有梯度
   - $H_{emb,emb}$ 中仅 $O(B \times d)$ 个元素非零
   - 总参数 $(N_u + N_i) \times d$ 中 $B << N_u + N_i$

2. **$H_{emb,dense}$ 近似为零**：
   - Embedding层和Dense层的参数交互通过前向传播间接关联
   - 对于大部分未被查询的Embedding行，交叉二阶导为0
   - 实验验证此项的Frobenius范数远小于对角块

3. **$H_{dense,dense}$ 密集但小**：
   - Dense层参数量远小于Embedding层
   - 这一块的计算量可接受

### 计算复杂度

| | 存储 | 计算 |
|---|---|---|
| Full Hessian | $O(|\theta|^2)$ | $O(|\theta|^3)$ |
| Hessian-vector product | $O(|\theta|)$ | $O(|\theta|)$ per product |
| FOMAML (无Hessian) | $O(|\theta|)$ | $O(|\theta|)$ |

对于推荐模型，$|\theta_{emb}| >> |\theta_{dense}|$（通常100x以上），
且 $H_{emb,emb}$ 的有效非零率 $< 0.1\%$。

**结论：Hessian的有效信息量远小于参数空间的二次方，忽略二阶项的代价很小。**

---

## 5. FOMAML一阶近似

### 近似公式

$$\nabla_\theta \mathcal{L}_{query}(f_{\theta'_u}) \approx \nabla_{\theta'_u} \mathcal{L}_{query}(f_{\theta'_u})$$

即忽略了 $-\alpha \nabla_{\theta'_u} \mathcal{L}_{query} \cdot H_{support}$ 项。

### 工程ROI分析

| 指标 | MAML (Full) | FOMAML |
|------|-------------|--------|
| AUC | 基准 | 基准 - 0.3~0.5pp |
| 训练时间 | 基准 | 基准 × 0.35~0.4 |
| 内存占用 | 基准 | 基准 × 0.5 |
| 实现复杂度 | 高（需要create_graph=True） | 低 |

**结论：AUC损失 < 0.5pp，训练成本降低 60%+，FOMAML是工程最优选择。**

---

## 6. Meta-Embedding梯度消失缓解

### 问题分析

在MAML内循环中，Embedding层面临的梯度消失问题：

$$\nabla_{\theta_{emb}} \mathcal{L} = \frac{\partial \mathcal{L}}{\partial e} \cdot \frac{\partial e}{\partial \theta_{emb}}$$

其中 $\frac{\partial e}{\partial \theta_{emb}}$ 是一个选择矩阵（one-hot），
只有被查询ID的那一行有梯度。

**对于低频ID**：在meta-train的多个任务中很少被采样到，
Embedding几乎没有得到有效更新 => 冷启动时适配效果差。

### 解决方案A：学习率解耦（核心）

内循环更新改为：

$$\theta'_{emb} = \theta_{emb} - \alpha_{emb} \nabla_{\theta_{emb}} \mathcal{L}$$

$$\theta'_{dense} = \theta_{dense} - \alpha_{dense} \nabla_{\theta_{dense}} \mathcal{L}$$

其中 $\alpha_{emb} > \alpha_{dense}$（例如 $\alpha_{emb} = 0.02$, $\alpha_{dense} = 0.005$）

**原理**：Embedding梯度稀疏 => 需要更大步长补偿更新不足

### 解决方案B：低频ID梯度补偿（辅助）

对于频率低于阈值 $\tau$ 的ID，梯度乘以补偿系数：

$$g'_{emb}[id] = \begin{cases} \gamma \cdot g_{emb}[id] & \text{if } \text{freq}(id) < \tau \\ g_{emb}[id] & \text{otherwise} \end{cases}$$

其中 $\gamma = s \cdot \frac{\tau}{\max(\text{freq}(id), 1)}$，$s$ 为基础补偿系数。

### 效果

冷启动AUC提升约 **1~2pp**。

---

## 7. Reptile + ANIL 分层架构

### Reptile预训练backbone

Reptile更新规则：

$$\theta \leftarrow \theta + \epsilon (\theta' - \theta)$$

其中 $\theta' = \text{SGD}(\theta, \mathcal{T}, k \text{ steps})$

Reptile等价于优化：

$$\min_\theta \mathbb{E}_{\mathcal{T}} \left[ \mathcal{L}(\theta') + \frac{1}{2\alpha k} \|\theta' - \theta\|^2 \right]$$

即：在任务适配后的性能 + 参数不要偏离初始值太远（隐式正则化）

### ANIL在线适配head

基于MAML成功来自**特征复用**的发现：
- Backbone（特征提取）在元训练中学到的通用表示已经很好
- 只需适配最后一层（head）即可适配新任务

$$\theta_{head}' = \theta_{head} - \alpha \nabla_{\theta_{head}} \mathcal{L}_{support}(f_{\theta_{backbone}, \theta_{head}})$$

Backbone冻结：$\theta_{backbone}$ 不变

**优势**：适配参数量从 ~800K 降至 ~100，在线推理速度提升100x+。
