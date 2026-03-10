**Meta-Learning for Cold-Start Recommendation**

元学习冷启动推荐系统

完整数学推导与技术文档

GitHub: qian9332/meta-rec-cold-start

核心技术：MAML / FOMAML / Reptile / ANIL / Meta-Embedding

一、问题定义与形式化

1.1 冷启动问题

推荐系统的冷启动问题是指：当新用户或新物品进入系统时，由于缺乏历史交互数据，传统协同过滤方法无法为其生成准确的推荐。设推荐系统有
N 个用户，每个用户 u 的交互数据为：

*D_u = {(x_i, y_i)}\_{i=1}\^{n_u}*

其中 x_i = (user_id, item_id) 表示用户-物品对，y_i ∈ {0, 1}
为交互标签（点击/未点击）。

**冷启动用户定义：**交互数 n_u ≤ K 的用户，其中 K 为阈值（如 K=5）。

1.2 元学习形式化

将冷启动问题形式化为元学习框架，核心思想是：学习一个良好的初始化参数
θ\*，使得对任意冷启动用户
u，只需在其少量数据上做几步梯度更新，即可获得良好的推荐效果。

**元学习任务构建：**

-   每个用户的数据构成一个任务 T_u

-   Support Set: S_u = {(x_i,
    y_i)}\_{i=1}\^K（前K条交互，模拟少量已知行为）

-   Query Set: Q_u = {(x_j,
    y_j)}\_{j=K+1}\^{K+Q}（后续交互，评估适配效果）

*p(T) = Uniform({T_u}\_{u=1}\^N)*

任务分布：从所有用户任务中均匀采样。

二、MAML两级优化推导

2.1 内循环（任务适配）

对于任务 T_u，在 Support Set 上做 M 步梯度下降，从初始参数 θ
适配到任务特定参数 θ\'\_u：

*θ_u\^{(0)} = θ*

初始化：任务特定参数从全局初始参数开始。

*θ_u\^{(m+1)} = θ_u\^{(m)} - α ∇\_{θ_u\^{(m)}}
L\_{support}(f\_{θ_u\^{(m)}}, S_u)*

迭代更新：每一步在 Support Set 上计算梯度并更新参数。其中 α
为内循环学习率。

简化为一步更新（常用设置）：

*θ\'\_u = θ - α ∇\_θ L\_{support}(f_θ, S_u)*

一步适配后的参数 θ\'\_u 是初始参数 θ 的函数。

2.2 外循环（元优化）

外循环的目标是优化初始参数 θ，使得在所有任务上适配后的平均损失最小：

*min_θ E\_{T\~p(T)} \[L\_{query}(f\_{θ\'\_T}, Q_T)\]*

目标函数：期望在 Query Set 上的损失最小化。

*θ ← θ - β ∇\_θ Σ\_{u\~p(T)} L\_{query}(f\_{θ\'\_u}, Q_u)*

梯度更新：β 为外循环学习率。关键挑战在于如何计算 ∇\_θ
L\_{query}(f\_{θ\'\_u})。

2.3 二阶梯度完整推导

外循环梯度的核心推导：由于 θ\'\_u = θ - α∇\_θ L\_{support} 是 θ
的函数，需要使用链式法则：

*∇\_θ L\_{query}(f\_{θ\'\_u}) = ∇\_{θ\'} L\_{query} · ∂θ\'/∂θ*

链式法则展开。

计算雅可比矩阵 ∂θ\'/∂θ：

*θ\'\_u = θ - α ∇\_θ L\_{support}(f_θ)*

回顾 θ\' 的表达式。

*∂θ\'\_u/∂θ = I - α ∇²_θ L\_{support}(f_θ) = I - α H\_{support}*

其中 H\_{support} = ∇²_θ L\_{support} 是 Hessian 矩阵（二阶导数矩阵）。

**完整二阶梯度公式：**

*∇\_θ L\_{query}(f\_{θ\'\_u}) = ∇\_{θ\'} L\_{query} · (I - α
H\_{support})*

展开后得到：

*= ∇\_{θ\'} L\_{query} - α · ∇\_{θ\'} L\_{query} · H\_{support}*

第一项是一阶项（FOMAML保留），第二项是二阶项（涉及 Hessian 计算）。

三、Hessian矩阵分析与计算瓶颈

3.1 推荐模型的参数分块

推荐模型的参数 θ = (θ\_{emb}, θ\_{dense}) 具有特殊的块结构：

-   θ\_{emb}: Embedding 层参数（用户 + 物品），维度 (N_u + N_i) × d

-   θ\_{dense}: MLP 层参数，维度远小于 Embedding

3.2 Hessian的块结构

*H = \[H\_{emb,emb} H\_{emb,dense}\]*

\[H\_{dense,emb} H\_{dense,dense}\]

**关键观察：块对角稀疏性**

**1. H\_{emb,emb} 极度稀疏：**

Embedding 查找操作 e = W\[id\] 只涉及一行，每次前向传播只有 batch 中的 B
个 ID 有梯度。H\_{emb,emb} 中仅 O(B × d) 个元素非零，而总参数为 (N_u +
N_i) × d，其中 B \<\< N_u + N_i。

**2. H\_{emb,dense} 近似为零：**

Embedding 层和 Dense
层的参数交互通过前向传播间接关联。对于大部分未被查询的 Embedding
行，交叉二阶导为 0。实验验证此项的 Frobenius 范数远小于对角块。

**3. H\_{dense,dense} 密集但小：**

Dense 层参数量远小于 Embedding 层，这一块的计算量可接受。

3.3 计算复杂度对比

  -----------------------------------------------------------------------
  **方法**                **存储复杂度**          **计算复杂度**
  ----------------------- ----------------------- -----------------------
  Full Hessian            O(\|θ\|²)               O(\|θ\|³)

  Hessian-vector product  O(\|θ\|)                O(\|θ\|) per product

  FOMAML (无Hessian)      O(\|θ\|)                O(\|θ\|)
  -----------------------------------------------------------------------

结论：对于推荐模型，\|θ\_{emb}\| \>\> \|θ\_{dense}\|（通常 100x
以上），且 H\_{emb,emb} 的有效非零率 \< 0.1%。Hessian
的有效信息量远小于参数空间的二次方，忽略二阶项的代价很小。

四、FOMAML一阶近似与工程ROI

4.1 一阶近似公式

*∇\_θ L\_{query}(f\_{θ\'\_u}) ≈ ∇\_{θ\'} L\_{query}(f\_{θ\'\_u})*

即忽略 -α · ∇\_{θ\'} L\_{query} · H\_{support} 项。

实现上，只需在计算内循环梯度时设置 create_graph=False，PyTorch
不会保留梯度计算图，从而避免二阶导数计算。

4.2 工程ROI分析

  -----------------------------------------------------------------------
  **指标**                **MAML (Full)**         **FOMAML**
  ----------------------- ----------------------- -----------------------
  AUC                     基准                    基准 - 0.3\~0.5pp

  训练时间                基准                    基准 × 0.35\~0.4

  内存占用                基准                    基准 × 0.5

  实现复杂度              高                      低
  -----------------------------------------------------------------------

**结论：AUC 损失 \< 0.5pp，训练成本降低 60%+，FOMAML 是工程最优选择。**

五、Meta-Embedding梯度消失缓解

5.1 问题分析

在 MAML 内循环中，Embedding 层面临梯度消失问题：

*∇\_{θ\_{emb}} L = ∂L/∂e · ∂e/∂θ\_{emb}*

其中 ∂e/∂θ\_{emb} 是选择矩阵（one-hot），只有被查询 ID 的那一行有梯度。

对于低频 ID：在 meta-train 的多个任务中很少被采样到，Embedding
几乎没有得到有效更新，导致冷启动时适配效果差。

5.2 解决方案A：学习率解耦（核心）

内循环更新改为对不同参数层使用不同学习率：

*θ\'\_{emb} = θ\_{emb} - α\_{emb} · ∇\_{θ\_{emb}} L\_{support}*

Embedding 层使用较大学习率。

*θ\'\_{dense} = θ\_{dense} - α\_{dense} · ∇\_{θ\_{dense}} L\_{support}*

Dense 层使用较小学习率。

典型设置：α\_{emb} = 0.02, α\_{dense} = 0.005。原理是 Embedding
梯度天然稀疏，需要更大步长补偿更新不足。

5.3 解决方案B：低频ID梯度补偿（辅助）

对于频率低于阈值 τ 的 ID，梯度乘以补偿系数：

*g\'\_{emb}\[id\] = γ · g\_{emb}\[id\] if freq(id) \< τ*

否则保持原梯度。

*γ = s · τ / max(freq(id), 1)*

其中 s 为基础补偿系数。

**效果：冷启动 AUC 提升约 1\~2pp。**

六、Reptile + ANIL 分层架构

6.1 Reptile预训练backbone

Reptile 是一种简化的一阶元学习算法，更新规则：

*θ ← θ + ε(θ\' - θ)*

其中 θ\' = SGD(θ, T, k steps)，ε 为外循环学习率。

Reptile 等价于优化：

*min_θ E_T \[L(θ\') + 1/(2αk) \|\|θ\' - θ\|\|²\]*

即在任务适配后的性能 + 参数不要偏离初始值太远（隐式正则化）之间权衡。

6.2 ANIL在线适配head

基于 MAML 成功来自特征复用的发现（Raghu et al. 2020）：

-   Backbone（特征提取层）在元训练中学到的通用表示已经很好

-   只需适配最后一层（head）即可适配新任务

*θ\'\_{head} = θ\_{head} - α ∇\_{θ\_{head}}
L\_{support}(f\_{θ\_{backbone}, θ\_{head}})*

Backbone 冻结：θ\_{backbone} 不变。

6.3 分层架构优势

  -----------------------------------------------------------------------
  **指标**                            **对比**
  ----------------------------------- -----------------------------------
  适配参数量                          ANIL: \~100 vs MAML: \~800K

  内循环速度                          ANIL 快 100x+

  效果                                在特征复用假设成立时接近 MAML
  -----------------------------------------------------------------------

七、核心代码实现

7.1 内循环实现（MAMLTrainer）

Python - 内循环核心逻辑

> def inner_loop(self, support_users, support_items, support_labels):
>
> \"\"\"内循环：在support set上适配参数\"\"\"
>
> \# 获取当前参数的克隆
>
> params = {name: param.clone()
>
> for name, param in self.model.named_parameters()}
>
> for step in range(self.inner_steps):
>
> \# 计算 support loss
>
> loss = self.model.compute_loss(
>
> support_users, support_items, support_labels,
>
> params=params
>
> )
>
> \# 计算梯度
>
> grads = torch.autograd.grad(
>
> loss, params.values(),
>
> create_graph=not self.first_order \# MAML需要保留计算图
>
> )
>
> \# 解耦学习率更新
>
> if self.use_decoupled_lr:
>
> params = decoupled_inner_update(
>
> params, grads,
>
> lr_emb=self.lr_emb, \# Embedding层大学习率
>
> lr_dense=self.lr_dense \# Dense层小学习率
>
> )
>
> return params

7.2 元学习任务数据集

Python - 元学习任务构建

> class MetaTaskDataset(Dataset):
>
> \"\"\"元学习任务数据集
>
> 每个任务对应一个用户的冷启动场景：
>
> \- Support Set: 用户的前K条交互
>
> \- Query Set: 用户的后续交互
>
> \"\"\"
>
> def \_\_getitem\_\_(self, index):
>
> uid = self.task_users\[index\]
>
> interactions = self.user_interactions\[uid\]
>
> \# Support: 前n_support条
>
> support_ints = interactions\[:self.support_size\]
>
> \# Query: 之后的n_query条
>
> query_ints = interactions\[self.support_size:self.support_size +
> self.query_size\]
>
> \# 构建support set (包含正负样本)
>
> for it in support_ints:
>
> support_users.append(uid)
>
> support_items.append(it\[\"item_idx\"\])
>
> support_labels.append(it\[\"label\"\])
>
> \# 负采样
>
> for neg_item in self.\_sample_negatives(uid, self.neg_ratio):
>
> support_items.append(neg_item)
>
> support_labels.append(0.0)
>
> return {
>
> \"support_users\": torch.LongTensor(support_users),
>
> \"support_items\": torch.LongTensor(support_items),
>
> \"support_labels\": torch.FloatTensor(support_labels),
>
> \"query_users\": torch.LongTensor(query_users),
>
> \# \...
>
> }

7.3 ANIL适配实现

Python - ANIL head适配

> def inner_loop_head_only(self, support_users, support_items,
> support_labels):
>
> \"\"\"ANIL内循环：只适配head参数\"\"\"
>
> \# 获取head参数
>
> head_params = {}
>
> for name, param in self.model.named_parameters():
>
> if \"head\" in name:
>
> head_params\[name\] = param.clone().detach().requires_grad\_(True)
>
> \# 内循环: 只更新head
>
> for step in range(self.inner_steps):
>
> loss = self.model.compute_loss(
>
> support_users, support_items, support_labels,
>
> head_params=head_params
>
> )
>
> \# 只对head参数求梯度
>
> grads = torch.autograd.grad(
>
> loss, head_params.values(),
>
> create_graph=False \# ANIL不需要二阶梯度
>
> )
>
> \# 更新head参数
>
> new_head_params = {}
>
> for (name, param), grad in zip(head_params.items(), grads):
>
> new_head_params\[name\] = param - self.inner_lr \* grad
>
> head_params = new_head_params
>
> return head_params

八、实验结果汇总

  -----------------------------------------------------------------------------
  **方法**          **Cold-Start      **训练时间(相对)**   **备注**
                    AUC**                                  
  ----------------- ----------------- -------------------- --------------------
  Baseline (MLP)    \~0.68            1.0x                 无元学习

  MAML (Full        \~0.72            5.0x                 二阶梯度，计算昂贵
  2nd-order)                                               

  FOMAML            \~0.715           2.0x                 AUC损失\<0.5pp

  FOMAML + Meta-Emb \~0.735           2.2x                 +梯度解耦+低频补偿

  Reptile + ANIL    \~0.740           1.8x                 分层架构最优
  -----------------------------------------------------------------------------

九、参考文献

1.  Finn et al., \"Model-Agnostic Meta-Learning for Fast Adaptation of
    Deep Networks\", ICML 2017

2.  Nichol et al., \"On First-Order Meta-Learning Algorithms\", arXiv
    2018

3.  Raghu et al., \"Rapid Learning or Feature Reuse? Towards
    Understanding the Effectiveness of MAML\", ICLR 2020

4.  Lee et al., \"MeLU: Meta-Learned User Preference Estimator for
    Cold-Start Recommendation\", KDD 2019
