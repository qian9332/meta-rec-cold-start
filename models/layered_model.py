"""
分层元学习模型：Reptile预训练backbone + ANIL在线适配head

设计思想:
  Raghu et al. (ICLR 2020) 发现 MAML 的有效性主要来自特征复用(feature reuse)
  而非快速学习(rapid learning)。这意味着：
  
  1. Backbone（特征提取层）: 学习到的特征表示是跨任务通用的
     => 用 Reptile 预训练，学习通用特征提取能力
     
  2. Head（预测层）: 需要针对每个新任务进行适配
     => 用 ANIL 只适配head，冻结backbone
  
  优势:
  - Reptile预训练: 简单高效，无需二阶梯度，适合大规模模型
  - ANIL适配: 只更新head参数，在线适配速度快
  - 整体: 兼顾训练效率与冷启动适配精度

架构:
  ┌──────────────────────────────────────┐
  │ Reptile Pre-trained (Frozen)          │
  │ ┌────────────────────────────────────┐│
  │ │ User Embedding → Meta-Embedding   ││
  │ │ Item Embedding → Meta-Embedding   ││
  │ │ Concat → MLP Backbone (fc0→fc1→fc2)││
  │ └────────────────────────────────────┘│
  └──────────────────────────────────────┘
               ↓
  ┌──────────────────────────────────────┐
  │ ANIL Adaptive Head                    │
  │ ┌────────────────────────────────────┐│
  │ │ Adaptive Linear → Sigmoid          ││
  │ └────────────────────────────────────┘│
  └──────────────────────────────────────┘
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import copy


class LayeredMetaRecModel(nn.Module):
    """
    分层元学习推荐模型
    
    将模型明确分为 backbone 和 head：
    - backbone: 包含 embedding + MLP hidden layers
    - head: 最终预测层
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_emb_dim: int = 64,
        item_emb_dim: int = 64,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        
        # === Backbone: Embedding + MLP ===
        self.user_embedding = nn.Embedding(num_users, user_emb_dim)
        self.item_embedding = nn.Embedding(num_items, item_emb_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        input_dim = user_emb_dim + item_emb_dim
        backbone_layers = []
        for i, hdim in enumerate(hidden_dims):
            backbone_layers.append((f"fc_{i}", nn.Linear(input_dim, hdim)))
            backbone_layers.append((f"relu_{i}", nn.ReLU()))
            backbone_layers.append((f"drop_{i}", nn.Dropout(dropout)))
            input_dim = hdim
        self.backbone = nn.Sequential(OrderedDict(backbone_layers))
        
        # === Adaptive Head ===
        self.head = nn.Sequential(OrderedDict([
            ("adaptive_fc", nn.Linear(hidden_dims[-1], 32)),
            ("adaptive_relu", nn.ReLU()),
            ("output", nn.Linear(32, 1)),
        ]))
        
        self.backbone_dim = hidden_dims[-1]
    
    def get_backbone_params(self) -> List[nn.Parameter]:
        """获取backbone参数（用于Reptile预训练）"""
        params = list(self.user_embedding.parameters()) + \
                 list(self.item_embedding.parameters()) + \
                 list(self.backbone.parameters())
        return params
    
    def get_head_params(self) -> List[nn.Parameter]:
        """获取head参数（用于ANIL适配）"""
        return list(self.head.parameters())
    
    def get_backbone_state(self) -> Dict[str, torch.Tensor]:
        """获取backbone的state dict"""
        state = {}
        for name, param in self.named_parameters():
            if "head" not in name:
                state[name] = param.data.clone()
        return state
    
    def get_head_state(self) -> Dict[str, torch.Tensor]:
        """获取head的state dict"""
        state = {}
        for name, param in self.named_parameters():
            if "head" in name:
                state[name] = param.data.clone()
        return state
    
    def set_backbone_state(self, state: Dict[str, torch.Tensor]):
        """设置backbone的state"""
        for name, param in self.named_parameters():
            if name in state:
                param.data.copy_(state[name])
    
    def freeze_backbone(self):
        """冻结backbone参数（ANIL适配时使用）"""
        for param in self.get_backbone_params():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """解冻backbone参数"""
        for param in self.get_backbone_params():
            param.requires_grad = True
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        head_params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            user_ids: [B] 用户ID
            item_ids: [B] 物品ID
            head_params: 可选，外部head参数（用于ANIL函数式前向）
        """
        # Backbone forward (always use self params)
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        x = torch.cat([user_emb, item_emb], dim=-1)
        features = self.backbone(x)  # [B, backbone_dim]
        
        # Head forward (can use external params)
        if head_params is not None:
            x = F.linear(features, head_params["head.adaptive_fc.weight"],
                         head_params["head.adaptive_fc.bias"])
            x = F.relu(x)
            logits = F.linear(x, head_params["head.output.weight"],
                              head_params["head.output.bias"]).squeeze(-1)
        else:
            logits = self.head(features).squeeze(-1)
        
        return logits
    
    def compute_loss(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        labels: torch.Tensor,
        head_params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """计算BCE Loss"""
        logits = self.forward(user_ids, item_ids, head_params=head_params)
        return F.binary_cross_entropy_with_logits(logits, labels)
    
    def extract_features(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """提取backbone特征（用于分析）"""
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)
            item_emb = self.item_embedding(item_ids)
            x = torch.cat([user_emb, item_emb], dim=-1)
            features = self.backbone(x)
        return features


class ReptilePretrainWrapper:
    """
    Reptile预训练包装器
    
    Reptile算法:
      for each meta-iteration:
        sample task T
        θ' = SGD(θ, T, k steps)    # 内循环：在任务T上训练k步
        θ = θ + ε * (θ' - θ)       # 外循环：向任务适配后的参数靠近
    
    只更新backbone参数，head重新初始化
    """
    
    def __init__(
        self,
        model: LayeredMetaRecModel,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        epsilon: float = 0.1,
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.epsilon = epsilon
    
    def reptile_step(
        self,
        support_users: torch.Tensor,
        support_items: torch.Tensor,
        support_labels: torch.Tensor,
    ):
        """
        执行一步 Reptile 更新
        
        1. 保存当前backbone参数
        2. 在support set上训练几步
        3. 计算参数差异，更新backbone
        """
        # 保存初始backbone状态
        init_state = self.model.get_backbone_state()
        
        # 内循环：在support set上训练
        inner_optimizer = torch.optim.SGD(self.model.get_backbone_params(), lr=self.inner_lr)
        
        self.model.train()
        for step in range(self.inner_steps):
            inner_optimizer.zero_grad()
            loss = self.model.compute_loss(support_users, support_items, support_labels)
            loss.backward()
            inner_optimizer.step()
        
        # 外循环：Reptile更新 θ = θ + ε * (θ' - θ)
        adapted_state = self.model.get_backbone_state()
        
        with torch.no_grad():
            for name in init_state:
                if name in adapted_state:
                    diff = adapted_state[name] - init_state[name]
                    init_state[name] = init_state[name] + self.epsilon * diff
        
        # 恢复为Reptile更新后的参数
        self.model.set_backbone_state(init_state)


class ANILAdaptWrapper:
    """
    ANIL在线适配包装器
    
    ANIL (Almost No Inner Loop):
      - 冻结backbone（使用Reptile预训练的参数）
      - 只在内循环中适配head参数
      - 大幅减少适配时的计算量
    """
    
    def __init__(
        self,
        model: LayeredMetaRecModel,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
    
    def adapt_head(
        self,
        support_users: torch.Tensor,
        support_items: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        在support set上适配head
        
        Returns:
            适配后的head参数字典
        """
        # 冻结backbone
        self.model.freeze_backbone()
        
        # 获取初始head参数
        head_params = {}
        for name, param in self.model.named_parameters():
            if "head" in name:
                head_params[name] = param.clone().detach().requires_grad_(True)
        
        # 内循环：只更新head
        for step in range(self.inner_steps):
            loss = self.model.compute_loss(
                support_users, support_items, support_labels,
                head_params=head_params
            )
            
            grads = torch.autograd.grad(loss, head_params.values(), create_graph=False)
            
            head_params = {
                name: param - self.inner_lr * grad
                for (name, param), grad in zip(head_params.items(), grads)
            }
        
        # 解冻backbone
        self.model.unfreeze_backbone()
        
        return head_params
    
    def meta_step(
        self,
        task_batch: Dict[str, List[torch.Tensor]],
        outer_optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        执行一步ANIL元更新
        
        1. 对每个任务，在support set上适配head
        2. 在query set上计算meta loss
        3. 更新整个模型（但backbone梯度很小因为被冻结过）
        """
        meta_loss = 0.0
        num_tasks = len(task_batch["support_users"])
        
        for i in range(num_tasks):
            s_users = task_batch["support_users"][i]
            s_items = task_batch["support_items"][i]
            s_labels = task_batch["support_labels"][i]
            q_users = task_batch["query_users"][i]
            q_items = task_batch["query_items"][i]
            q_labels = task_batch["query_labels"][i]
            
            device = next(self.model.parameters()).device
            s_users, s_items, s_labels = s_users.to(device), s_items.to(device), s_labels.to(device)
            q_users, q_items, q_labels = q_users.to(device), q_items.to(device), q_labels.to(device)
            
            # 适配head
            adapted_head = self.adapt_head(s_users, s_items, s_labels)
            
            # 在query set上评估
            query_loss = self.model.compute_loss(q_users, q_items, q_labels, head_params=adapted_head)
            meta_loss += query_loss
        
        meta_loss = meta_loss / num_tasks
        
        outer_optimizer.zero_grad()
        meta_loss.backward()
        outer_optimizer.step()
        
        return meta_loss.item()
