"""
基础推荐模型：Embedding + MLP
用于冷启动场景的元学习基线模型

模型结构:
  User Embedding ─┐
                  ├─ Concat ─ MLP Backbone ─ Prediction Head ─ sigmoid ─ output
  Item Embedding ─┘

块对角稀疏结构分析:
  - Embedding层参数: 仅依赖输入的特定ID，梯度天然稀疏（只有被查询的行有梯度）
  - Dense层参数: 全连接，梯度密集
  - 整体Hessian矩阵呈块对角结构: H = diag(H_emb, H_dense)
    其中 H_emb 本身是极度稀疏的
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict


class BaseRecModel(nn.Module):
    """
    基础推荐模型
    
    支持两种前向传播模式：
    1. 标准前向（使用self.parameters()）
    2. 函数式前向（使用外部传入的params字典）—— 用于MAML的内循环
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
        self.user_emb_dim = user_emb_dim
        self.item_emb_dim = item_emb_dim
        
        # === Embedding Layer ===
        self.user_embedding = nn.Embedding(num_users, user_emb_dim)
        self.item_embedding = nn.Embedding(num_items, item_emb_dim)
        
        # 初始化
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # === MLP Backbone ===
        input_dim = user_emb_dim + item_emb_dim
        backbone_layers = []
        for i, hdim in enumerate(hidden_dims):
            backbone_layers.append((f"fc_{i}", nn.Linear(input_dim, hdim)))
            backbone_layers.append((f"bn_{i}", nn.BatchNorm1d(hdim)))
            backbone_layers.append((f"relu_{i}", nn.ReLU()))
            backbone_layers.append((f"drop_{i}", nn.Dropout(dropout)))
            input_dim = hdim
        self.backbone = nn.Sequential(OrderedDict(backbone_layers))
        
        # === Prediction Head ===
        self.head = nn.Linear(hidden_dims[-1], 1)
        
        # 标记各参数的层级归属（用于分层元学习）
        self._param_groups = self._build_param_groups()
    
    def _build_param_groups(self) -> Dict[str, List[str]]:
        """将参数分为 embedding / backbone / head 三组"""
        groups = {"embedding": [], "backbone": [], "head": []}
        for name, _ in self.named_parameters():
            if "embedding" in name:
                groups["embedding"].append(name)
            elif "head" in name:
                groups["head"].append(name)
            else:
                groups["backbone"].append(name)
        return groups
    
    def get_param_groups(self) -> Dict[str, List[str]]:
        return self._param_groups
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            user_ids: [B] 用户ID
            item_ids: [B] 物品ID
            params: 可选，外部参数字典（用于MAML函数式前向）
        
        Returns:
            logits: [B] 预测分数 (未过sigmoid)
        """
        if params is not None:
            return self._functional_forward(user_ids, item_ids, params)
        
        # 标准前向
        user_emb = self.user_embedding(user_ids)     # [B, user_emb_dim]
        item_emb = self.item_embedding(item_ids)     # [B, item_emb_dim]
        
        x = torch.cat([user_emb, item_emb], dim=-1)  # [B, user_emb_dim + item_emb_dim]
        x = self.backbone(x)                          # [B, hidden_dims[-1]]
        logits = self.head(x).squeeze(-1)             # [B]
        
        return logits
    
    def _functional_forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        函数式前向传播：使用外部参数而非self的参数
        这是MAML内循环的关键——允许在不修改模型参数的情况下用更新后的参数做前向
        """
        user_emb = F.embedding(user_ids, params["user_embedding.weight"])
        item_emb = F.embedding(item_ids, params["item_embedding.weight"])
        
        x = torch.cat([user_emb, item_emb], dim=-1)
        
        # 手动通过 backbone 各层
        layer_idx = 0
        hidden_dims_count = len([k for k in params if k.startswith("backbone.fc_") and k.endswith(".weight")])
        
        for i in range(hidden_dims_count):
            # Linear
            w = params[f"backbone.fc_{i}.weight"]
            b = params[f"backbone.fc_{i}.bias"]
            x = F.linear(x, w, b)
            
            # BatchNorm (使用running stats from self, 但 gamma/beta from params)
            bn_weight = params.get(f"backbone.bn_{i}.weight")
            bn_bias = params.get(f"backbone.bn_{i}.bias")
            bn_layer = getattr(self.backbone, f"bn_{i}")
            x = F.batch_norm(
                x,
                bn_layer.running_mean,
                bn_layer.running_var,
                bn_weight,
                bn_bias,
                training=self.training,
            )
            
            # ReLU
            x = F.relu(x)
            # Dropout（训练时）
            if self.training:
                x = F.dropout(x, p=0.2)
        
        # Head
        logits = F.linear(x, params["head.weight"], params["head.bias"]).squeeze(-1)
        
        return logits
    
    def get_params_dict(self) -> Dict[str, torch.Tensor]:
        """获取当前参数的字典形式"""
        return {name: param.clone() for name, param in self.named_parameters()}
    
    def analyze_block_diagonal_structure(self) -> Dict[str, dict]:
        """
        分析模型参数的块对角稀疏结构
        
        推荐模型的Hessian矩阵具有天然的块对角结构：
        
        H = | H_user_emb    0           0       |
            |    0       H_item_emb      0       |
            |    0           0       H_dense     |
        
        其中 H_user_emb 和 H_item_emb 是极度稀疏的（仅mini-batch中涉及的ID对应行有非零值）
        
        Returns:
            各参数块的统计信息
        """
        analysis = {}
        total_params = 0
        
        for group_name, param_names in self._param_groups.items():
            group_params = 0
            for pname in param_names:
                param = dict(self.named_parameters())[pname]
                group_params += param.numel()
            
            analysis[group_name] = {
                "num_params": group_params,
                "param_names": param_names,
                "is_sparse": group_name == "embedding",
                "sparsity_note": (
                    "Embedding梯度天然稀疏：每次前向只有batch中出现的ID有梯度"
                    if group_name == "embedding"
                    else "Dense层梯度密集，但参数量远小于Embedding层"
                ),
            }
            total_params += group_params
        
        # 计算各块占比
        for gn in analysis:
            analysis[gn]["ratio"] = analysis[gn]["num_params"] / total_params
        
        analysis["total_params"] = total_params
        analysis["hessian_note"] = (
            "Hessian矩阵呈块对角结构：\n"
            "1. Embedding块: 极度稀疏，每次仅O(batch_size)行有梯度\n"
            "2. Dense块: 密集但规模小\n"
            "3. 交叉项(off-diagonal): 近似为0\n"
            "=> FOMAML忽略二阶项的损失很小，因为Hessian本身就很稀疏"
        )
        
        return analysis
    
    def compute_loss(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        labels: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """计算 BCE Loss"""
        logits = self.forward(user_ids, item_ids, params=params)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss
