"""
Meta-Embedding 层：解决稀疏ID特征导致的梯度消失问题

核心问题：
  在元学习的内循环中，Embedding层的梯度更新面临两个挑战：
  1. 稀疏性：每个mini-batch只有少数ID被查询，大部分Embedding行梯度为0
  2. 低频ID：冷启动用户/物品的Embedding几乎没有得到过训练，梯度信号极弱

解决方案（两个层面）：
  A. 核心方案 —— Embedding层与Dense层学习率解耦
     - Embedding层使用更大的内循环学习率 (lr_emb >> lr_dense)
     - 补偿稀疏梯度导致的更新不足
  
  B. 辅助方案 —— 低频ID梯度补偿
     - 统计各ID的出现频率
     - 对低频ID的梯度乘以补偿系数 (freq < threshold => grad *= compensation_scale)
     - 让冷启动实体获得更强的梯度信号
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class MetaEmbeddingLayer(nn.Module):
    """
    增强版Embedding层，支持：
    1. 学习率解耦（内循环中Emb和Dense使用不同lr）
    2. 低频ID梯度补偿
    3. 梯度裁剪
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        lr_emb: float = 0.01,
        lr_dense: float = 0.001,
        low_freq_threshold: int = 10,
        compensation_scale: float = 2.0,
        grad_clip_norm: float = 1.0,
        name: str = "meta_emb",
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.lr_emb = lr_emb
        self.lr_dense = lr_dense
        self.low_freq_threshold = low_freq_threshold
        self.compensation_scale = compensation_scale
        self.grad_clip_norm = grad_clip_norm
        self.name = name
        
        # Embedding 参数
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        # ID频率统计（不参与梯度计算）
        self.register_buffer("id_freq", torch.zeros(num_embeddings, dtype=torch.long))
        self.register_buffer("total_updates", torch.tensor(0, dtype=torch.long))
    
    def forward(self, ids: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            ids: [B] or [B, L] ID张量
            weight: 可选，外部权重（用于函数式前向）
        """
        w = weight if weight is not None else self.weight
        return F.embedding(ids, w)
    
    def update_freq(self, ids: torch.Tensor):
        """更新ID频率统计"""
        unique_ids = ids.unique()
        self.id_freq[unique_ids] += 1
        self.total_updates += 1
    
    def get_compensation_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """
        计算低频ID的梯度补偿掩码
        
        对于出现频率低于阈值的ID，返回 compensation_scale
        对于高频ID，返回 1.0
        """
        freqs = self.id_freq[ids].float()  # [B] or [B, L]
        
        # 低频ID => 补偿系数
        mask = torch.where(
            freqs < self.low_freq_threshold,
            torch.full_like(freqs, self.compensation_scale),
            torch.ones_like(freqs),
        )
        
        return mask.unsqueeze(-1)  # [B, 1] or [B, L, 1]，用于逐元素乘
    
    def compensated_forward(self, ids: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        带梯度补偿的前向传播
        
        通过在前向传播中引入补偿系数，间接放大低频ID的梯度
        原理: 如果 output = scale * emb(id), 则 d(loss)/d(emb) = scale * d(loss)/d(output)
        """
        emb = self.forward(ids, weight)
        
        if self.training:
            self.update_freq(ids)
            comp_mask = self.get_compensation_mask(ids)
            # 通过 scale trick 实现梯度补偿
            # 正向传播时值不变，但反向传播时梯度被放大
            emb = emb * comp_mask.detach() + emb * (comp_mask - comp_mask.detach())
        
        return emb
    
    def meta_update(
        self,
        grad: torch.Tensor,
        current_weight: torch.Tensor,
        active_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        元学习内循环的参数更新（使用解耦学习率）
        
        Args:
            grad: Embedding权重的梯度 [num_embeddings, embedding_dim]
            current_weight: 当前Embedding权重
            active_ids: 本次batch中活跃的ID（用于稀疏更新）
        
        Returns:
            更新后的Embedding权重
        
        关键：使用 lr_emb 而非 lr_dense，因为：
        1. Embedding梯度天然稀疏，需要更大步长补偿
        2. 冷启动ID在meta-train中出现次数少，需要更激进的更新
        """
        # 梯度裁剪
        if self.grad_clip_norm > 0:
            grad_norm = grad.norm()
            if grad_norm > self.grad_clip_norm:
                grad = grad * (self.grad_clip_norm / grad_norm)
        
        # 低频ID梯度补偿
        if active_ids is not None:
            comp_factors = torch.ones(self.num_embeddings, 1, device=grad.device)
            low_freq_mask = self.id_freq < self.low_freq_threshold
            comp_factors[low_freq_mask] = self.compensation_scale
            grad = grad * comp_factors
        
        # 使用解耦的 lr_emb 进行更新
        updated_weight = current_weight - self.lr_emb * grad
        
        return updated_weight


class DecoupledMetaEmbedding(nn.Module):
    """
    解耦学习率的完整 Meta-Embedding 模块
    
    封装了 user_embedding 和 item_embedding，
    提供统一的元学习更新接口
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_emb_dim: int = 64,
        item_emb_dim: int = 64,
        lr_emb: float = 0.01,
        lr_dense: float = 0.001,
        low_freq_threshold: int = 10,
        compensation_scale: float = 2.0,
        grad_clip_norm: float = 1.0,
    ):
        super().__init__()
        
        self.user_emb = MetaEmbeddingLayer(
            num_users, user_emb_dim,
            lr_emb=lr_emb, lr_dense=lr_dense,
            low_freq_threshold=low_freq_threshold,
            compensation_scale=compensation_scale,
            grad_clip_norm=grad_clip_norm,
            name="user_emb",
        )
        
        self.item_emb = MetaEmbeddingLayer(
            num_items, item_emb_dim,
            lr_emb=lr_emb, lr_dense=lr_dense,
            low_freq_threshold=low_freq_threshold,
            compensation_scale=compensation_scale,
            grad_clip_norm=grad_clip_norm,
            name="item_emb",
        )
        
        self.lr_emb = lr_emb
        self.lr_dense = lr_dense
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        user_weight: Optional[torch.Tensor] = None,
        item_weight: Optional[torch.Tensor] = None,
        compensated: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Returns:
            (user_emb, item_emb): 用户和物品的嵌入向量
        """
        if compensated and self.training:
            user_emb = self.user_emb.compensated_forward(user_ids, user_weight)
            item_emb = self.item_emb.compensated_forward(item_ids, item_weight)
        else:
            user_emb = self.user_emb(user_ids, user_weight)
            item_emb = self.item_emb(item_ids, item_weight)
        
        return user_emb, item_emb
    
    def get_decoupled_lr(self, param_name: str) -> float:
        """
        根据参数名返回解耦的学习率
        
        Args:
            param_name: 参数名
        Returns:
            对应的学习率（Embedding层 -> lr_emb, Dense层 -> lr_dense）
        """
        if "emb" in param_name or "embedding" in param_name:
            return self.lr_emb
        else:
            return self.lr_dense
    
    def get_freq_stats(self) -> Dict[str, dict]:
        """获取频率统计信息"""
        return {
            "user_emb": {
                "total_ids": self.user_emb.num_embeddings,
                "low_freq_count": (self.user_emb.id_freq < self.user_emb.low_freq_threshold).sum().item(),
                "zero_freq_count": (self.user_emb.id_freq == 0).sum().item(),
                "mean_freq": self.user_emb.id_freq.float().mean().item(),
                "max_freq": self.user_emb.id_freq.max().item(),
            },
            "item_emb": {
                "total_ids": self.item_emb.num_embeddings,
                "low_freq_count": (self.item_emb.id_freq < self.item_emb.low_freq_threshold).sum().item(),
                "zero_freq_count": (self.item_emb.id_freq == 0).sum().item(),
                "mean_freq": self.item_emb.id_freq.float().mean().item(),
                "max_freq": self.item_emb.id_freq.max().item(),
            },
        }
