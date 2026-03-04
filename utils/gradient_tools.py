"""
梯度工具：梯度补偿、裁剪、统计分析
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple


class GradientCompensator:
    """
    梯度补偿器
    
    用于解决元学习中低频ID梯度消失的问题：
    - 统计各ID的出现频率
    - 对低频ID的梯度进行放大补偿
    - 支持自适应补偿系数
    """
    
    def __init__(
        self,
        num_ids: int,
        low_freq_threshold: int = 10,
        compensation_scale: float = 2.0,
        adaptive: bool = True,
    ):
        self.num_ids = num_ids
        self.low_freq_threshold = low_freq_threshold
        self.compensation_scale = compensation_scale
        self.adaptive = adaptive
        
        # 频率统计
        self.id_freq = np.zeros(num_ids, dtype=np.int64)
        self.total_batches = 0
    
    def update_freq(self, ids: np.ndarray):
        """更新ID频率"""
        unique_ids, counts = np.unique(ids, return_counts=True)
        self.id_freq[unique_ids] += counts
        self.total_batches += 1
    
    def get_compensation_factors(self, ids: np.ndarray) -> np.ndarray:
        """
        计算补偿系数
        
        低频ID: scale = compensation_scale * (threshold / max(freq, 1))
        高频ID: scale = 1.0
        """
        freqs = self.id_freq[ids]
        
        if self.adaptive:
            # 自适应补偿：频率越低，补偿越大
            factors = np.where(
                freqs < self.low_freq_threshold,
                self.compensation_scale * (self.low_freq_threshold / np.maximum(freqs, 1)),
                np.ones_like(freqs, dtype=np.float32),
            )
            # 上限裁剪
            factors = np.clip(factors, 1.0, self.compensation_scale * 5)
        else:
            # 固定补偿
            factors = np.where(
                freqs < self.low_freq_threshold,
                np.full_like(freqs, self.compensation_scale, dtype=np.float32),
                np.ones_like(freqs, dtype=np.float32),
            )
        
        return factors.astype(np.float32)
    
    def get_stats(self) -> Dict[str, float]:
        """获取频率统计信息"""
        return {
            "total_ids": self.num_ids,
            "zero_freq_count": int(np.sum(self.id_freq == 0)),
            "low_freq_count": int(np.sum(self.id_freq < self.low_freq_threshold)),
            "high_freq_count": int(np.sum(self.id_freq >= self.low_freq_threshold)),
            "mean_freq": float(np.mean(self.id_freq)),
            "median_freq": float(np.median(self.id_freq)),
            "max_freq": int(np.max(self.id_freq)),
            "total_batches": self.total_batches,
        }


def compute_grad_stats(
    model: nn.Module,
    loss: torch.Tensor,
    retain_graph: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    计算模型各参数的梯度统计信息
    
    Returns:
        {param_name: {norm, mean, std, max, min, sparsity}}
    """
    grads = torch.autograd.grad(
        loss,
        model.parameters(),
        create_graph=False,
        retain_graph=retain_graph,
    )
    
    stats = {}
    for (name, param), grad in zip(model.named_parameters(), grads):
        g = grad.detach()
        stats[name] = {
            "norm": g.norm().item(),
            "mean": g.mean().item(),
            "std": g.std().item(),
            "max": g.max().item(),
            "min": g.min().item(),
            "sparsity": (g == 0).float().mean().item(),
            "shape": list(g.shape),
        }
    
    return stats


def decoupled_inner_update(
    params: Dict[str, torch.Tensor],
    grads: Tuple[torch.Tensor, ...],
    param_names: List[str],
    lr_emb: float = 0.01,
    lr_dense: float = 0.001,
    grad_clip_norm: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    解耦学习率的内循环参数更新
    
    Embedding层使用 lr_emb（较大）
    Dense层使用 lr_dense（较小）
    
    这是缓解Meta-Embedding梯度消失的核心方案:
    - Embedding层梯度天然稀疏，需要更大步长
    - Dense层梯度密集，正常步长即可
    - 解耦后，冷启动AUC可提升1~2pp
    
    Args:
        params: 当前参数字典
        grads: 对应的梯度
        param_names: 参数名列表
        lr_emb: Embedding层学习率
        lr_dense: Dense层学习率
        grad_clip_norm: 梯度裁剪范数
    
    Returns:
        更新后的参数字典
    """
    updated_params = {}
    
    for (name, param), grad in zip(zip(param_names, params.values()), grads):
        # 选择学习率
        lr = lr_emb if ("embedding" in name or "emb" in name) else lr_dense
        
        # 梯度裁剪
        if grad_clip_norm > 0:
            grad_norm = grad.norm()
            if grad_norm > grad_clip_norm:
                grad = grad * (grad_clip_norm / grad_norm)
        
        # 更新
        updated_params[name] = param - lr * grad
    
    return updated_params


def apply_gradient_compensation(
    grad: torch.Tensor,
    embedding_weight: torch.Tensor,
    active_ids: torch.Tensor,
    id_freq: torch.Tensor,
    low_freq_threshold: int = 10,
    compensation_scale: float = 2.0,
) -> torch.Tensor:
    """
    对Embedding梯度应用低频ID补偿
    
    Args:
        grad: Embedding权重的梯度 [num_embeddings, emb_dim]
        embedding_weight: 当前Embedding权重
        active_ids: 本次batch中活跃的ID
        id_freq: ID频率统计
        low_freq_threshold: 低频阈值
        compensation_scale: 补偿系数
    
    Returns:
        补偿后的梯度
    """
    compensated_grad = grad.clone()
    
    unique_ids = active_ids.unique()
    for uid in unique_ids:
        freq = id_freq[uid].item()
        if freq < low_freq_threshold:
            # 补偿系数 = base_scale * (threshold / max(freq, 1))
            factor = compensation_scale * (low_freq_threshold / max(freq, 1))
            factor = min(factor, compensation_scale * 5)  # 上限
            compensated_grad[uid] *= factor
    
    return compensated_grad
