"""
评估指标：AUC, HR@K, NDCG@K
"""
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Optional, Tuple


def compute_auc(predictions: np.ndarray, labels: np.ndarray) -> float:
    """计算 AUC"""
    if len(np.unique(labels)) < 2:
        return 0.5  # 只有一个类别时返回0.5
    try:
        return roc_auc_score(labels, predictions)
    except ValueError:
        return 0.5


def compute_hr_at_k(predictions: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """
    计算 Hit Rate @ K
    将预测分数最高的K个物品与正样本对比
    """
    if len(predictions) == 0:
        return 0.0
    
    # 按预测分数降序排列
    topk_indices = np.argsort(predictions)[::-1][:k]
    topk_labels = labels[topk_indices]
    
    # 是否命中正样本
    return float(np.sum(topk_labels > 0) > 0)


def compute_ndcg_at_k(predictions: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """
    计算 NDCG @ K
    """
    if len(predictions) == 0 or np.sum(labels) == 0:
        return 0.0
    
    # 按预测分数降序排列
    topk_indices = np.argsort(predictions)[::-1][:k]
    topk_labels = labels[topk_indices]
    
    # DCG
    dcg = np.sum(topk_labels / np.log2(np.arange(2, k + 2)))
    
    # IDCG（理想排序）
    ideal_labels = np.sort(labels)[::-1][:k]
    idcg = np.sum(ideal_labels / np.log2(np.arange(2, k + 2)))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def evaluate_meta_model(
    model,
    eval_loader,
    device: torch.device,
    inner_lr: float = 0.01,
    inner_steps: int = 3,
    first_order: bool = True,
    use_head_params: bool = False,
) -> Dict[str, float]:
    """
    评估元学习模型在冷启动任务上的表现
    
    对每个任务：
    1. 在support set上做内循环适配
    2. 在query set上评估
    
    Args:
        model: 推荐模型
        eval_loader: 评估数据加载器
        device: 设备
        inner_lr: 内循环学习率
        inner_steps: 内循环步数
        first_order: 是否使用一阶近似
        use_head_params: 是否仅适配head（用于ANIL）
    
    Returns:
        各指标的平均值
    """
    model.eval()
    
    all_auc = []
    all_hr10 = []
    all_ndcg10 = []
    
    for batch in eval_loader:
        num_tasks = len(batch["support_users"])
        
        for i in range(num_tasks):
            s_users = batch["support_users"][i].to(device)
            s_items = batch["support_items"][i].to(device)
            s_labels = batch["support_labels"][i].to(device)
            q_users = batch["query_users"][i].to(device)
            q_items = batch["query_items"][i].to(device)
            q_labels = batch["query_labels"][i].to(device)
            
            if len(q_labels) == 0 or len(s_labels) == 0:
                continue
            
            # === 内循环适配 ===
            if use_head_params:
                # ANIL: 只适配head
                head_params = {}
                for name, param in model.named_parameters():
                    if "head" in name:
                        head_params[name] = param.clone().detach().requires_grad_(True)
                
                for step in range(inner_steps):
                    loss = model.compute_loss(s_users, s_items, s_labels, head_params=head_params)
                    grads = torch.autograd.grad(loss, head_params.values())
                    head_params = {
                        name: param - inner_lr * grad
                        for (name, param), grad in zip(head_params.items(), grads)
                    }
                
                # 评估
                with torch.no_grad():
                    logits = model(q_users, q_items, head_params=head_params)
            else:
                # MAML/FOMAML: 适配所有参数
                params = {name: param.clone() for name, param in model.named_parameters()}
                
                for step in range(inner_steps):
                    loss = model.compute_loss(s_users, s_items, s_labels, params=params)
                    grads = torch.autograd.grad(loss, params.values(), create_graph=not first_order)
                    params = {
                        name: param - inner_lr * grad
                        for (name, param), grad in zip(params.items(), grads)
                    }
                
                # 评估
                with torch.no_grad():
                    logits = model(q_users, q_items, params=params)
            
            preds = torch.sigmoid(logits).cpu().numpy()
            labels = q_labels.cpu().numpy()
            
            if len(np.unique(labels)) >= 2:
                all_auc.append(compute_auc(preds, labels))
            all_hr10.append(compute_hr_at_k(preds, labels, k=10))
            all_ndcg10.append(compute_ndcg_at_k(preds, labels, k=10))
    
    results = {
        "auc": np.mean(all_auc) if all_auc else 0.5,
        "hr@10": np.mean(all_hr10) if all_hr10 else 0.0,
        "ndcg@10": np.mean(all_ndcg10) if all_ndcg10 else 0.0,
        "num_tasks_evaluated": len(all_auc),
    }
    
    return results
