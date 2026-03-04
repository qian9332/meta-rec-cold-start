"""
Reptile预训练器

Reptile算法 (Nichol et al., 2018):
  简化版元学习，无需二阶梯度

  for each meta-iteration:
    θ_init = θ
    sample task T
    θ' = SGD(θ, T, k steps)        # 内循环
    θ = θ + ε * (θ' - θ_init)      # 外循环（插值）

  等价于: θ = (1 - ε) * θ_init + ε * θ'

Reptile vs MAML:
  - 不需要二阶梯度 => 计算量接近普通训练
  - 理论上近似一阶MAML + L2正则
  - 适合预训练backbone的通用特征
"""
import torch
import torch.nn as nn
import numpy as np
import time
import copy
from typing import Dict, List, Optional
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import evaluate_meta_model


class ReptileTrainer:
    """Reptile预训练器，用于预训练backbone"""
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        epsilon: float = 0.1,
        inner_steps: int = 5,
        device: torch.device = None,
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.epsilon = epsilon
        self.inner_steps = inner_steps
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(self.device)
        
        self.train_history = {"loss": [], "eval_auc": [], "time_per_epoch": []}
    
    def reptile_step(
        self,
        support_users: torch.Tensor,
        support_items: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> float:
        """
        执行一步Reptile更新
        
        1. 保存θ_init
        2. 在task上SGD k步 => θ'
        3. θ = θ + ε(θ' - θ_init)
        """
        # 保存初始参数
        init_state = {name: param.data.clone() for name, param in self.model.named_parameters()}
        
        # 内循环SGD
        inner_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)
        
        self.model.train()
        total_loss = 0
        for step in range(self.inner_steps):
            inner_optimizer.zero_grad()
            
            if hasattr(self.model, 'compute_loss'):
                loss = self.model.compute_loss(support_users, support_items, support_labels)
            else:
                logits = self.model(support_users, support_items)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, support_labels)
            
            loss.backward()
            inner_optimizer.step()
            total_loss += loss.item()
        
        # Reptile外循环: θ = θ + ε(θ' - θ)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                adapted = param.data.clone()
                original = init_state[name]
                # 插值更新
                param.data.copy_(original + self.epsilon * (adapted - original))
        
        return total_loss / self.inner_steps
    
    def train(
        self,
        train_loader,
        eval_loader=None,
        num_epochs: int = 30,
        eval_every: int = 5,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """Reptile预训练完整流程"""
        
        print(f"\n{'='*60}")
        print(f"Reptile 预训练")
        print(f"内循环步数: {self.inner_steps}, 内循环LR: {self.inner_lr}")
        print(f"Epsilon: {self.epsilon}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            epoch_losses = []
            
            pbar = tqdm(train_loader, desc=f"Reptile Epoch {epoch}/{num_epochs}", disable=not verbose)
            for batch in pbar:
                num_tasks = len(batch["support_users"])
                
                batch_loss = 0
                for i in range(num_tasks):
                    s_users = batch["support_users"][i].to(self.device)
                    s_items = batch["support_items"][i].to(self.device)
                    s_labels = batch["support_labels"][i].to(self.device)
                    
                    if len(s_labels) == 0:
                        continue
                    
                    loss = self.reptile_step(s_users, s_items, s_labels)
                    batch_loss += loss
                
                avg_batch_loss = batch_loss / max(num_tasks, 1)
                epoch_losses.append(avg_batch_loss)
                pbar.set_postfix({"loss": f"{avg_batch_loss:.4f}"})
            
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0
            epoch_time = time.time() - epoch_start
            
            self.train_history["loss"].append(avg_loss)
            self.train_history["time_per_epoch"].append(epoch_time)
            
            # 评估
            if eval_loader is not None and epoch % eval_every == 0:
                eval_results = evaluate_meta_model(
                    self.model, eval_loader, self.device,
                    inner_lr=self.inner_lr,
                    inner_steps=self.inner_steps,
                    first_order=True,
                )
                self.train_history["eval_auc"].append(eval_results["auc"])
                
                if verbose:
                    print(f"\n  [Eval] Epoch {epoch}: AUC={eval_results['auc']:.4f}, "
                          f"HR@10={eval_results['hr@10']:.4f}, NDCG@10={eval_results['ndcg@10']:.4f}")
            
            if verbose and epoch % eval_every == 0:
                print(f"  [Train] Epoch {epoch}: loss={avg_loss:.4f}, time={epoch_time:.1f}s")
        
        return self.train_history
    
    def save_pretrained(self, path: str):
        """保存预训练模型"""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "train_history": self.train_history,
        }, path)
        print(f"[Reptile] 模型已保存到 {path}")
    
    def load_pretrained(self, path: str):
        """加载预训练模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.train_history = checkpoint.get("train_history", {})
        print(f"[Reptile] 模型已从 {path} 加载")
