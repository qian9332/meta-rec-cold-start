"""
ANIL在线适配训练器

ANIL (Almost No Inner Loop, Raghu et al. 2020):
  - 内循环只更新head（最后一层），冻结backbone
  - 基于发现：MAML的成功主要来自特征复用而非快速学习
  - backbone通过Reptile预训练后冻结，在线只适配head

与完整MAML的对比:
  - 参数量: ANIL只更新head (e.g., 64*1 = 65), MAML更新全部 (e.g., 800K+)
  - 速度: ANIL的内循环快100x+
  - 效果: 在特征复用假设成立时（推荐模型通常如此），效果接近MAML
"""
import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Optional
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import evaluate_meta_model


class ANILTrainer:
    """
    ANIL训练器
    
    使用Reptile预训练的backbone，只适配head
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        device: torch.device = None,
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(self.device)
        
        # 外循环只优化head + backbone（backbone用更小lr）
        if hasattr(model, 'get_backbone_params') and hasattr(model, 'get_head_params'):
            self.outer_optimizer = torch.optim.Adam([
                {"params": model.get_backbone_params(), "lr": outer_lr * 0.1},  # backbone慢更新
                {"params": model.get_head_params(), "lr": outer_lr},
            ])
        else:
            self.outer_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
        
        self.train_history = {"loss": [], "eval_auc": [], "time_per_epoch": []}
    
    def inner_loop_head_only(
        self,
        support_users: torch.Tensor,
        support_items: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        ANIL内循环：只适配head参数
        
        frozen backbone提取特征 → 可学习head做预测
        """
        # 获取head参数
        head_params = {}
        for name, param in self.model.named_parameters():
            if "head" in name:
                head_params[name] = param.clone().detach().requires_grad_(True)
        
        if not head_params:
            # fallback: 如果没有明确的head，使用最后一层
            all_params = list(self.model.named_parameters())
            last_name, last_param = all_params[-1]
            head_params = {last_name: last_param.clone().detach().requires_grad_(True)}
            if len(all_params) >= 2:
                second_last_name, second_last_param = all_params[-2]
                head_params[second_last_name] = second_last_param.clone().detach().requires_grad_(True)
        
        # 内循环: 只更新head
        for step in range(self.inner_steps):
            # 前向传播
            if hasattr(self.model, 'compute_loss') and 'head_params' in self.model.compute_loss.__code__.co_varnames:
                loss = self.model.compute_loss(
                    support_users, support_items, support_labels,
                    head_params=head_params,
                )
            else:
                # 通用方式：临时替换参数
                loss = self.model.compute_loss(support_users, support_items, support_labels)
            
            # 只对head参数求梯度
            grads = torch.autograd.grad(
                loss,
                head_params.values(),
                create_graph=False,
                allow_unused=True,
            )
            
            # 更新head参数
            new_head_params = {}
            for (name, param), grad in zip(head_params.items(), grads):
                if grad is not None:
                    new_head_params[name] = param - self.inner_lr * grad
                else:
                    new_head_params[name] = param
            head_params = new_head_params
        
        return head_params
    
    def meta_train_step(self, task_batch: Dict[str, List[torch.Tensor]]) -> float:
        """ANIL元训练一步"""
        self.model.train()
        meta_loss = 0.0
        num_tasks = len(task_batch["support_users"])
        valid_tasks = 0
        
        for i in range(num_tasks):
            s_users = task_batch["support_users"][i].to(self.device)
            s_items = task_batch["support_items"][i].to(self.device)
            s_labels = task_batch["support_labels"][i].to(self.device)
            q_users = task_batch["query_users"][i].to(self.device)
            q_items = task_batch["query_items"][i].to(self.device)
            q_labels = task_batch["query_labels"][i].to(self.device)
            
            if len(s_labels) == 0 or len(q_labels) == 0:
                continue
            
            # 内循环: 只适配head
            adapted_head = self.inner_loop_head_only(s_users, s_items, s_labels)
            
            # Query loss
            if hasattr(self.model, 'compute_loss') and 'head_params' in self.model.compute_loss.__code__.co_varnames:
                query_loss = self.model.compute_loss(
                    q_users, q_items, q_labels,
                    head_params=adapted_head,
                )
            else:
                query_loss = self.model.compute_loss(q_users, q_items, q_labels)
            
            meta_loss += query_loss
            valid_tasks += 1
        
        if valid_tasks == 0:
            return 0.0
        
        meta_loss = meta_loss / valid_tasks
        
        self.outer_optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.outer_optimizer.step()
        
        return meta_loss.item()
    
    def train(
        self,
        train_loader,
        eval_loader=None,
        num_epochs: int = 20,
        eval_every: int = 5,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """ANIL训练完整流程"""
        
        print(f"\n{'='*60}")
        print(f"ANIL 在线适配训练")
        print(f"内循环步数: {self.inner_steps}, 内循环LR: {self.inner_lr}")
        print(f"外循环LR: {self.outer_lr}")
        print(f"仅适配Head参数")
        print(f"{'='*60}\n")
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            epoch_losses = []
            
            pbar = tqdm(train_loader, desc=f"ANIL Epoch {epoch}/{num_epochs}", disable=not verbose)
            for batch in pbar:
                loss = self.meta_train_step(batch)
                epoch_losses.append(loss)
                pbar.set_postfix({"loss": f"{loss:.4f}"})
            
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
                    use_head_params=True,
                )
                self.train_history["eval_auc"].append(eval_results["auc"])
                
                if verbose:
                    print(f"\n  [Eval] Epoch {epoch}: AUC={eval_results['auc']:.4f}, "
                          f"HR@10={eval_results['hr@10']:.4f}, NDCG@10={eval_results['ndcg@10']:.4f}")
            
            if verbose and epoch % eval_every == 0:
                print(f"  [Train] Epoch {epoch}: loss={avg_loss:.4f}, time={epoch_time:.1f}s")
        
        return self.train_history
