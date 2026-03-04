"""
元学习任务数据集构建
将用户的交互数据组织为 N-way K-shot 元学习任务
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class MetaTaskDataset(Dataset):
    """
    元学习任务数据集
    
    每个任务对应一个用户的冷启动场景：
    - Support Set: 用户的前 K 条交互（模拟少量已知行为）
    - Query Set: 用户的后续交互（用于评估适配效果）
    """
    
    def __init__(
        self,
        ratings: pd.DataFrame,
        num_items: int,
        support_size: int = 5,
        query_size: int = 10,
        cold_threshold: int = 5,
        mode: str = "train",  # train / cold_eval
        neg_ratio: int = 1,   # 负样本比例
        seed: int = 42,
    ):
        super().__init__()
        self.num_items = num_items
        self.support_size = support_size
        self.query_size = query_size
        self.cold_threshold = cold_threshold
        self.mode = mode
        self.neg_ratio = neg_ratio
        self.rng = np.random.RandomState(seed)
        
        # 按用户组织交互
        self.user_interactions = defaultdict(list)
        for _, row in ratings.iterrows():
            self.user_interactions[int(row["user_idx"])].append({
                "item_idx": int(row["item_idx"]),
                "label": float(row["label"]),
                "timestamp": int(row["timestamp"]),
            })
        
        # 按时间排序
        for uid in self.user_interactions:
            self.user_interactions[uid].sort(key=lambda x: x["timestamp"])
        
        # 划分用户
        user_counts = {uid: len(ints) for uid, ints in self.user_interactions.items()}
        
        if mode == "train":
            # 训练任务：交互数 > cold_threshold + query_size 的用户
            self.task_users = [
                uid for uid, cnt in user_counts.items()
                if cnt >= support_size + query_size
            ]
        elif mode == "cold_eval":
            # 冷启动评估：交互数 <= cold_threshold 的用户
            self.task_users = [
                uid for uid, cnt in user_counts.items()
                if cold_threshold < cnt < support_size + query_size + 5
            ]
            if len(self.task_users) == 0:
                # fallback: 使用交互较少的用户
                sorted_users = sorted(user_counts.items(), key=lambda x: x[1])
                self.task_users = [uid for uid, cnt in sorted_users[:500] if cnt >= support_size + 2]
        else:
            self.task_users = list(self.user_interactions.keys())
        
        # 构建全局正样本集（用于负采样）
        self.user_pos_items = defaultdict(set)
        for uid, ints in self.user_interactions.items():
            for it in ints:
                if it["label"] > 0:
                    self.user_pos_items[uid].add(it["item_idx"])
        
        print(f"[MetaTaskDataset] mode={mode}, #tasks={len(self.task_users)}, "
              f"support_size={support_size}, query_size={query_size}")
    
    def __len__(self):
        return len(self.task_users)
    
    def _sample_negatives(self, user_idx: int, num_neg: int) -> List[int]:
        """为用户采样负样本物品"""
        pos_items = self.user_pos_items[user_idx]
        neg_items = []
        while len(neg_items) < num_neg:
            item = self.rng.randint(0, self.num_items)
            if item not in pos_items:
                neg_items.append(item)
        return neg_items
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        返回一个元学习任务
        
        Returns:
            dict with keys:
            - support_users: [S]
            - support_items: [S]
            - support_labels: [S]
            - query_users: [Q]
            - query_items: [Q]
            - query_labels: [Q]
        """
        uid = self.task_users[index]
        interactions = self.user_interactions[uid]
        
        # 确保有足够交互
        n_total = min(len(interactions), self.support_size + self.query_size)
        n_support = min(self.support_size, n_total // 2)
        n_query = min(self.query_size, n_total - n_support)
        
        # Support: 前n_support条
        support_ints = interactions[:n_support]
        # Query: 之后的n_query条
        query_ints = interactions[n_support:n_support + n_query]
        
        # 构建 support set (包含正负样本)
        support_users, support_items, support_labels = [], [], []
        for it in support_ints:
            # 正样本
            support_users.append(uid)
            support_items.append(it["item_idx"])
            support_labels.append(it["label"])
            # 负样本
            for neg_item in self._sample_negatives(uid, self.neg_ratio):
                support_users.append(uid)
                support_items.append(neg_item)
                support_labels.append(0.0)
        
        # 构建 query set
        query_users, query_items, query_labels = [], [], []
        for it in query_ints:
            query_users.append(uid)
            query_items.append(it["item_idx"])
            query_labels.append(it["label"])
            for neg_item in self._sample_negatives(uid, self.neg_ratio):
                query_users.append(uid)
                query_items.append(neg_item)
                query_labels.append(0.0)
        
        return {
            "support_users": torch.LongTensor(support_users),
            "support_items": torch.LongTensor(support_items),
            "support_labels": torch.FloatTensor(support_labels),
            "query_users": torch.LongTensor(query_users),
            "query_items": torch.LongTensor(query_items),
            "query_labels": torch.FloatTensor(query_labels),
            "user_idx": uid,
        }


def meta_collate_fn(batch: List[Dict]) -> Dict[str, List[torch.Tensor]]:
    """
    自定义 collate 函数 - 不做 stack（每个任务大小可能不同）
    """
    collated = defaultdict(list)
    for sample in batch:
        for key, val in sample.items():
            collated[key].append(val)
    return dict(collated)


def create_meta_dataloaders(
    ratings: pd.DataFrame,
    num_items: int,
    support_size: int = 5,
    query_size: int = 10,
    cold_threshold: int = 5,
    meta_batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """创建训练和评估的元学习数据加载器"""
    
    train_dataset = MetaTaskDataset(
        ratings=ratings,
        num_items=num_items,
        support_size=support_size,
        query_size=query_size,
        cold_threshold=cold_threshold,
        mode="train",
        seed=seed,
    )
    
    eval_dataset = MetaTaskDataset(
        ratings=ratings,
        num_items=num_items,
        support_size=support_size,
        query_size=query_size,
        cold_threshold=cold_threshold,
        mode="cold_eval",
        seed=seed + 1,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=meta_batch_size,
        shuffle=True,
        collate_fn=meta_collate_fn,
        num_workers=num_workers,
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=meta_batch_size,
        shuffle=False,
        collate_fn=meta_collate_fn,
        num_workers=num_workers,
    )
    
    return train_loader, eval_loader
