#!/usr/bin/env python3
"""ANIL在线适配脚本（需先运行Reptile预训练）"""
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from data.download_data import load_movielens_1m, preprocess_data
from data.dataset import create_meta_dataloaders
from models.layered_model import LayeredMetaRecModel
from trainers.reptile_trainer import ReptileTrainer
from trainers.anil_trainer import ANILTrainer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join(project_root, "data", "ml-1m")
    
    try:
        ratings, users, movies = load_movielens_1m(data_dir)
        ratings, users, movies, info = preprocess_data(ratings, users, movies)
        num_users, num_items = info["num_users"], info["num_items"]
    except Exception:
        from scripts.run_full_pipeline import generate_synthetic_data
        ratings, users, movies, info = generate_synthetic_data()
        num_users, num_items = info["num_users"], info["num_items"]
    
    train_loader, eval_loader = create_meta_dataloaders(
        ratings=ratings, num_items=num_items,
        support_size=5, query_size=5, meta_batch_size=8,
    )
    
    model = LayeredMetaRecModel(
        num_users=num_users, num_items=num_items,
        user_emb_dim=32, item_emb_dim=32, hidden_dims=[128, 64],
    )
    
    # 尝试加载Reptile预训练权重
    pretrained_path = os.path.join(project_root, "checkpoints", "reptile_pretrained.pt")
    if os.path.exists(pretrained_path):
        reptile_trainer = ReptileTrainer(model=model, device=device)
        reptile_trainer.load_pretrained(pretrained_path)
        print("[INFO] 已加载Reptile预训练权重")
    else:
        print("[WARN] 未找到Reptile预训练权重，使用随机初始化")
        print("[TIP] 请先运行: python scripts/run_reptile_pretrain.py")
    
    # ANIL适配
    anil_trainer = ANILTrainer(
        model=model, inner_lr=0.01, outer_lr=0.001, inner_steps=5, device=device,
    )
    
    anil_trainer.train(train_loader, eval_loader, num_epochs=5, eval_every=1)


if __name__ == "__main__":
    main()
