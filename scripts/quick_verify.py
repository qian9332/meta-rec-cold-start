#!/usr/bin/env python3
"""快速验证脚本 - 验证所有模块可以正确import和基本运行"""
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import numpy as np
import pandas as pd

print("="*50)
print("🔧 快速验证 - Meta-Learning Cold-Start Rec")
print("="*50)

# 1. 验证模块导入
print("\n[1/6] 验证模块导入...")
from models.base_rec_model import BaseRecModel
from models.meta_embedding import MetaEmbeddingLayer, DecoupledMetaEmbedding
from models.layered_model import LayeredMetaRecModel, ReptilePretrainWrapper, ANILAdaptWrapper
from trainers.maml_trainer import MAMLTrainer
from trainers.reptile_trainer import ReptileTrainer
from trainers.anil_trainer import ANILTrainer
from utils.metrics import compute_auc, compute_hr_at_k, compute_ndcg_at_k
from utils.hessian_analysis import HessianAnalyzer
from utils.gradient_tools import GradientCompensator, decoupled_inner_update
print("  ✅ 所有模块导入成功")

# 2. 验证基础模型
print("\n[2/6] 验证 BaseRecModel...")
device = torch.device("cpu")
model = BaseRecModel(num_users=100, num_items=50, user_emb_dim=16, item_emb_dim=16, hidden_dims=[32, 16])
users = torch.randint(0, 100, (8,))
items = torch.randint(0, 50, (8,))
labels = torch.randint(0, 2, (8,)).float()
logits = model(users, items)
loss = model.compute_loss(users, items, labels)
print(f"  logits shape: {logits.shape}, loss: {loss.item():.4f}")

# 函数式前向
params = model.get_params_dict()
logits_fn = model(users, items, params=params)
print(f"  函数式前向 shape: {logits_fn.shape}")

# 块对角分析
analysis = model.analyze_block_diagonal_structure()
print(f"  总参数量: {analysis['total_params']:,}")
for g, info in analysis.items():
    if isinstance(info, dict) and 'ratio' in info:
        print(f"    {g}: {info['num_params']:,} ({info['ratio']:.1%})")
print("  ✅ BaseRecModel 验证通过")

# 3. 验证 Meta-Embedding
print("\n[3/6] 验证 Meta-Embedding...")
meta_emb = DecoupledMetaEmbedding(num_users=100, num_items=50, user_emb_dim=16, item_emb_dim=16)
u_emb, i_emb = meta_emb(users, items)
print(f"  user_emb: {u_emb.shape}, item_emb: {i_emb.shape}")
print(f"  解耦LR - emb: {meta_emb.lr_emb}, dense: {meta_emb.lr_dense}")
print("  ✅ Meta-Embedding 验证通过")

# 4. 验证 LayeredModel
print("\n[4/6] 验证 LayeredMetaRecModel...")
layered = LayeredMetaRecModel(num_users=100, num_items=50, user_emb_dim=16, item_emb_dim=16, hidden_dims=[32, 16])
logits_l = layered(users, items)
print(f"  logits shape: {logits_l.shape}")
print(f"  backbone params: {sum(p.numel() for p in layered.get_backbone_params()):,}")
print(f"  head params: {sum(p.numel() for p in layered.get_head_params()):,}")

# head-only前向
head_params = layered.get_head_state()
logits_h = layered(users, items, head_params=head_params)
print(f"  head-only前向: {logits_h.shape}")
print("  ✅ LayeredMetaRecModel 验证通过")

# 5. 验证MAML内循环
print("\n[5/6] 验证 MAML 内循环...")
model2 = BaseRecModel(num_users=100, num_items=50, user_emb_dim=16, item_emb_dim=16, hidden_dims=[32, 16])
params = {name: param.clone() for name, param in model2.named_parameters()}
loss = model2.compute_loss(users, items, labels, params=params)
grads = torch.autograd.grad(loss, params.values(), create_graph=False)
param_names = list(params.keys())

# 解耦更新
updated = decoupled_inner_update(params, grads, param_names, lr_emb=0.02, lr_dense=0.005)
print(f"  内循环更新完成, 参数数量: {len(updated)}")

# 对比一阶 vs 二阶
params_fo = {name: param.clone() for name, param in model2.named_parameters()}
loss_fo = model2.compute_loss(users, items, labels, params=params_fo)
grads_fo = torch.autograd.grad(loss_fo, params_fo.values(), create_graph=False)

params_so = {name: param.clone() for name, param in model2.named_parameters()}
loss_so = model2.compute_loss(users, items, labels, params=params_so)
grads_so = torch.autograd.grad(loss_so, params_so.values(), create_graph=True)

cos_sims = []
for g1, g2 in zip(grads_fo, grads_so):
    cos = torch.nn.functional.cosine_similarity(g1.flatten().unsqueeze(0), g2.flatten().unsqueeze(0))
    cos_sims.append(cos.item())
print(f"  一阶vs二阶 梯度余弦相似度: {np.mean(cos_sims):.6f}")
print("  ✅ MAML 内循环验证通过")

# 6. 验证评估指标
print("\n[6/6] 验证评估指标...")
preds = np.random.rand(100)
labs = np.random.randint(0, 2, 100)
auc = compute_auc(preds, labs)
hr = compute_hr_at_k(preds, labs, k=10)
ndcg = compute_ndcg_at_k(preds, labs, k=10)
print(f"  AUC={auc:.4f}, HR@10={hr:.4f}, NDCG@10={ndcg:.4f}")
print("  ✅ 评估指标验证通过")

# Summary
print("\n" + "="*50)
print("🎉 所有模块验证通过！项目代码完整可运行")
print("="*50)
print(f"\n项目路径: {project_root}")
print("运行完整实验: python scripts/run_full_pipeline.py")
