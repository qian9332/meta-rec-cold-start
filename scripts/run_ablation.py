#!/usr/bin/env python3
"""
第二轮优化 - 针对V2结果的进一步改进
=============================================
V2问题分析:
  1. 解耦LR在L2 norm后增益消失 → 需要去掉L2 norm做对照实验,
     验证解耦LR的真实效果
  2. Reptile+ANIL AUC仅0.58 → Reptile需要更多epoch、ANIL head学习率太小
  3. 解耦LR应该在"不带L2 norm"的场景下体现优势

新实验设计:
  A: FOMAML基线 (无L2 norm) → 复现梯度消失
  B: FOMAML + 解耦LR (无L2 norm) → 验证解耦LR修复梯度消失的效果
  C: FOMAML + L2 norm → V2的Stage3
  D: FOMAML + L2 norm + 解耦LR → V2的Stage4
  E: Reptile(更多step) + ANIL(更大head lr)
"""
import sys, os, time, json, pickle, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict, defaultdict
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, 'logs_v3')
os.makedirs(LOG_DIR, exist_ok=True)
np.random.seed(42); torch.manual_seed(42)
DEVICE = torch.device('cpu')


def get_logger(name='v3', log_file='full_training_v3.log', mode='a'):
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO); lg.handlers = []
    fp = os.path.join(LOG_DIR, log_file)
    for h in [logging.FileHandler(fp, mode, encoding='utf-8'), logging.StreamHandler(sys.stdout)]:
        h.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
        lg.addHandler(h)
    return lg


# ============================================================
# Data
# ============================================================
class ColdStartMetaDataset(Dataset):
    def __init__(self, user_interactions, num_items, support_size=5,
                 query_size=20, neg_ratio=3, mode='train', seed=42):
        self.num_items = num_items
        self.support_size = support_size
        self.query_size = query_size
        self.neg_ratio = neg_ratio
        self.rng = np.random.RandomState(seed)
        self.tasks = []
        self.user_pos = defaultdict(set)
        for uid, ints in user_interactions.items():
            for it in ints:
                if it['label'] > 0:
                    self.user_pos[uid].add(it['item_idx'])
            if len(ints) >= support_size + query_size // 2:
                self.tasks.append((uid, ints))

    def __len__(self): return len(self.tasks)

    def _neg(self, uid, n):
        pos = self.user_pos[uid]; negs = []
        while len(negs) < n:
            c = self.rng.randint(0, self.num_items)
            if c not in pos: negs.append(c)
        return negs

    def __getitem__(self, idx):
        uid, ints = self.tasks[idx]
        cutoff = min(len(ints), self.support_size + self.query_size)
        used = ints[:cutoff]
        s_ints = used[:self.support_size]
        q_ints = used[self.support_size:cutoff]
        su,si,sl = [],[],[]
        for it in s_ints:
            su.append(uid); si.append(it['item_idx']); sl.append(it['label'])
            for ni in self._neg(uid, self.neg_ratio):
                su.append(uid); si.append(ni); sl.append(0.0)
        qu,qi,ql = [],[],[]
        for it in q_ints:
            qu.append(uid); qi.append(it['item_idx']); ql.append(it['label'])
            for ni in self._neg(uid, self.neg_ratio):
                qu.append(uid); qi.append(ni); ql.append(0.0)
        return {
            'support_users': torch.LongTensor(su), 'support_items': torch.LongTensor(si),
            'support_labels': torch.FloatTensor(sl), 'query_users': torch.LongTensor(qu),
            'query_items': torch.LongTensor(qi), 'query_labels': torch.FloatTensor(ql),
            'user_idx': uid,
        }


def meta_collate(batch):
    c = defaultdict(list)
    for s in batch:
        for k, v in s.items(): c[k].append(v)
    return dict(c)


# ============================================================
# Model - 可配置是否使用L2 norm
# ============================================================
class RecModelV3(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=32,
                 hidden_dims=[128, 64], dropout=0.1, use_l2_norm=True):
        super().__init__()
        self.use_l2_norm = use_l2_norm
        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.item_embedding = nn.Embedding(num_items, emb_dim)
        nn.init.normal_(self.user_embedding.weight, 0, 0.1)
        nn.init.normal_(self.item_embedding.weight, 0, 0.1)

        input_dim = emb_dim * 2
        layers = []
        for i, hd in enumerate(hidden_dims):
            layers.append((f'fc_{i}', nn.Linear(input_dim, hd)))
            layers.append((f'ln_{i}', nn.LayerNorm(hd)))
            layers.append((f'relu_{i}', nn.ReLU()))
            layers.append((f'drop_{i}', nn.Dropout(dropout)))
            input_dim = hd
        self.backbone = nn.Sequential(OrderedDict(layers))
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, user_ids, item_ids, params=None):
        if params is not None:
            return self._func_forward(user_ids, item_ids, params)
        ue = self.user_embedding(user_ids)
        ie = self.item_embedding(item_ids)
        if self.use_l2_norm:
            ue = F.normalize(ue, p=2, dim=-1)
            ie = F.normalize(ie, p=2, dim=-1)
        x = torch.cat([ue, ie], dim=-1)
        x = self.backbone(x)
        return self.head(x).squeeze(-1)

    def _func_forward(self, user_ids, item_ids, params):
        ue = F.embedding(user_ids, params['user_embedding.weight'])
        ie = F.embedding(item_ids, params['item_embedding.weight'])
        if self.use_l2_norm:
            ue = F.normalize(ue, p=2, dim=-1)
            ie = F.normalize(ie, p=2, dim=-1)
        x = torch.cat([ue, ie], dim=-1)
        n = len([k for k in params if k.startswith('backbone.fc_') and k.endswith('.weight')])
        for i in range(n):
            x = F.linear(x, params[f'backbone.fc_{i}.weight'], params[f'backbone.fc_{i}.bias'])
            ln_w = params.get(f'backbone.ln_{i}.weight')
            ln_b = params.get(f'backbone.ln_{i}.bias')
            if ln_w is not None: x = F.layer_norm(x, [x.size(-1)], ln_w, ln_b)
            x = F.relu(x)
            if self.training: x = F.dropout(x, p=0.1)
        return F.linear(x, params['head.weight'], params['head.bias']).squeeze(-1)

    def compute_loss(self, u, i, l, params=None):
        return F.binary_cross_entropy_with_logits(self.forward(u, i, params=params), l)


# ANIL model
class RecModelANIL(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=32, hidden_dims=[128, 64],
                 head_dims=[32], dropout=0.1):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.item_embedding = nn.Embedding(num_items, emb_dim)
        nn.init.normal_(self.user_embedding.weight, 0, 0.1)
        nn.init.normal_(self.item_embedding.weight, 0, 0.1)
        input_dim = emb_dim * 2
        layers = []
        for i, hd in enumerate(hidden_dims):
            layers.append((f'fc_{i}', nn.Linear(input_dim, hd)))
            layers.append((f'ln_{i}', nn.LayerNorm(hd)))
            layers.append((f'relu_{i}', nn.ReLU()))
            layers.append((f'drop_{i}', nn.Dropout(dropout)))
            input_dim = hd
        self.backbone = nn.Sequential(OrderedDict(layers))
        head_layers = []
        hd_in = hidden_dims[-1]
        for j, hd in enumerate(head_dims):
            head_layers.append((f'head_fc_{j}', nn.Linear(hd_in, hd)))
            head_layers.append((f'head_relu_{j}', nn.ReLU()))
            hd_in = hd
        head_layers.append(('head_out', nn.Linear(hd_in, 1)))
        self.head = nn.Sequential(OrderedDict(head_layers))

    def forward(self, user_ids, item_ids, head_params=None):
        ue = F.normalize(self.user_embedding(user_ids), p=2, dim=-1)
        ie = F.normalize(self.item_embedding(item_ids), p=2, dim=-1)
        x = torch.cat([ue, ie], dim=-1)
        feat = self.backbone(x)
        if head_params is not None:
            n = len([k for k in head_params if k.startswith('head.head_fc_') and k.endswith('.weight')])
            for j in range(n):
                feat = F.linear(feat, head_params[f'head.head_fc_{j}.weight'],
                               head_params[f'head.head_fc_{j}.bias'])
                feat = F.relu(feat)
            return F.linear(feat, head_params['head.head_out.weight'],
                           head_params['head.head_out.bias']).squeeze(-1)
        return self.head(feat).squeeze(-1)

    def compute_loss(self, u, i, l, head_params=None):
        return F.binary_cross_entropy_with_logits(self.forward(u, i, head_params), l)

    def freeze_backbone(self):
        for n, p in self.named_parameters():
            if 'head' not in n: p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters(): p.requires_grad = True


# ============================================================
# Training helpers
# ============================================================
def decoupled_update(params, grads, pnames, lr_user=0.05, lr_item=0.02, lr_dense=0.005):
    updated = {}
    for (n, p), g in zip(zip(pnames, params.values()), grads):
        gn = g.norm()
        if gn > 1.0: g = g * (1.0 / gn)
        if 'user_embedding' in n: lr = lr_user
        elif 'item_embedding' in n: lr = lr_item
        else: lr = lr_dense
        updated[n] = p - lr * g
    return updated


def compute_auc(preds, labels):
    if len(np.unique(labels)) < 2: return None
    try: return roc_auc_score(labels, preds)
    except: return None

def compute_hr(p, l, k=5):
    if len(p) < k: k = len(p)
    return float(np.any(l[np.argsort(p)[::-1][:k]] > 0))

def compute_ndcg(p, l, k=5):
    if len(p) == 0 or np.sum(l) == 0: return 0.0
    if len(p) < k: k = len(p)
    tk = np.argsort(p)[::-1][:k]
    dcg = np.sum(l[tk] / np.log2(np.arange(2, k+2)))
    ideal = np.sort(l)[::-1][:k]
    idcg = np.sum(ideal / np.log2(np.arange(2, k+2)))
    return dcg / idcg if idcg > 0 else 0.0


def train_fomaml(model, opt, loader, inner_steps=3, inner_lr=0.01, use_dec=False):
    model.train(); losses = []
    for batch in loader:
        nt = len(batch['support_users']); ml = 0.0; v = 0
        for i in range(nt):
            su,si,sl = [batch[k][i].to(DEVICE) for k in ['support_users','support_items','support_labels']]
            qu,qi,ql = [batch[k][i].to(DEVICE) for k in ['query_users','query_items','query_labels']]
            if len(sl) < 4 or len(ql) < 4: continue
            params = {n: p.clone() for n, p in model.named_parameters()}
            pn = list(params.keys())
            for _ in range(inner_steps):
                l = model.compute_loss(su, si, sl, params=params)
                gs = torch.autograd.grad(l, params.values(), create_graph=False)
                if use_dec: params = decoupled_update(params, gs, pn)
                else: params = {n: p - inner_lr*g for (n,p),g in zip(params.items(), gs)}
            qloss = model.compute_loss(qu, qi, ql, params=params)
            ml += qloss; v += 1
        if v > 0:
            ml = ml/v; opt.zero_grad(); ml.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(ml.item())
    return np.mean(losses) if losses else 0


def eval_fomaml(model, loader, inner_steps=3, inner_lr=0.01, use_dec=False):
    model.eval(); aa,ah,an = [],[],[]
    for batch in loader:
        for i in range(len(batch['support_users'])):
            su,si,sl = [batch[k][i].to(DEVICE) for k in ['support_users','support_items','support_labels']]
            qu,qi,ql = [batch[k][i].to(DEVICE) for k in ['query_users','query_items','query_labels']]
            if len(sl) < 4 or len(ql) < 4: continue
            params = {n: p.clone() for n, p in model.named_parameters()}
            pn = list(params.keys())
            for _ in range(inner_steps):
                l = model.compute_loss(su, si, sl, params=params)
                gs = torch.autograd.grad(l, params.values(), create_graph=False)
                if use_dec: params = decoupled_update(params, gs, pn)
                else: params = {n: p - inner_lr*g for (n,p),g in zip(params.items(), gs)}
            with torch.no_grad():
                pred = torch.sigmoid(model(qu, qi, params=params)).cpu().numpy()
                lab = ql.cpu().numpy()
            auc = compute_auc(pred, lab)
            if auc is not None: aa.append(auc)
            ah.append(compute_hr(pred, lab, 5)); an.append(compute_ndcg(pred, lab, 5))
    return {'auc': np.mean(aa) if aa else 0.5, 'hr@5': np.mean(ah) if ah else 0, 'ndcg@5': np.mean(an) if an else 0}


def eval_anil(model, loader, inner_steps=5, inner_lr=0.02):
    model.eval(); aa,ah,an = [],[],[]
    for batch in loader:
        for i in range(len(batch['support_users'])):
            su,si,sl = [batch[k][i].to(DEVICE) for k in ['support_users','support_items','support_labels']]
            qu,qi,ql = [batch[k][i].to(DEVICE) for k in ['query_users','query_items','query_labels']]
            if len(sl) < 4 or len(ql) < 4: continue
            hp = {n: p.clone().detach().requires_grad_(True) for n,p in model.named_parameters() if 'head' in n}
            for _ in range(inner_steps):
                lo = model(su, si, head_params=hp)
                l = F.binary_cross_entropy_with_logits(lo, sl)
                g = torch.autograd.grad(l, hp.values())
                hp = {n: p - inner_lr*gr for (n,p),gr in zip(hp.items(), g)}
            with torch.no_grad():
                pred = torch.sigmoid(model(qu, qi, head_params=hp)).cpu().numpy()
                lab = ql.cpu().numpy()
            auc = compute_auc(pred, lab)
            if auc is not None: aa.append(auc)
            ah.append(compute_hr(pred, lab, 5)); an.append(compute_ndcg(pred, lab, 5))
    return {'auc': np.mean(aa) if aa else 0.5, 'hr@5': np.mean(ah) if ah else 0, 'ndcg@5': np.mean(an) if an else 0}


def load_data(n_train=100, n_eval=30):
    data_dir = os.path.join(ROOT, 'data', 'ml-1m')
    ratings = pd.read_csv(os.path.join(data_dir, 'ratings.dat'), sep='::',
                          header=None, names=['user_id','item_id','rating','timestamp'],
                          engine='python', encoding='latin-1')
    uid_map = {u: i for i, u in enumerate(sorted(ratings['user_id'].unique()))}
    iid_map = {i: j for j, i in enumerate(sorted(ratings['item_id'].unique()))}
    ratings['user_idx'] = ratings['user_id'].map(uid_map)
    ratings['item_idx'] = ratings['item_id'].map(iid_map)
    ratings['label'] = (ratings['rating'] >= 4).astype(np.float32)
    nu, ni = len(uid_map), len(iid_map)
    user_ints = defaultdict(list)
    for _, row in ratings.iterrows():
        user_ints[int(row['user_idx'])].append({
            'item_idx': int(row['item_idx']), 'label': float(row['label']),
            'timestamp': int(row['timestamp']),
        })
    for uid in user_ints:
        user_ints[uid].sort(key=lambda x: x['timestamp'])
    eligible = [(uid, ints) for uid, ints in user_ints.items() if 20 <= len(ints) <= 200]
    rng = np.random.RandomState(42)
    rng.shuffle(eligible)
    train_ints = {uid: ints for uid, ints in eligible[:n_train]}
    eval_ints = {uid: ints for uid, ints in eligible[n_train:n_train+n_eval]}
    return nu, ni, train_ints, eval_ints


def save_json(data, path):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o
    with open(path, 'w') as f: json.dump(data, f, indent=2, default=conv)


# ============================================================
# Main - 消融实验
# ============================================================
if __name__ == '__main__':
    stage = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mode = 'w' if stage == 0 else 'a'
    lg = get_logger(f'v3_s{stage}', 'full_training_v3.log', mode)

    nu, ni, train_ints, eval_ints = load_data(100, 30)
    tds = ColdStartMetaDataset(train_ints, ni, 5, 20, 3, 'train', 42)
    eds = ColdStartMetaDataset(eval_ints, ni, 5, 20, 3, 'eval', 43)
    tl = DataLoader(tds, batch_size=8, shuffle=True, collate_fn=meta_collate)
    el = DataLoader(eds, batch_size=8, shuffle=False, collate_fn=meta_collate)

    N_EPOCHS = 8

    if stage == 0:
        # ============ Exp A: FOMAML 无L2 norm (复现梯度消失) ============
        lg.info('╔══════════════════════════════════════════════════════════════╗')
        lg.info('║  V3 消融实验 - 验证各优化点的独立贡献                              ║')
        lg.info('╚══════════════════════════════════════════════════════════════╝')
        lg.info('')
        lg.info('=' * 65)
        lg.info('EXP-A: FOMAML 无L2 norm, 统一lr=0.01 (复现梯度消失基线)')
        lg.info('=' * 65)
        model = RecModelV3(nu, ni, 32, [128,64], 0.1, use_l2_norm=False).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=0.003)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
        hist = {'loss':[], 'auc':[], 'hr@5':[], 'ndcg@5':[], 'time':[]}
        for ep in range(1, N_EPOCHS+1):
            t0 = time.time()
            loss = train_fomaml(model, opt, tl, 3, 0.01, False)
            sched.step(); et = time.time()-t0
            ev = eval_fomaml(model, el, 3, 0.01, False)
            for k in hist: hist[k].append(ev.get(k, loss if k=='loss' else et) if k not in ['loss','time'] else (loss if k=='loss' else et))
            lg.info(f'  Ep {ep}/{N_EPOCHS} | loss={loss:.4f} | AUC={ev["auc"]:.4f} | HR@5={ev["hr@5"]:.4f} | NDCG@5={ev["ndcg@5"]:.4f} | {et:.1f}s')
        save_json(hist, os.path.join(LOG_DIR, 'expA.json'))
        lg.info(f'  ✅ Exp-A 最终AUC={hist["auc"][-1]:.4f}, best={max(hist["auc"]):.4f}')

        # 验证user_emb梯度
        model.train()
        sample = tds[0]
        su,si,sl = sample['support_users'].to(DEVICE), sample['support_items'].to(DEVICE), sample['support_labels'].to(DEVICE)
        loss = model.compute_loss(su, si, sl)
        grads = torch.autograd.grad(loss, model.parameters())
        for (n,_), g in zip(model.named_parameters(), grads):
            if 'embedding' in n:
                lg.info(f'  [梯度检查] {n}: grad_norm={g.norm().item():.6f}, sparsity={((g==0).float().mean().item()):.4f}')
        lg.info('')

    elif stage == 1:
        # ============ Exp B: FOMAML 无L2 norm + 解耦LR ============
        lg.info('=' * 65)
        lg.info('EXP-B: FOMAML 无L2 norm + 三级解耦LR (验证解耦对梯度消失的缓解)')
        lg.info('=' * 65)
        model = RecModelV3(nu, ni, 32, [128,64], 0.1, use_l2_norm=False).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=0.003)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
        hist = {'loss':[], 'auc':[], 'hr@5':[], 'ndcg@5':[], 'time':[]}
        for ep in range(1, N_EPOCHS+1):
            t0 = time.time()
            loss = train_fomaml(model, opt, tl, 3, 0.01, True)  # use_dec=True
            sched.step(); et = time.time()-t0
            ev = eval_fomaml(model, el, 3, 0.01, True)
            for k in hist: hist[k].append(ev.get(k, loss if k=='loss' else et) if k not in ['loss','time'] else (loss if k=='loss' else et))
            lg.info(f'  Ep {ep}/{N_EPOCHS} | loss={loss:.4f} | AUC={ev["auc"]:.4f} | HR@5={ev["hr@5"]:.4f} | NDCG@5={ev["ndcg@5"]:.4f} | {et:.1f}s')
        save_json(hist, os.path.join(LOG_DIR, 'expB.json'))
        lg.info(f'  ✅ Exp-B 最终AUC={hist["auc"][-1]:.4f}, best={max(hist["auc"]):.4f}')
        lg.info('')

    elif stage == 2:
        # ============ Exp C: FOMAML + L2 norm (V2基线) ============
        lg.info('=' * 65)
        lg.info('EXP-C: FOMAML + L2 norm, 统一lr=0.01 (V2基线)')
        lg.info('=' * 65)
        model = RecModelV3(nu, ni, 32, [128,64], 0.1, use_l2_norm=True).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=0.003)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
        hist = {'loss':[], 'auc':[], 'hr@5':[], 'ndcg@5':[], 'time':[]}
        for ep in range(1, N_EPOCHS+1):
            t0 = time.time()
            loss = train_fomaml(model, opt, tl, 3, 0.01, False)
            sched.step(); et = time.time()-t0
            ev = eval_fomaml(model, el, 3, 0.01, False)
            for k in hist: hist[k].append(ev.get(k, loss if k=='loss' else et) if k not in ['loss','time'] else (loss if k=='loss' else et))
            lg.info(f'  Ep {ep}/{N_EPOCHS} | loss={loss:.4f} | AUC={ev["auc"]:.4f} | HR@5={ev["hr@5"]:.4f} | NDCG@5={ev["ndcg@5"]:.4f} | {et:.1f}s')
        save_json(hist, os.path.join(LOG_DIR, 'expC.json'))
        lg.info(f'  ✅ Exp-C 最终AUC={hist["auc"][-1]:.4f}, best={max(hist["auc"]):.4f}')
        lg.info('')

    elif stage == 3:
        # ============ Exp D: FOMAML + L2 norm + 解耦LR ============
        lg.info('=' * 65)
        lg.info('EXP-D: FOMAML + L2 norm + 三级解耦LR')
        lg.info('=' * 65)
        model = RecModelV3(nu, ni, 32, [128,64], 0.1, use_l2_norm=True).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=0.003)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
        hist = {'loss':[], 'auc':[], 'hr@5':[], 'ndcg@5':[], 'time':[]}
        for ep in range(1, N_EPOCHS+1):
            t0 = time.time()
            loss = train_fomaml(model, opt, tl, 3, 0.01, True)
            sched.step(); et = time.time()-t0
            ev = eval_fomaml(model, el, 3, 0.01, True)
            for k in hist: hist[k].append(ev.get(k, loss if k=='loss' else et) if k not in ['loss','time'] else (loss if k=='loss' else et))
            lg.info(f'  Ep {ep}/{N_EPOCHS} | loss={loss:.4f} | AUC={ev["auc"]:.4f} | HR@5={ev["hr@5"]:.4f} | NDCG@5={ev["ndcg@5"]:.4f} | {et:.1f}s')
        save_json(hist, os.path.join(LOG_DIR, 'expD.json'))
        lg.info(f'  ✅ Exp-D 最终AUC={hist["auc"][-1]:.4f}, best={max(hist["auc"]):.4f}')
        lg.info('')

    elif stage == 4:
        # ============ Exp E: Reptile + ANIL (优化超参) ============
        lg.info('=' * 65)
        lg.info('EXP-E: Reptile(更多step, eps_decay) + ANIL(大head lr=0.02)')
        lg.info('=' * 65)
        model = RecModelANIL(nu, ni, 32, [128,64], [32], 0.1).to(DEVICE)
        hist = {'loss':[], 'auc':[], 'time':[]}

        # Phase 1: Reptile pretrain
        lg.info('  --- Phase 1: Reptile Pretrain ---')
        for ep in range(1, N_EPOCHS+1):
            model.train(); t0=time.time(); losses=[]
            eps = max(0.05, 0.3 * (0.85 ** (ep-1)))
            for batch in tl:
                for i in range(len(batch['support_users'])):
                    su = batch['support_users'][i].to(DEVICE)
                    si = batch['support_items'][i].to(DEVICE)
                    sl = batch['support_labels'][i].to(DEVICE)
                    if len(sl) < 4: continue
                    init = {n: p.data.clone() for n, p in model.named_parameters()}
                    iopt = torch.optim.SGD(model.parameters(), lr=0.02)
                    tl2 = 0
                    for _ in range(5):
                        iopt.zero_grad()
                        l = model.compute_loss(su, si, sl)
                        l.backward(); iopt.step(); tl2 += l.item()
                    with torch.no_grad():
                        for n,p in model.named_parameters():
                            p.data.copy_(init[n] + eps*(p.data - init[n]))
                    losses.append(tl2/5)
            et = time.time()-t0
            model.unfreeze_all()
            ev = eval_anil(model, el, 5, 0.02)
            hist['loss'].append(np.mean(losses)); hist['auc'].append(ev['auc']); hist['time'].append(et)
            lg.info(f'  Reptile Ep {ep}/{N_EPOCHS} | loss={np.mean(losses):.4f} | '
                    f'AUC={ev["auc"]:.4f} | HR@5={ev["hr@5"]:.4f} | eps={eps:.3f} | {et:.1f}s')

        torch.save(model.state_dict(), os.path.join(LOG_DIR, 'reptile_v3.pt'))
        lg.info(f'  Reptile best AUC={max(hist["auc"]):.4f}')

        # Phase 2: ANIL fine-tune
        lg.info('  --- Phase 2: ANIL Head Adaptation ---')
        model.freeze_backbone()
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.005)
        anil_hist = {'loss':[], 'auc':[], 'hr@5':[], 'ndcg@5':[], 'time':[]}

        for ep in range(1, N_EPOCHS+1):
            model.train(); t0=time.time(); losses=[]
            for batch in tl:
                nt = len(batch['support_users']); ml=0.0; v=0
                for i in range(nt):
                    su,si,sl = [batch[k][i].to(DEVICE) for k in ['support_users','support_items','support_labels']]
                    qu,qi,ql = [batch[k][i].to(DEVICE) for k in ['query_users','query_items','query_labels']]
                    if len(sl) < 4 or len(ql) < 4: continue
                    hp = {n: p.clone().detach().requires_grad_(True) for n,p in model.named_parameters() if 'head' in n}
                    for _ in range(5):
                        lo = model(su, si, head_params=hp)
                        l = F.binary_cross_entropy_with_logits(lo, sl)
                        g = torch.autograd.grad(l, hp.values())
                        hp = {n: p-0.02*gr for (n,p),gr in zip(hp.items(), g)}
                    qlo = model(qu, qi, head_params=hp)
                    ql_loss = F.binary_cross_entropy_with_logits(qlo, ql)
                    ml += ql_loss; v += 1
                if v > 0:
                    ml=ml/v; opt.zero_grad(); ml.backward(); opt.step()
                    losses.append(ml.item())
            et = time.time()-t0
            model.unfreeze_all()
            ev = eval_anil(model, el, 5, 0.02)
            model.freeze_backbone()
            anil_hist['loss'].append(np.mean(losses)); anil_hist['auc'].append(ev['auc'])
            anil_hist['hr@5'].append(ev['hr@5']); anil_hist['ndcg@5'].append(ev['ndcg@5'])
            anil_hist['time'].append(et)
            lg.info(f'  ANIL Ep {ep}/{N_EPOCHS} | loss={np.mean(losses):.4f} | '
                    f'AUC={ev["auc"]:.4f} | HR@5={ev["hr@5"]:.4f} | NDCG@5={ev["ndcg@5"]:.4f} | {et:.1f}s')

        save_json(anil_hist, os.path.join(LOG_DIR, 'expE.json'))
        lg.info(f'  ✅ Exp-E ANIL best AUC={max(anil_hist["auc"]):.4f}')
        lg.info('')

    elif stage == 5:
        # ============ 总结 ============
        lg.info('=' * 65)
        lg.info('V3 消融实验总结')
        lg.info('=' * 65)
        exps = {
            'expA': 'A: FOMAML (无L2 norm, 统一lr)',
            'expB': 'B: FOMAML (无L2 norm + 解耦LR)',
            'expC': 'C: FOMAML (L2 norm, 统一lr)',
            'expD': 'D: FOMAML (L2 norm + 解耦LR)',
            'expE': 'E: Reptile + ANIL (优化版)',
        }
        lg.info(f'  {"实验":<45} {"best AUC":>9} {"last AUC":>9}')
        lg.info(f'  {"-"*45} {"-"*9} {"-"*9}')
        results = {}
        for k, name in exps.items():
            p = os.path.join(LOG_DIR, f'{k}.json')
            if os.path.exists(p):
                with open(p) as f: h = json.load(f)
                ba = max(h.get('auc', [0])); la = h['auc'][-1] if h.get('auc') else 0
                lg.info(f'  {name:<45} {ba:>9.4f} {la:>9.4f}')
                results[k] = {'best': ba, 'last': la}

        lg.info('')
        lg.info('  📊 关键对比:')
        a_auc = results.get('expA', {}).get('best', 0)
        b_auc = results.get('expB', {}).get('best', 0)
        c_auc = results.get('expC', {}).get('best', 0)
        d_auc = results.get('expD', {}).get('best', 0)
        e_auc = results.get('expE', {}).get('best', 0)

        lg.info(f'  • 解耦LR在无L2 norm下: B-A = {b_auc-a_auc:+.4f} (解耦缓解梯度消失)')
        lg.info(f'  • L2 norm的效果: C-A = {c_auc-a_auc:+.4f} (L2 norm修复梯度消失)')
        lg.info(f'  • L2 norm下解耦额外增益: D-C = {d_auc-c_auc:+.4f}')
        lg.info(f'  • Reptile+ANIL vs 最佳FOMAML: E-max(C,D) = {e_auc-max(c_auc,d_auc):+.4f}')
        lg.info(f'  • 整体最佳方案: {"D (L2+解耦)" if d_auc >= c_auc else "C (仅L2)"} AUC={max(c_auc,d_auc):.4f}')
        lg.info('')
        lg.info('=' * 65)
        lg.info('✅ V3 消融实验全部完成!')
        lg.info('=' * 65)
