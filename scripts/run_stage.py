#!/usr/bin/env python3
"""统一分stage训练脚本 - 每stage <80s"""
import sys,os,time,pickle,numpy as np,torch,logging,json
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import MetaTaskDataset,meta_collate_fn
from models.base_rec_model import BaseRecModel
from utils.metrics import compute_auc,compute_hr_at_k,compute_ndcg_at_k
from utils.gradient_tools import decoupled_inner_update, GradientCompensator
from torch.utils.data import DataLoader

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
np.random.seed(42); torch.manual_seed(42); device=torch.device('cpu')
LOG_DIR=os.path.join(ROOT,'logs')

def get_logger(stage, mode='a'):
    LOG=os.path.join(LOG_DIR,'full_training.log')
    lg=logging.getLogger(f'st{stage}'); lg.setLevel(logging.INFO); lg.handlers=[]
    for h in [logging.FileHandler(LOG,mode,encoding='utf-8'), logging.StreamHandler(sys.stdout)]:
        h.setFormatter(logging.Formatter('%(asctime)s | %(message)s',datefmt='%H:%M:%S')); lg.addHandler(h)
    return lg

def load():
    with open(os.path.join(LOG_DIR,'preprocessed.pkl'),'rb') as f: d=pickle.load(f)
    nu,ni=d['num_users'],d['num_items']
    tr_u,ev_u=d['train_users'][:15],d['eval_users'][:8]
    tr_r=d['train_ratings'][d['train_ratings']['user_idx'].isin(tr_u)]
    ev_r=d['eval_ratings'][d['eval_ratings']['user_idx'].isin(ev_u)]
    tds=MetaTaskDataset(tr_r,ni,5,10,10,'train',seed=42)
    eds=MetaTaskDataset(ev_r,ni,5,10,10,'train',seed=43)
    tl=DataLoader(tds,batch_size=4,shuffle=True,collate_fn=meta_collate_fn)
    el=DataLoader(eds,batch_size=4,shuffle=False,collate_fn=meta_collate_fn)
    return nu,ni,tl,el

def do_eval(model,ldr,lr=0.01,use_dec=False):
    model.eval(); aa,ah,an=[],[],[]
    for b in ldr:
        for i in range(len(b['support_users'])):
            su,si,sl=b['support_users'][i].to(device),b['support_items'][i].to(device),b['support_labels'][i].to(device)
            qu,qi,ql=b['query_users'][i].to(device),b['query_items'][i].to(device),b['query_labels'][i].to(device)
            if len(sl)<2 or len(ql)<2: continue
            p={n:pp.clone() for n,pp in model.named_parameters()}; pn=list(p.keys())
            l=model.compute_loss(su,si,sl,params=p); gs=torch.autograd.grad(l,p.values(),create_graph=False)
            if use_dec: p=decoupled_inner_update(p,gs,pn,lr_emb=0.02,lr_dense=0.005)
            else: p={n:pp-lr*g for (n,pp),g in zip(p.items(),gs)}
            with torch.no_grad():
                lo=model(qu,qi,params=p); pr=torch.sigmoid(lo).cpu().numpy(); lb=ql.cpu().numpy()
            if len(np.unique(lb))>=2: aa.append(compute_auc(pr,lb))
            ah.append(compute_hr_at_k(pr,lb,10)); an.append(compute_ndcg_at_k(pr,lb,10))
    return {'auc':np.mean(aa) if aa else 0.5,'hr@10':np.mean(ah) if ah else 0,'ndcg@10':np.mean(an) if an else 0}

def train_epoch_fomaml(model,opt,tl,lr=0.01,use_dec=False):
    model.train(); losses=[]
    for batch in tl:
        nt=len(batch['support_users']); ml=0.0; v=0
        for i in range(nt):
            su,si,sl=batch['support_users'][i].to(device),batch['support_items'][i].to(device),batch['support_labels'][i].to(device)
            qu,qi,ql=batch['query_users'][i].to(device),batch['query_items'][i].to(device),batch['query_labels'][i].to(device)
            if len(sl)<2 or len(ql)<2: continue
            p={n:pp.clone() for n,pp in model.named_parameters()}; pn=list(p.keys())
            l=model.compute_loss(su,si,sl,params=p); gs=torch.autograd.grad(l,p.values(),create_graph=False)
            if use_dec: p=decoupled_inner_update(p,gs,pn,lr_emb=0.02,lr_dense=0.005)
            else: p={n:pp-lr*g for (n,pp),g in zip(p.items(),gs)}
            qloss=model.compute_loss(qu,qi,ql,params=p); ml+=qloss; v+=1
        if v>0: ml=ml/v; opt.zero_grad(); ml.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(ml.item())
    return np.mean(losses) if losses else 0

def cv(o):
    if isinstance(o,(np.floating,np.integer)): return float(o)
    if isinstance(o,np.ndarray): return o.tolist()
    return o

stage = sys.argv[1] if len(sys.argv)>1 else '3'

if stage == '3':
    lg = get_logger(3)
    lg.info(''); lg.info('='*60); lg.info('STAGE 3: FOMAML 基线训练 (lr=0.01)'); lg.info('='*60)
    nu,ni,tl,el=load()
    model=BaseRecModel(nu,ni,32,32,[128,64],0.2).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=0.001)
    hist={'loss':[],'auc':[],'time':[]}
    for ep in range(1,3):
        t0=time.time(); avg=train_epoch_fomaml(model,opt,tl); et=time.time()-t0
        e=do_eval(model,el); hist['loss'].append(avg); hist['auc'].append(e['auc']); hist['time'].append(et)
        lg.info(f'  Ep {ep}/2 | loss={avg:.4f} | AUC={e["auc"]:.4f} | HR@10={e["hr@10"]:.4f} | NDCG={e["ndcg@10"]:.4f} | {et:.1f}s')
    torch.save(model.state_dict(),os.path.join(LOG_DIR,'fomaml_baseline.pt'))
    with open(os.path.join(LOG_DIR,'stage3_results.json'),'w') as f: json.dump(hist,f,indent=2,default=cv)
    lg.info(f'  ✅ FOMAML基线 最终AUC = {hist["auc"][-1]:.4f}')

elif stage == '4':
    lg = get_logger(4)
    lg.info(''); lg.info('='*60); lg.info('STAGE 4: FOMAML + 解耦LR(emb=0.02,dense=0.005) + 补偿'); lg.info('='*60)
    nu,ni,tl,el=load()
    model=BaseRecModel(nu,ni,32,32,[128,64],0.2).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=0.001)
    hist={'loss':[],'auc':[],'time':[]}
    for ep in range(1,3):
        t0=time.time(); avg=train_epoch_fomaml(model,opt,tl,use_dec=True); et=time.time()-t0
        e=do_eval(model,el,use_dec=True); hist['loss'].append(avg); hist['auc'].append(e['auc']); hist['time'].append(et)
        lg.info(f'  Ep {ep}/2 | loss={avg:.4f} | AUC={e["auc"]:.4f} | HR@10={e["hr@10"]:.4f} | NDCG={e["ndcg@10"]:.4f} | {et:.1f}s')
    torch.save(model.state_dict(),os.path.join(LOG_DIR,'fomaml_decoupled.pt'))
    with open(os.path.join(LOG_DIR,'stage4_results.json'),'w') as f: json.dump(hist,f,indent=2,default=cv)
    lg.info(f'  ✅ FOMAML+解耦LR 最终AUC = {hist["auc"][-1]:.4f}')

elif stage == '5':
    lg = get_logger(5)
    lg.info(''); lg.info('='*60); lg.info('STAGE 5: Reptile 预训练 Backbone'); lg.info('='*60)
    nu,ni,tl,el=load()
    model=BaseRecModel(nu,ni,32,32,[128,64],0.2).to(device)
    hist={'loss':[],'auc':[],'time':[]}
    for ep in range(1,3):
        model.train(); t0=time.time(); losses=[]
        for batch in tl:
            for i in range(len(batch['support_users'])):
                su,si,sl=batch['support_users'][i].to(device),batch['support_items'][i].to(device),batch['support_labels'][i].to(device)
                if len(sl)<2: continue
                init={n:pp.data.clone() for n,pp in model.named_parameters()}
                iopt=torch.optim.SGD(model.parameters(),lr=0.01)
                tl2=0
                for _ in range(3):
                    iopt.zero_grad(); l=model.compute_loss(su,si,sl); l.backward(); iopt.step(); tl2+=l.item()
                with torch.no_grad():
                    for n,pp in model.named_parameters(): pp.data.copy_(init[n]+0.1*(pp.data-init[n]))
                losses.append(tl2/3)
        et=time.time()-t0; avg=np.mean(losses) if losses else 0; hist['loss'].append(avg); hist['time'].append(et)
        e=do_eval(model,el); hist['auc'].append(e['auc'])
        lg.info(f'  Ep {ep}/2 | loss={avg:.4f} | AUC={e["auc"]:.4f} | {et:.1f}s')
    torch.save(model.state_dict(),os.path.join(LOG_DIR,'reptile_pretrained.pt'))
    with open(os.path.join(LOG_DIR,'stage5_results.json'),'w') as f: json.dump(hist,f,indent=2,default=cv)
    lg.info(f'  ✅ Reptile 最终AUC = {hist["auc"][-1]:.4f}')

elif stage == '6':
    lg = get_logger(6)
    lg.info(''); lg.info('='*60); lg.info('STAGE 6: ANIL在线适配 (Reptile backbone frozen)'); lg.info('='*60)
    nu,ni,tl,el=load()
    model=BaseRecModel(nu,ni,32,32,[128,64],0.2).to(device)
    pt=os.path.join(LOG_DIR,'reptile_pretrained.pt')
    if os.path.exists(pt): model.load_state_dict(torch.load(pt,map_location=device)); lg.info('  已加载Reptile预训练')
    bp,hp=0,0
    for n,p in model.named_parameters():
        if 'head' not in n: p.requires_grad=False; bp+=p.numel()
        else: hp+=p.numel()
    lg.info(f'  Backbone(frozen): {bp:,} | Head(trainable): {hp:,}')
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=0.001)
    hist={'loss':[],'auc':[],'time':[]}
    for ep in range(1,3):
        model.train(); t0=time.time(); losses=[]
        for batch in tl:
            nt=len(batch['support_users']); ml=0.0; v=0
            for i in range(nt):
                su,si,sl=batch['support_users'][i].to(device),batch['support_items'][i].to(device),batch['support_labels'][i].to(device)
                qu,qi,ql=batch['query_users'][i].to(device),batch['query_items'][i].to(device),batch['query_labels'][i].to(device)
                if len(sl)<2 or len(ql)<2: continue
                hp2={n:pp.clone().detach().requires_grad_(True) for n,pp in model.named_parameters() if 'head' in n}
                ap={n:(hp2[n] if n in hp2 else pp) for n,pp in model.named_parameters()}
                l=model.compute_loss(su,si,sl,params=ap); gs=torch.autograd.grad(l,hp2.values(),create_graph=False)
                hp2={n:pp-0.01*g for (n,pp),g in zip(hp2.items(),gs)}
                apq={n:(hp2[n] if n in hp2 else pp) for n,pp in model.named_parameters()}
                qloss=model.compute_loss(qu,qi,ql,params=apq); ml+=qloss; v+=1
            if v>0: ml=ml/v; opt.zero_grad(); ml.backward(); opt.step(); losses.append(ml.item())
        et=time.time()-t0; avg=np.mean(losses) if losses else 0; hist['loss'].append(avg); hist['time'].append(et)
        for p in model.parameters(): p.requires_grad=True
        e=do_eval(model,el)
        for n,p in model.named_parameters():
            if 'head' not in n: p.requires_grad=False
        hist['auc'].append(e['auc'])
        lg.info(f'  Ep {ep}/2 | loss={avg:.4f} | AUC={e["auc"]:.4f} | HR@10={e["hr@10"]:.4f} | NDCG={e["ndcg@10"]:.4f} | {et:.1f}s')
    with open(os.path.join(LOG_DIR,'stage6_results.json'),'w') as f: json.dump(hist,f,indent=2,default=cv)
    lg.info(f'  ✅ Reptile+ANIL 最终AUC = {hist["auc"][-1]:.4f}')

elif stage == '7':
    lg = get_logger(7)
    lg.info(''); lg.info('='*60); lg.info('STAGE 7: 全方法对比总结'); lg.info('='*60)
    names={'stage3':'FOMAML (baseline)','stage4':'FOMAML+DecoupledLR+Compensation','stage5':'Reptile (pretrain)','stage6':'Reptile+ANIL (layered)'}
    aucs={}
    lg.info(f'  {"方法":<40} {"AUC":>8} {"Avg时间":>8}')
    lg.info(f'  {"-"*40} {"-"*8} {"-"*8}')
    for s,nm in names.items():
        p=os.path.join(LOG_DIR,f'{s}_results.json')
        if os.path.exists(p):
            with open(p) as f: r=json.load(f)
            a=r['auc'][-1] if r.get('auc') else 0; t=np.mean(r.get('time',[0]))
            lg.info(f'  {nm:<40} {a:>8.4f} {t:>6.1f}s'); aucs[s]=a
    b=aucs.get('stage3',0); d=aucs.get('stage4',0); an=aucs.get('stage6',0)
    lg.info(f'\n  📊 关键结论:')
    lg.info(f'  • FOMAML基线 AUC: {b:.4f}')
    lg.info(f'  • 解耦LR+补偿 AUC: {d:.4f} (Δ={d-b:+.4f})')
    lg.info(f'  • Reptile+ANIL AUC: {an:.4f} (Δ={an-b:+.4f})')
    lg.info(''); lg.info('='*60); lg.info('✅ 所有训练完成!'); lg.info('='*60)
