"""Fresh fitting, full-state resume and per-target incremental evaluation."""
import gc,json,math,time
from pathlib import Path
import numpy as np
import torch
from v5.common import atomic,read,sha,digest,scientific_identity
from v5.models import build,objective,predict_logits,entropy

def finite(x):
    if isinstance(x,dict):return {str(k):finite(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [finite(v) for v in x]
    if isinstance(x,(float,np.floating)) and not math.isfinite(float(x)):return None
    if isinstance(x,np.generic):return x.item()
    return x

def check(deadline):
    from utils.runtime_budget import BudgetExpired
    if time.time()>=deadline:raise BudgetExpired('Cooperative session deadline; resumable state preserved')

def loaders(cfg,role):
    from full_worker import loaders as original
    return original(cfg,role)

def is_bayesian(cfg):return cfg['v5']['variant'].startswith('bayes') or cfg['v5']['variant']=='balora_v1'

def infer(model,cfg,role,root,deadline,setting=None,limit_indices=None,diagnostics=True):
    """No optimizer, fixed oracle points; complete means EXACT audited key coverage."""
    from inference_common import seed_for,save_arrays
    from utils.prompting import sample_foreground_points
    from training.train import _clear_adapter_graphs
    from evaluation.point_metrics import compute_point_metrics
    from auditlib.pixel_metrics import summarize_prediction
    from utils.checkpoint import _rng_state,_restore_rng
    root=Path(root);root.mkdir(parents=True,exist_ok=True);ds=loaders(cfg,role).dataset
    indices=list(range(len(ds))) if limit_indices is None else list(limit_indices)
    setting=setting or {};default_k=3 if cfg['v5']['variant']=='det_ensemble' else (4 if is_bayesian(cfg) else 1)
    k=int(setting.get('k',default_k));probes=int(setting.get('probes',2));draw=int(setting.get('prompt_draw',0))
    rng=_rng_state();model.eval();base=(90043 if role=='validation' else 700043)+draw*1000003
    # Hash raw NIfTI bytes. Cache is outside the scientific source and bound to bytes.
    data_sha={c:{r:sha(p) for r,p in ds.pairs[c].items() if r in ('image','label')} for c in ds.case_ids}
    from inference_common import fingerprint
    ident={'config':scientific_identity(cfg),'weights':fingerprint(model),'data':data_sha,'role':role,'indices':indices,'setting':setting,'base_prompt_seed':base,'version':'v5_infer1'}
    fp=digest(ident);old=read(root/'identity.json')
    if old and old['fingerprint']!=fp:raise ValueError('Inference resume identity differs; choose a separate output directory')
    atomic(root/'identity.json',{'fingerprint':fp,'identity':ident})
    rows=[];expected=set();qual={}
    for cid in ds.case_ids:
        _,lab=ds._read_volume(cid)
        for organ in (6,11,12):
            area=np.count_nonzero(np.asarray(lab)==organ,axis=(0,1))
            if area.max()>0:qual[f'{cid}:{int(area.argmax())}:{organ}']=True
    started=time.perf_counter();peak=0
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad():
            for i in indices:
                check(deadline);s=ds[i];cid=s.case_id;z=int(s.slice_indices[0]);pending=[]
                for organ in range(1,14):
                    if not bool(s.label[0,organ].any()):continue
                    key=f'{cid}:{z}:{organ}';expected.add(key);p=root/'records'/(key.replace(':','_')+'.json')
                    if p.is_file():
                        row=read(p)
                        if row['fingerprint']!=fp:raise ValueError('Record fingerprint mismatch')
                        if row.get('array_file') and sha(root/row['array_file'])!=row['array_sha256']:raise ValueError('Array checksum mismatch')
                        rows.append(row)
                    else:pending.append((organ,key,p))
                if not pending:continue
                image=s.image.to(next(model.parameters()).device)
                if torch.cuda.is_available():torch.cuda.synchronize()
                t=time.perf_counter();embedding=model.encode_image(image)
                if torch.cuda.is_available():torch.cuda.synchronize()
                enc_seconds=time.perf_counter()-t
                for organ,key,p in pending:
                    check(deadline);target=s.label[:,organ].to(image.device)
                    # Match v4's case/target RNG stream. K never changes the point,
                    # which is sampled before all posterior draws. Probe-count changes
                    # also change subsequent posterior draws; report this coupling.
                    torch.manual_seed(seed_for(base,cid,z,organ));points=sample_foreground_points(target)
                    pl=torch.ones((1,1),dtype=torch.int32,device=image.device)
                    if torch.cuda.is_available():torch.cuda.synchronize()
                    t=time.perf_counter();logits,ranks,_=predict_logits(model,embedding,points,pl,k=k,probes=probes,forced_rank=setting.get('rank'))
                    probs=logits.float().sigmoid();prob=probs.mean(0)[0,0].detach();mi=None
                    if is_bayesian(cfg) or cfg['v5']['variant']=='det_ensemble':mi=(entropy(probs.mean(0))-entropy(probs).mean(0)).clamp_min(0)[0,0].detach()
                    if torch.cuda.is_available():torch.cuda.synchronize()
                    dec_seconds=time.perf_counter()-t
                    gt=target[0].cpu();pp=prob.cpu();seg,cal,empty=compute_point_metrics(pp,gt,surface=role=='test' or bool(setting.get('surface_profile')))
                    row={'fingerprint':fp,'key':key,'case_id':cid,'slice_index':z,'organ_class':organ,'prompt_xy':[int(x) for x in points[0,0].cpu().tolist()],
                         'seg':seg.to_dict(),'cal':cal.to_dict(),'dice':seg.dice,'empty_prediction':empty,'ranks':ranks,'encoder_seconds':enc_seconds,'decode_seconds':dec_seconds,
                         'epistemic_available':mi is not None,'prediction_samples_or_independent_members':k,'surface_units':'resized 2D pixels','timing_scope':'synchronized encoder once per slice and decoder per prompt including allocator probes; excludes CPU metrics and IO'}
                    if diagnostics:
                        row['diagnostics']=summarize_prediction(pp.numpy(),gt.numpy(),None if mi is None else mi.cpu().numpy())
                        if cfg['v5']['variant']=='det_ensemble':row['diagnostics']['ranking']['between_seed_MI']=row['diagnostics']['ranking'].pop('epistemic_MI')
                    row['uncertainty_kind']='between_independent_training_seeds' if cfg['v5']['variant']=='det_ensemble' else ('Gaussian posterior draws' if is_bayesian(cfg) else 'predictive entropy only')
                    if role=='test' and key in qual and not setting:
                        ap=root/'arrays'/(key.replace(':','_')+'.npz');row['array_file']=str(ap.relative_to(root))
                        row['array_sha256']=save_arrays(ap,s.image[0,0].numpy(),gt.numpy(),pp.numpy(),None if mi is None else mi.cpu().numpy(),{'key':key,'prompt_xy':row['prompt_xy'],'fingerprint':fp,'uncertainty_kind':row['uncertainty_kind'],'selection':'maximum raw GT area liver/pancreas/right adrenal per case, fixed independently of model performance'})
                    atomic(p,finite(row));rows.append(finite(row));_clear_adapter_graphs(model)
                    del logits,probs,prob,mi,pp,gt
                del embedding,image
                if (i+1)%25==0:print(role,'slice',i+1,'records',len(rows),flush=True)
        if len(rows)!=len(expected) or {r['key'] for r in rows}!=expected:raise ValueError('Coverage mismatch')
        groups={}
        for r in rows:groups.setdefault((r['case_id'],r['organ_class']),[]).append(r['dice'])
        result={'complete':True,'mean_dice':float(np.mean([np.mean(v) for v in groups.values()])),
                'records':len(rows),'foreground_slices':len(indices),'volumes':len(ds.case_ids),'volume_organ_pairs':len(groups),'fingerprint':fp,
                'elapsed_seconds_this_call':time.perf_counter()-started,'aggregation':'slice mean per volume-organ, equal mean across pairs','setting':setting,
                'bounded_profile':limit_indices is not None,'stored_trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'trainable_tensor_count':sum(1 for p in model.parameters() if p.requires_grad)}
        if torch.cuda.is_available():result['peak_reserved_bytes_this_call']=int(torch.cuda.max_memory_reserved())
        atomic(root/'status.json',result);return result
    finally:_restore_rng(rng);gc.collect()

def fit(cfg,home,deadline):
    from torch.utils.data import DataLoader
    from training.train import train_one_epoch,_make_grad_scaler
    from losses.losses import LossWeights
    from utils.seed import set_seed
    from utils.checkpoint import save_checkpoint,load_checkpoint
    home=Path(home);research=home/'research';research.mkdir(parents=True,exist_ok=True)
    device=cfg['experiment'].get('device','cuda:0')
    set_seed(cfg['experiment']['seed']);model=build(cfg,device)
    train=loaders(cfg,'train');data_sha={c:{r:sha(p) for r,p in train.dataset.pairs[c].items() if r in ('image','label')} for c in train.dataset.case_ids}
    ident=digest({'scientific':scientific_identity(cfg),'train_data':data_sha})
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20);scaler=_make_grad_scaler(str(device).startswith('cuda'))
    state={'identity':ident,'epoch':0,'phase':'train','cursor':0,'progress':{},'history':[],'best_dice':-1.,'best_epoch':None,'patience':0,'complete':False}
    last=research/'last.pt';status=research/'training_status.json'
    if last.is_file():
        payload=torch.load(last,map_location='cpu',weights_only=False)
        if payload['v5_state']['identity']!=ident:raise ValueError('Training resume identity/data mismatch')
        load_checkpoint(model,optimizer,str(last),scheduler=scheduler,restore_rng=True);scaler.load_state_dict(payload['scaler_state_dict']);state=payload['v5_state']
    def save(path=last):
        save_checkpoint(model,optimizer,state['epoch'],str(path),extra={'config':cfg,'v5_state':state,'scaler_state_dict':scaler.state_dict()},scheduler=scheduler,adapter_only=True)
        atomic(status,finite(state))
    weights=LossWeights(**{k:cfg['loss'][k] for k in ('dice_weight','bce_weight','lambda_kl','lambda_cal','lambda_rank')})
    last_saved=time.time();batch_k=2 if is_bayesian(cfg) else 1
    def callback(progress):
        nonlocal last_saved
        state['progress']=dict(progress);state['cursor']=progress['next_batch']
        if time.time()-last_saved>=180:save();last_saved=time.time()
    try:
        while not state['complete']:
            check(deadline);e=state['epoch'];start=time.time()
            if state['phase']=='train':
                g=torch.Generator().manual_seed(cfg['experiment']['seed']+e*1009);order=torch.randperm(len(train.dataset),generator=g).tolist()
                remaining=order[state['cursor']:]
                if remaining:
                    loader=DataLoader(train.dataset,batch_size=1,sampler=remaining,num_workers=0,collate_fn=train.collate_fn,generator=torch.Generator().manual_seed(cfg['experiment']['seed']+e))
                    loss=train_one_epoch(model,loader,optimizer,weights,batch_k,device,epoch=e,log_every=100,amp=str(device).startswith('cuda'),scaler=scaler,deadline=deadline,progress_state=state['progress'],progress_callback=callback,start_step=state['cursor'],objective_fn=objective)
                    state['loss']=loss
                else:state['loss']=state['progress']['total_loss']/state['progress']['updates']
                if state['progress']['updates']!=len(train.dataset):raise ValueError('Not all foreground training slices received an update')
                state['phase']='validation';save()
            report=infer(model,cfg,'validation',home/'validation'/f'epoch_{e:02d}',deadline,diagnostics=False)
            score=report['mean_dice'];improved=score>state['best_dice']
            state['history'].append({'epoch_one_based':e+1,'train_loss':state['loss'],'val_dice':score,'epoch_seconds_this_session_part':time.time()-start,'updates':state['progress']['updates'],'amp_retries':state['progress'].get('amp_overflow_retries',0)})
            if improved:state.update(best_dice=score,best_epoch=e+1,patience=0);save(research/'best.pt')
            else:state['patience']+=1
            scheduler.step();state.update(epoch=e+1,phase='train',cursor=0,progress={})
            state['complete']=state['epoch']>=20 or (state['epoch']>=8 and state['patience']>=6)
            save();print(cfg['v5']['id'],'epoch',e+1,'val Dice',score,'selected epoch',state['best_epoch'],flush=True)
        state['selected_checkpoint_sha256']=sha(research/'best.pt');atomic(status,finite(state));return state
    finally:save()

def profile(cfg,home,deadline):
    """Disposable real-SAM2 updates; never reused for research fitting."""
    from torch.utils.data import DataLoader,Subset
    from training.train import train_one_epoch,_make_grad_scaler
    from training.deterministic_control import balanced_indices
    from losses.losses import LossWeights
    from utils.seed import set_seed
    import shutil
    shutil.rmtree(Path(home)/'profile_validation',ignore_errors=True)
    set_seed(cfg['experiment']['seed']);model=build(cfg,'cuda:0');train=loaders(cfg,'train');val=loaders(cfg,'validation')
    indices=balanced_indices(train.dataset);optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay']);scaler=_make_grad_scaler(True)
    weights=LossWeights(**{k:cfg['loss'][k] for k in ('dice_weight','bce_weight','lambda_kl','lambda_cal','lambda_rank')})
    before={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad};times=[];torch.cuda.reset_peak_memory_stats()
    for j,i in enumerate([indices[0],*indices]):
        check(deadline);ds=Subset(train.dataset,[i]);ds.total_clips_before_filter=train.dataset.total_clips_before_filter
        loader=DataLoader(ds,batch_size=1,num_workers=0,collate_fn=train.collate_fn)
        torch.cuda.synchronize();t=time.perf_counter();train_one_epoch(model,loader,optimizer,weights,2 if is_bayesian(cfg) else 1,'cuda:0',amp=True,scaler=scaler,deadline=deadline,objective_fn=objective,log_every=10000)
        torch.cuda.synchronize()
        if j:times.append(time.perf_counter()-t)
    changed=[n for n,p in model.named_parameters() if p.requires_grad and not torch.equal(before[n],p.detach().cpu())]
    if not changed:raise ValueError('Smoke: no trainable parameter changed')
    if cfg['v5']['variant']!='point_unet' and (not any('image_encoder' in n for n in changed) or not any('sam_mask_decoder' in n for n in changed)):raise ValueError('Smoke: both encoder and decoder adapters must update')
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters() if p.requires_grad):raise FloatingPointError('Smoke: non-finite parameters')
    if 'adaptive_v2' in cfg['v5']['variant'] and not any('v2_gate' in n for n in changed):raise ValueError('Corrected gate did not update')
    peak=int(torch.cuda.max_memory_reserved());small=balanced_indices(val.dataset,3)
    report=infer(model,cfg,'validation',Path(home)/'profile_validation',deadline,setting={'surface_profile':True},limit_indices=small,diagnostics=True)
    epoch=float(np.quantile(times,.75))*len(train.dataset)+report['elapsed_seconds_this_call']/len(small)*len(val.dataset)
    result={'complete':True,'disposable_weights_discarded':True,'research_epochs':0,'train_p75_seconds':float(np.quantile(times,.75)),
        'sample_training_seconds':times,'train_slices':len(train.dataset),'validation_slices':len(val.dataset),'estimated_epoch_seconds':epoch,
        'estimated_full_test_diagnostic_seconds':report['elapsed_seconds_this_call']/len(small)*len(loaders(cfg,'test').dataset)*1.35,
        'peak_reserved_training_bytes':peak,'gpu':torch.cuda.get_device_name(),'changed_tensors':changed}
    atomic(Path(home)/'profile.json',result);return result
