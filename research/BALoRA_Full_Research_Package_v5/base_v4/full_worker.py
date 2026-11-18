"""Isolated workers for raw-data audit, disposable profiles, fitting and inference."""
from pathlib import Path
import argparse,copy,gc,json,math,sys,time,traceback
import numpy as np
from auditlib.io_budget import atomic_json,digest_file,digest_json
from protocol import split_ids

def finite(value):
    if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite(v) for v in value]
    if isinstance(value,(float,np.floating)) and not math.isfinite(float(value)):return None
    if isinstance(value,np.generic):return value.item()
    return value

def check(job):
    from utils.runtime_budget import BudgetExpired
    if time.time()>=job['deadline']:raise BudgetExpired('Full-run worker deadline')

def loaders(cfg,role):
    from datasets.btcv_dataset import BTCVDataset,collate_btcv
    from torch.utils.data import DataLoader
    d=cfg['data'];path=d[{'train':'train_split','validation':'val_split','test':'test_split'}[role]]
    ds=BTCVDataset(d['root_dir'],path,14,1,role=='train',1024,tuple(d['ct_window']),d['cache_dir'],True)
    return DataLoader(ds,batch_size=1,shuffle=False,num_workers=0,collate_fn=collate_btcv)

def make_model(cfg,method,checkpoint=None):
    if method=='balora':
        from training.train import build_model
        model=build_model(cfg,'cuda:0')
    elif method=='deterministic':
        from models.deterministic_sam2 import DeterministicSAM2
        model=DeterministicSAM2(cfg,'cuda:0')
    else:
        from models.frozen_sam2 import FrozenSAM2
        model=FrozenSAM2(cfg,'cuda:0')
    if checkpoint:
        from utils.checkpoint import load_checkpoint
        load_checkpoint(model,None,str(checkpoint),map_location='cpu',restore_rng=False)
    return model

def audit_data(job,cfg):
    output=Path(job['output_root']);summary={'complete':False,'splits':{},'data_sha256':{}}
    from datasets.splits import discover_btcv_pairs
    pairs=discover_btcv_pairs(cfg['data']['root_dir'])
    wanted=set(sum(split_ids().values(),[]))
    if set(pairs)!=wanted:raise ValueError('Expected exactly the 30 specified BTCV cases; set btcv_root to one dataset copy')
    test_keys=[];qualitative=[]
    for role,count,slices in [('train',19,1347),('validation',5,371),('test',6,460)]:
        check(job);loader=loaders(cfg,role);ds=loader.dataset
        if len(ds.case_ids)!=count or set(ds.case_ids)!=set(split_ids()[role]) or len(ds)!=slices:
            raise ValueError('The full Fold-1 '+role+' coverage differs from the fixed protocol')
        organ_counts={};slice_count=0
        for cid,pair in ds.pairs.items():
            check(job)
            for kind in ['image','label']:
                summary['data_sha256'].setdefault(cid,{})[kind]=digest_file(pair[kind])
            _,lab=ds._read_volume(cid);ds._validate_labels(lab,cid)
            # Ground-truth anatomy defines the example rule before model fitting.
            for organ in range(1,14):
                area=np.count_nonzero(np.asarray(lab)==organ,axis=(0,1))
                zs=np.flatnonzero(area)
                organ_counts[cid+':'+str(organ)]=int(len(zs))
                if role=='test':
                    test_keys.extend(f'{cid}:{int(z)}:{organ}' for z in zs)
                    if organ in (6,11,12) and len(zs):
                        qualitative.append(f'{cid}:{int(np.argmax(area))}:{organ}')
            slice_count+=int(np.any(np.asarray(lab)>0,axis=(0,1)).sum())
        summary['splits'][role]={'cases':ds.case_ids,'foreground_slices':slice_count,
            'organ_slice_counts':organ_counts,'dataset_index':ds._clip_index,
            'raw_slices':ds.total_clips_before_filter,'image_size':1024}
        del loader,ds;gc.collect()
    if len(test_keys)!=2466 or len(set(test_keys))!=2466:raise ValueError('Expected all 2,466 test slice-organ targets')
    summary.update(complete=True,expected_test_keys=sorted(test_keys),qualitative_keys=sorted(qualitative),
        qualitative_selection='Before fitting: maximum raw ground-truth area slice of liver, pancreas and right adrenal in each test volume; no score-based selection',
        independent_sampling_units='six volumes, subject linkage unverified',input_root=cfg['data']['root_dir'])
    previous=output/'data_audit.json'
    if previous.is_file() and json.loads(previous.read_text())['data_sha256']!=summary['data_sha256']:
        raise ValueError('Raw image/label bytes changed across sessions; do not resume fitted state on a changed dataset')
    atomic_json(output/'data_audit.json',summary)
    return summary

def profile(job,cfg):
    import torch
    from torch.utils.data import DataLoader,Subset
    from training.train import train_one_epoch,_make_grad_scaler
    from training.deterministic_control import balanced_indices,control_weights,deterministic_objective
    from losses.losses import LossWeights
    from evaluation.evaluate import run_evaluation
    from evaluation.point_estimate import evaluate_point_model
    from utils.seed import set_seed
    set_seed(int(cfg['experiment']['seed']));method=job['method']
    train,val=loaders(cfg,'train'),loaders(cfg,'validation');model=make_model(cfg,method)
    selected=balanced_indices(train.dataset)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay'])
    scaler=_make_grad_scaler(True);weights=control_weights(cfg) if method=='deterministic' else LossWeights(**{k:cfg['loss'][k] for k in ('dice_weight','bce_weight','lambda_kl','lambda_cal','lambda_rank')})
    before={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    elapsed=[];torch.cuda.reset_peak_memory_stats()
    for j,index in enumerate([selected[0],selected[-1],*selected]):
        check(job);subset=Subset(train.dataset,[index]);subset.total_clips_before_filter=train.dataset.total_clips_before_filter
        loader=DataLoader(subset,batch_size=1,collate_fn=train.collate_fn,num_workers=0)
        torch.cuda.synchronize();started=time.perf_counter()
        train_one_epoch(model,loader,optimizer,weights,1 if method=='deterministic' else 2,'cuda:0',
            amp=True,scaler=scaler,deadline=job['deadline'],log_every=10000,
            objective_fn=deterministic_objective if method=='deterministic' else None)
        torch.cuda.synchronize()
        if j>=2:elapsed.append(time.perf_counter()-started)
    changed=[n for n,p in model.named_parameters() if p.requires_grad and not torch.equal(before[n],p.detach().cpu())]
    if not any('image_encoder' in n for n in changed) or not any('sam_mask_decoder' in n for n in changed):
        raise ValueError('Disposable smoke/profile did not update both adapter groups')
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters() if p.requires_grad):
        raise FloatingPointError('Non-finite profiled adapter parameters')
    indices=balanced_indices(val.dataset,6);view=Subset(val.dataset,indices)
    view.case_ids=val.dataset.case_ids;view._clip_index=[val.dataset._clip_index[i] for i in indices]
    small=DataLoader(view,batch_size=1,collate_fn=val.collate_fn,num_workers=0)
    started=time.perf_counter();profile_root=Path(job['output_root'])/method/'profile'
    profile_root.mkdir(parents=True,exist_ok=True)
    progress=profile_root/'validation_progress.json';progress.unlink(missing_ok=True)
    if method=='deterministic':
        report=evaluate_point_model(model,small,'cuda:0',allowed_cases=set(view.case_ids),seed=90043,
            progress_path=progress,surface=False,deadline=job['deadline'],split='profiling_validation')
        complete=report['complete']
    else:
        report=run_evaluation(model,small,'cuda:0',k_infer=4,seed=90043,deadline=job['deadline'],compute_surface_distances=False,progress_path=progress)
        complete=report['protocol']['complete']
    peak=int(torch.cuda.max_memory_reserved());seconds=time.perf_counter()-started
    p75=float(np.quantile(elapsed,.75))
    result={'complete':complete,'method':method,'training_p75_seconds':p75,
        'training_sample_seconds':elapsed,'training_cases':train.dataset.case_ids,
        'train_slices':len(train.dataset),'validation_sample_slices':len(indices),
        'validation_sample_seconds':seconds,'validation_slices':len(val.dataset),
        'estimated_epoch_seconds':p75*len(train.dataset)+seconds/len(indices)*len(val.dataset),
        'profile_optimizer_updates':len(selected)+2,'profile_weights_discarded':True,
        'peak_reserved_bytes':peak,'research_updates':0,'gpu':torch.cuda.get_device_name(0),
        'encoder_checkpointing':cfg['runtime']['encoder_checkpointing'],
        'smoke_encoder_and_decoder_changed':True,'changed_trainable_tensors':len(changed)}
    atomic_json(profile_root/'report.json',result)
    del model,optimizer,scaler,train,val;gc.collect();torch.cuda.empty_cache()
    return result

def train(job,cfg):
    from training import train as training
    from evaluation import evaluate
    # Training seed may vary; validation prompts remain fixed across fitted seeds.
    original_eval=evaluate.run_evaluation
    def fixed_eval(*args,**kwargs):kwargs['seed']=90043;return original_eval(*args,**kwargs)
    evaluate.run_evaluation=fixed_eval
    if job['method']=='deterministic':
        from training import deterministic_control as control
        from evaluation.control_inputs import protocol_digest
        original_point=control.evaluate_point_model
        def fixed_point(*args,**kwargs):kwargs['seed']=90043;return original_point(*args,**kwargs)
        control.evaluate_point_model=fixed_point
        cfg['control']['protocol_sha256']=protocol_digest(cfg)
        train_loader,val_loader=training.build_dataloaders(cfg)
        model,result=control.run_fixed_training(cfg,train_loader,val_loader,'cuda:0',deadline=job['deadline'])
    else:
        last=Path(cfg['experiment']['output_dir'])/'last.pt'
        result=training.run_training(cfg,str(last) if last.is_file() else None);model=result.pop('model')
    result={k:v for k,v in result.items() if k!='model'}
    result['complete']=bool(result['protocol_complete']);result['fresh_run']=True
    result['historical_result_reused']=False
    selected=Path(cfg['experiment']['output_dir'])/'best.pt'
    if selected.is_file():result['selected_checkpoint_sha256']=digest_file(selected)
    return finite(result)

def dataset_identity(cfg,role,audit):
    return {'cases':audit['splits'][role]['cases'],'index':audit['splits'][role]['dataset_index'],
        'data_sha256':{cid:audit['data_sha256'][cid] for cid in audit['splits'][role]['cases']},
        'image_size':1024,'ct_window':[-175.,250.],'role':role}

def inference(job,cfg):
    import torch
    from inference_common import predict_one,save_arrays,fingerprint
    from auditlib.pixel_metrics import summarize_prediction
    from evaluation.point_metrics import compute_point_metrics
    from datasets.btcv_dataset import BTCV_ORGAN_NAMES
    method=job['method'];root=Path(job['output_root'])/method/'test'
    root.mkdir(parents=True,exist_ok=True)
    checkpoint=None if method=='frozen' else Path(cfg['experiment']['output_dir'])/'best.pt'
    weights_sha=cfg['sam2']['checkpoint_sha256'] if method=='frozen' else digest_file(checkpoint)
    started=time.perf_counter();model=make_model(cfg,method,checkpoint);model.eval()
    build_seconds=time.perf_counter()-started;before=fingerprint(model)
    audit=json.loads((Path(job['output_root'])/'data_audit.json').read_text())
    dataset=loaders(cfg,'test').dataset
    ident={'data':dataset_identity(cfg,'test',audit),'weights_sha256':weights_sha,
        'source_sha256':job['source_sha256'],'method':method,'k':4 if method=='balora' else 1,
        'seed':700043,'primary_policy':'none','runtime':job['runtime_identity'],
        'diagnostic_version':'full_run_v4','primary_metrics':'point_metrics_single_pass_v1_1 for all three methods'}
    fp=digest_json(ident);identity_path=root/'identity.json'
    if identity_path.is_file() and json.loads(identity_path.read_text())['fingerprint']!=fp:
        raise ValueError('Saved test records belong to different weights, data, runtime or protocol')
    atomic_json(identity_path,{'fingerprint':fp,'identity':ident})
    old=json.loads((root/'status.json').read_text()) if (root/'status.json').is_file() else {}
    torch.cuda.reset_peak_memory_stats();rows=[];complete=False;slice_times=[]
    from utils.runtime_budget import BudgetExpired
    try:
        with torch.no_grad():
            for i in range(len(dataset)):
                check(job);s=dataset[i];cid=s.case_id;z=int(s.slice_indices[0])
                image=s.image.to('cuda:0');labels=s.label.to('cuda:0');pending=[]
                for organ in range(1,14):
                    if not bool(labels[0,organ].any()):continue
                    key=f'{cid}:{z}:{organ}';path=root/'records'/(key.replace(':','_')+'.json')
                    if path.is_file():
                        value=json.loads(path.read_text())
                        if value['fingerprint']!=fp:raise ValueError('Changed test record identity')
                        if value.get('array_file') and digest_file(root/value['array_file'])!=value['array_sha256']:raise ValueError('Qualitative array checksum mismatch')
                        rows.append(value)
                    else:pending.append((organ,key,path))
                if not pending:continue
                start=time.perf_counter();torch.cuda.synchronize();enc_start=time.perf_counter()
                embedding=model.encode_image(image);torch.cuda.synchronize();enc_seconds=time.perf_counter()-enc_start
                for organ,key,path in pending:
                    check(job);target=labels[:,organ]
                    torch.cuda.synchronize();t=time.perf_counter()
                    p,epi,ranks,xy=predict_one(model,method,embedding,target,700043,cid,z,organ)
                    torch.cuda.synchronize();decode_seconds=time.perf_counter()-t
                    prob=p.float().cpu();truth=target[0].cpu()
                    seg,cal,empty=compute_point_metrics(prob,truth,surface=True)
                    pp=prob.numpy();gt=truth.numpy().astype(bool);ee=None if epi is None else epi.float().cpu().numpy()
                    row={'fingerprint':fp,'key':key,'case_id':cid,'slice_index':z,'organ_class':organ,
                        'organ':BTCV_ORGAN_NAMES[organ],'method':method,'prompt_xy':xy,
                        'seg':finite(seg.to_dict()),'cal':finite(cal.to_dict()),'empty_prediction':empty,
                        'dice':seg.dice,'ranks':ranks,'diagnostics':summarize_prediction(pp,gt,ee),
                        'encoder_seconds_for_slice':enc_seconds,'decode_seconds':decode_seconds,
                        'model_timing_scope':'CUDA synchronized encoder once per slice and decoder per prompt; excludes transfer, CPU metrics and export; BALoRA includes probes'}
                    if key in audit['qualitative_keys']:
                        array=root/'arrays'/(key.replace(':','_')+'.npz')
                        row['array_file']=str(array.relative_to(root))
                        row['array_sha256']=save_arrays(array,s.image[0,0].numpy(),gt,pp,ee,
                            {'key':key,'prompt_xy':xy,'fingerprint':fp,'selection':audit['qualitative_selection']})
                    atomic_json(path,finite(row));rows.append(row)
                    for module in model.modules():
                        if hasattr(module,'last_adapter_output'):module.last_adapter_output=None
                    del p,epi,pp,ee,prob,truth,target
                slice_times.append(time.perf_counter()-start)
                del embedding,image,labels
                if (i+1)%25==0:print(method,'test slices',i+1,'records',len(rows),flush=True)
            if {r['key'] for r in rows}!=set(audit['expected_test_keys']) or len(rows)!=2466:
                raise ValueError('Full test identities/coverage do not match the audited dataset')
            complete=True
    except BudgetExpired:pass
    finally:
        if fingerprint(model)!=before:raise RuntimeError('Inference changed fitted parameters')
        report={'complete':complete,'method':method,'records':len(rows),'weights_sha256':weights_sha,
            'fingerprint':fp,'model_build_seconds':build_seconds,'slice_wall_seconds_this_session':slice_times,
            'peak_reserved_bytes_this_session':int(torch.cuda.max_memory_reserved()),'parameters_unchanged':True,
            'peak_reserved_bytes_all_sessions':max(int(torch.cuda.max_memory_reserved()),int(old.get('peak_reserved_bytes_all_sessions',0))),
            'stored_trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'gpu':job['runtime_identity'],'no_epistemic_for_point_models':True}
        atomic_json(root/'status.json',report)
    return report

def refinement_audit(job,cfg):
    from evaluation import validation_audit,evaluate
    from inference_common import fingerprint
    audit_data=json.loads((Path(job['output_root'])/'data_audit.json').read_text())
    def semantic_fingerprint(model):
        return digest_json({'adapters':fingerprint(model),'source':job['source_sha256'],
            'backbone':cfg['sam2']['checkpoint_sha256'],'commit':cfg['sam2']['resolved_commit'],
            'validation_data':dataset_identity(cfg,'validation',audit_data),'runtime':job['runtime_identity']})
    # Temporary absolute checkpoint paths must not invalidate a byte-identical
    # resumed audit. Actual backbone/data/source/runtime identities remain fixed.
    validation_audit._model_fingerprint=semantic_fingerprint
    evaluate._model_fingerprint=semantic_fingerprint
    checkpoint=Path(cfg['experiment']['output_dir'])/'best.pt'
    model=make_model(cfg,'balora',checkpoint);loader=loaders(cfg,'validation')
    # This helper historically derives the prompt seed from experiment.seed.
    # Supply 42 only to its prompt-seed calculation; the fitted seed stays in
    # the model/checkpoint configuration and is reported separately below.
    audit_cfg=copy.deepcopy(cfg);audit_cfg['experiment']['seed']=42
    result=validation_audit.compare_validation(model,loader,audit_cfg,Path(job['output_root'])/'validation_refinement_audit','cuda:0',job['deadline'])
    result['training_seed']=int(cfg['experiment']['seed'])
    result['validation_prompt_seed']=90043
    result['primary_comparison_policy']='none fixed before this new fit; audit does not override it'
    result['historical_negative_finding_not_assumed_to_recur']=True
    return finite(result)

def run(job):
    sys.path.insert(0,str(Path(__file__).resolve().parent/'source'))
    action=job['action']
    try:
        cfg=json.loads(Path(job['configuration']).read_text())
        if action!='data':
            sys.path.insert(1,job['official_root']);import sam2
            if not Path(sam2.__file__).resolve().is_relative_to(Path(job['official_root']).resolve()):raise RuntimeError('Wrong official SAM2 import')
            from sam2.build_sam import build_sam2
        check(job)
        if action=='profile' and job['method']=='deterministic' and cfg['runtime']['encoder_checkpointing']=='off':
            import torch
            try:result=profile(job,cfg)
            except torch.cuda.OutOfMemoryError as exc:
                # The failed diagnostic model is disposable; no research state
                # is resumed from it. Only activation storage policy changes.
                exc.__traceback__=None
                cfg['runtime']['encoder_checkpointing']='per_block'
                atomic_json(job['configuration'],cfg)
                atomic_json(Path(job['output_root'])/'deterministic/chosen_runtime.json',
                    {'encoder_checkpointing':'per_block','reason':'Off-mode disposable profile exhausted GPU memory; restarted profile from fresh initialization'})
                gc.collect();torch.cuda.empty_cache();result=profile(job,cfg)
        else:
            result={'data':audit_data,'profile':profile,'train':train,'test':inference,'refinement':refinement_audit}[action](job,cfg)
        atomic_json(job['status_path'],finite(result));return result
    except Exception as exc:
        from utils.runtime_budget import BudgetExpired
        status={'complete':False,'status':'BUDGET_PAUSED' if isinstance(exc,BudgetExpired) else 'FAILED',
                'exception':type(exc).__name__,'message':str(exc),'action':action,'method':job['method'],
                'traceback':traceback.format_exc()}
        atomic_json(job['status_path'],status)
        if not isinstance(exc,BudgetExpired):raise
        return status

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job');a=p.parse_args();run(json.loads(Path(a.job).read_text()))
