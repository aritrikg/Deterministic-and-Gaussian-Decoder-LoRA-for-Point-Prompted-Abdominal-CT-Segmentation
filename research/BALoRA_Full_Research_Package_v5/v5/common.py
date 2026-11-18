"""Identity, protocol and finite, prespecified internal experiment register."""
import hashlib,json,os,tempfile,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'base_v4'
REFERENCE=json.loads((ROOT/'evidence/independent_verification.json').read_text())

def read(p,default=None):
    p=Path(p)
    return json.loads(p.read_text()) if p.is_file() else default

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def atomic(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=p.parent)
    try:
        with os.fdopen(fd,'w') as f:json.dump(x,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def programme():
    jobs=[]
    def add(variant,seed,fold,priority,hypothesis):
        jobs.append(dict(id=f'{variant}_f{fold}_s{seed}',variant=variant,seed=seed,fold=fold,priority=priority,hypothesis=hypothesis))
    for seed in (123,2026):
        for v in ('det_v1','balora_v1'):add(v,seed,1,1,'Training-seed sensitivity of the completed v1 comparison')
    add('bayes_fixed_v1',42,1,2,'Fixed Bayesian decoder versus deterministic fixed rank; no allocator or rank penalty')
    # The fresh fixed Bayesian arm above is shared with the factorial; do not pay for a duplicate fit.
    for v in ('det_fixed_v2','det_adaptive_v2','bayes_adaptive_v2'):
        add(v,42,1,3,'Fresh 2x2 v2 Bayesian x adaptive-rank factorial; common entropy attribution and straight-through controller')
    for fold in range(2,6):
        for v in ('det_v1','balora_v1'):add(v,42,fold,4,'Internal outer-fold replication, same BTCV cohort; not external confirmation')
    add('point_unet',42,1,5,'Custom point-conditioned U-Net trained from scratch; not a published MedSAM reproduction')
    return jobs

def folds():
    """Preserve Fold 1; define a new, explicit disjoint partition of the other 24.
    Do not pretend this is the unexecuted historical five-fold split generator.
    """
    from protocol import split_ids
    import random
    f1=split_ids();ids=sorted(sum(f1.values(),[]));out={1:f1}
    remainder=sorted(set(ids)-set(f1['test']));random.Random(9042).shuffle(remainder)
    for f in range(2,6):
        test=sorted(remainder[(f-2)*6:(f-1)*6]);inner=sorted(set(ids)-set(test));random.Random(10042+f).shuffle(inner)
        out[f]={'train':sorted(inner[5:]),'validation':sorted(inner[:5]),'test':test}
    tests=[c for x in out.values() for c in x['test']]
    if len(tests)!=30 or len(set(tests))!=30:raise ValueError('Outer folds do not cover each volume exactly once')
    for x in out.values():
        if [len(x[r]) for r in ('train','validation','test')]!=[19,5,6] or len(set(sum(x.values(),[])))!=30:raise ValueError('Invalid split')
    return out

def configuration(job,config,environment,output,deadline):
    import copy
    variant=job['variant'];det=variant.startswith('det') or variant=='point_unet'
    cfg=copy.deepcopy(read(BASE/'assets'/('control_config.json' if det else 'original_config.json')))
    split=folds()[job['fold']];home=Path(output)/'jobs'/job['id']
    cfg['experiment'].update(seed=job['seed'],split_case_ids=split,split_hash=digest(split),output_dir=str(home/'research'),device='cuda:0')
    cfg['experiment'].pop('runtime_budget',None)
    cfg['sam2'].update(checkpoint=environment['backbone_checkpoint'],resolved_commit=environment['official_commit'],checkpoint_sha256=environment['backbone_sha256'])
    cfg['data'].update(root_dir=config['btcv_root'],cache_dir='/tmp/balora_v5_raw',image_size=1024,clip_len=1,foreground_only=True)
    for short,role in [('train','train'),('val','validation'),('test','test')]:
        p=Path(output)/'splits'/str(job['fold'])/(role+'.json');atomic(p,{'cases':split[role]});cfg['data'][short+'_split']=str(p)
    cfg['training'].update(num_epochs=20,min_epochs=8,early_stopping_patience=6,batch_size=1,num_workers=0,checkpoint_seconds=180)
    cfg['training'].pop('continuation_lr',None)
    cfg['runtime']={'encoder_checkpointing':'per_block','training_deadline_unix':deadline,'environment_versions':environment.get('versions',{}),'hardware':{'gpu':environment.get('gpu'),'cuda':environment.get('cuda')}}
    cfg.setdefault('refinement',{})['enabled']=False
    if 'control' in cfg:
        cfg['control'].pop('reference_checkpoint_sha256',None);cfg['control'].pop('protocol_sha256',None)
        cfg['control']['study_role']=variant
    cfg['v5']={**job,'validation_prompt_seed':90043,'test_prompt_seed':700043,'programme':'v5_internal_1','policy':'none'}
    if det:cfg['loss'].update(lambda_kl=0,lambda_rank=.01 if variant=='det_adaptive_v2' else 0)
    if variant=='bayes_fixed_v1' or variant.endswith('fixed_v2'):cfg['loss']['lambda_rank']=0
    return cfg

def scientific_identity(cfg):
    """Location-independent resume fingerprint: include every scientific coefficient."""
    x=json.loads(json.dumps(cfg))
    for k in ('root_dir','cache_dir','train_split','val_split','test_split'):x['data'].pop(k,None)
    for k in ('output_dir','device'):x['experiment'].pop(k,None)
    x['sam2'].pop('checkpoint',None);x['runtime'].pop('training_deadline_unix',None)
    return digest(x)
