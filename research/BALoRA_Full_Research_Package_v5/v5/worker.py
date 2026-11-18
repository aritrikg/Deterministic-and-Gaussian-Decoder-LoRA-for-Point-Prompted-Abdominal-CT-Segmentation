"""Entry point deliberately imports pinned SAM2 before the generic model package."""
import json,sys,time,traceback
from pathlib import Path

job=json.loads(Path(sys.argv[1]).read_text())
root=Path(job['source_root']);sys.path[:0]=[job['official_root'],str(root),str(root/'base_v4'),str(root/'base_v4/source')]
from v5.common import read,atomic,sha
from v5.engine import fit,profile,infer,build,is_bayesian

cfg=read(job['config_path']);home=Path(job['home']);action=job['action'];status={'complete':False}
try:
    import sam2,torch
    if not Path(sam2.__file__).resolve().is_relative_to(Path(job['official_root']).resolve()):raise ImportError('Wrong SAM2 import provenance')
    if action=='profile':status=profile(cfg,home,job['deadline'])
    elif action=='train':status=fit(cfg,home,job['deadline'])
    elif action=='ensemble':
        import copy
        from v5.models import IndependentSeedEnsemble
        from utils.checkpoint import load_checkpoint
        members=[];hashes=[]
        output=home.parents[1]
        for seed in (42,123,2026):
            if seed==42:
                mc=copy.deepcopy(cfg);mc['v5'].update(variant='det_v1',seed=42);mc['experiment']['seed']=42
                p=Path(job['reference_root'])/'deterministic/research/best.pt'
            else:
                h=output/'jobs'/f'det_v1_f1_s{seed}';mc=read(h/'configuration.json');p=h/'research/best.pt'
                if not read(h/'test_status.json',{}).get('complete'):raise ValueError('Ensemble member incomplete')
            m=build(mc,'cuda:0');load_checkpoint(m,None,str(p),restore_rng=False);m.eval();members.append(m);hashes.append(sha(p))
        if len(set(hashes))!=3:raise ValueError('Ensemble requires different fitted checkpoints')
        model=IndependentSeedEnsemble(members,cfg);status=infer(model,cfg,'test',home/'test',job['deadline']);status['member_hashes']=hashes
    else:
        from utils.checkpoint import load_checkpoint
        training=read(home/'research/training_status.json',{})
        if not training.get('complete'):raise ValueError('Full selected-checkpoint evaluation requires complete research training')
        selected=home/'research/best.pt'
        if sha(selected)!=training['selected_checkpoint_sha256']:raise ValueError('Selected checkpoint hash mismatch')
        model=build(cfg,'cuda:0');load_checkpoint(model,None,str(selected),restore_rng=False)
        if action=='test':status=infer(model,cfg,'test',home/'test',job['deadline'])
        elif action=='sensitivity':
            settings=[{'prompt_draw':d} for d in (1,2)]
            if is_bayesian(cfg):settings += [{'k':k} for k in (2,8,16)]+[{'probes':p} for p in (4,8)]+[{'rank':r} for r in (4,8,16)]
            for setting in settings:
                key='_'.join(f'{k}{v}' for k,v in sorted(setting.items()))
                infer(model,cfg,'validation',home/'sensitivity'/key,job['deadline'],setting=setting)
            status={'complete':True,'role':'validation_only','settings':settings,'interpretation':'post-test exploratory sensitivity; no re-selection of primary test policy'}
        else:raise ValueError('Unknown action '+action)
except BaseException as exc:
    status={'complete':False,'reason':str(exc),'type':type(exc).__name__,'traceback':traceback.format_exc()}
    print(status['traceback'],flush=True)
    atomic(home/(action+'_status.json'),status)
    # Budget stop is a resumable cooperative pause, actual failures retain nonzero exit.
    if type(exc).__name__ not in ('BudgetExpired','BudgetStop','KeyboardInterrupt'):raise
finally:atomic(home/(action+'_status.json'),status)
