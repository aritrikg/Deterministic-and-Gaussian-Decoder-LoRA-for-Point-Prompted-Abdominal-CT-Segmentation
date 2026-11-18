"""Two isolated GPU workers; no hidden budget reset or completed-run retraining."""
import json,os,shutil,subprocess,sys,time,traceback,zipfile
from pathlib import Path
from v5.common import ROOT,BASE,REFERENCE,atomic,read,sha,digest,programme,configuration

def find_reference(config,output):
    from run_budget import safe_extract
    explicit=config.get('reference_root','')
    candidates=[Path(explicit)] if explicit else list(Path(config['input_root']).rglob('study_identity.json'))
    if not explicit:candidates=[p.parent for p in candidates]
    def valid(p):
        return all((p/m/'research/best.pt').is_file() and sha(p/m/'research/best.pt')==REFERENCE['methods'][m]['checkpoint_sha256'] for m in ('deterministic','balora'))
    for p in candidates:
        if valid(p):return p
    for p in sorted(Path(config['input_root']).rglob('*.zip')):
        try:
            with zipfile.ZipFile(p) as z:
                names=z.namelist()
                if not any(n.endswith('deterministic/research/best.pt') for n in names):continue
                if not any(n.endswith('balora/research/best.pt') for n in names):continue
            dest=Path(output)/'reference_import'/sha(p)[:16];safe_extract(p,dest)
            for identity in dest.rglob('study_identity.json'):
                if valid(identity.parent):return identity.parent
        except zipfile.BadZipFile:continue
    raise FileNotFoundError('Attach the COMPLETED latest BALoRA_Full_Run_Results ZIP with selected fitted checkpoints (D 40ff778f..., B cb0d82ca...). A manuscript/supplement alone has no fitted weights. Set REFERENCE_ROOT if needed.')

def audit_reference(root,output):
    """Recompute the declared means from actual 7,398 primary records."""
    allrows={};report={}
    for method,expected in REFERENCE['methods'].items():
        rows=[read(p) for p in (Path(root)/method/'test/records').glob('*.json')]
        identity=read(Path(root)/method/'test/identity.json',{});ident=identity.get('identity',{})
        if ident.get('weights_sha256')!=expected['checkpoint_sha256'] or ident.get('k')!=expected['k'] or ident.get('seed')!=700043 or ident.get('primary_policy')!='none':raise ValueError('Reference inference policy/weight identity differs: '+method)
        if ident.get('source_sha256')!='7a1c30bb594633075838cfb886cd77d494dd7e605ed6ab076905a690eb2584cb':raise ValueError('Reference scientific source differs')
        if any(r.get('fingerprint')!=identity.get('fingerprint') for r in rows):raise ValueError('Mixed reference record identities')
        groups={};keys={}
        for r in rows:
            groups.setdefault((r['case_id'],r['organ_class']),[]).append(r['dice']);keys[r['key']]=r['prompt_xy']
        import statistics
        score=statistics.mean(statistics.mean(v) for v in groups.values()) if groups else None
        if len(rows)!=2466 or len(keys)!=2466 or len(groups)!=78 or score is None or abs(score-expected['dice'])>1e-9:raise ValueError('Reference coverage/score differs: '+method)
        if len({r['case_id'] for r in rows})!=6 or len({(r['case_id'],r['slice_index']) for r in rows})!=460:raise ValueError('Reference volume/slice coverage differs')
        if method!='frozen' and sha(Path(root)/method/'research/best.pt')!=expected['checkpoint_sha256']:raise ValueError('Selected reference weights differ')
        allrows[method]=keys;report[method]={'complete':True,'mean_dice':score,'records':len(rows),'pairs':len(groups),'checkpoint_sha256':expected['checkpoint_sha256']}
    if allrows['frozen']!=allrows['deterministic'] or allrows['balora']!=allrows['deterministic']:raise ValueError('Reference prompts differ')
    atomic(Path(output)/'reference_audit.json',{'complete':True,'methods':report,'paired_prompts':True,'source_root':str(root)})
    return report

def restore(config,output):
    from run_budget import safe_extract
    src=config.get('resume_path','')
    if not src:
        archives=sorted(Path(config['input_root']).rglob('BALoRA_Strengthening_v5_Results.zip'))
        if len(archives)>1:raise ValueError('Multiple v5 resumes; set RESUME_PATH to the latest archive')
        if archives:src=str(archives[0])
    if not src:return
    src=Path(src);tmp=Path('/tmp/balora_v5_resume')
    if src.is_file():safe_extract(src,tmp);roots=list(tmp.rglob('programme_identity.json'));src=roots[0].parent if len(roots)==1 else None
    if src is None or not (src/'programme_identity.json').is_file():raise ValueError('Resume has no unique v5 programme identity')
    if Path(output).exists() and list(Path(output).glob('jobs/*/research/last.pt')):return
    shutil.copytree(src,output,dirs_exist_ok=True)

def archive(output,ledger,status):
    # Working state is atomic throughout. ZIP is an additional resumable export.
    output=Path(output);dest=output.parent/'BALoRA_Strengthening_v5_Results.zip';tmp=dest.with_suffix('.tmp')
    atomic(output/'run_status.json',status)
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(output.rglob('*')):
            if not p.is_file() or 'reference_import' in p.parts or 'setup' in p.parts:continue
            z.write(p,'BALoRA_Strengthening_v5_Results/'+str(p.relative_to(output)),compress_type=zipfile.ZIP_STORED if p.suffix in ('.pt','.npz') else zipfile.ZIP_DEFLATED)
    ledger.close(status['status'])
    with zipfile.ZipFile(tmp,'a') as z:
        # Use a unique final path; reader merges session IDs, not summing copies.
        z.write(ledger.path,'BALoRA_Strengthening_v5_Results/final_accounting/'+ledger.path.name)
    os.replace(tmp,dest);return dest

def run(config,started):
    from run_budget import Ledger,BudgetStop
    output=Path(config['output_root']);output.mkdir(parents=True,exist_ok=True)
    restore(config,output)
    # Trusted carried closed ledgers are always included even if prior outputs are omitted.
    imported=output/'carried_accounting';shutil.copytree(ROOT/'evidence',imported,dirs_exist_ok=True)
    ledger=Ledger(output,config['input_root'],config.get('resume_path',''),config,started=started)
    status={'status':'RUNNING','complete_internal_programme':False,'external_validation':'REQUIRES_EXTERNAL_DATA','author_declarations':'AUTHOR INPUT','prior_hours':ledger.prior_hours,'active_cap':ledger.limit}
    processes=[]
    try:
        from protocol import verify_source
        from data_root import resolve_btcv_root
        from auditlib.setup_inputs import setup_sam2
        source_hash=verify_source()
        root=find_reference(config,output);audit_reference(root,output)
        print('Verified completed comparison. Prior accounted hours:',ledger.prior_hours,'active cap:',ledger.limit,flush=True)
        ledger.check(300)
        env=setup_sam2(output/'setup',config['input_root'],config.get('sam2_checkpoint',''),deadline=ledger.deadline)
        config['btcv_root'],resolution=resolve_btcv_root(config['input_root'],config.get('btcv_root',''))
        atomic(output/'dataset_resolution.json',resolution)
        # setup returns official root; children use that exact path.
        atomic(output/'environment.json',env)
        import torch
        gpus=torch.cuda.device_count()
        if gpus<1:raise RuntimeError('Select Kaggle GPU accelerator')
        jobs=programme();identity={'schema':'v5_internal_1','source_sha256':source_hash,'implementation_sha256':digest({str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'v5').glob('*.py')}),'jobs':jobs,'reference_checkpoints':{k:v['checkpoint_sha256'] for k,v in REFERENCE['methods'].items()},'no_refinement':True,'runtime_identity':{k:env[k] for k in ('versions','gpu','cuda')}}
        old=read(output/'programme_identity.json')
        if old and old!=identity:raise ValueError('Programme/source changed; preserve old state and use a new output identity')
        atomic(output/'programme_identity.json',identity);atomic(output/'configuration.json',config)
        shutil.copytree(ROOT,output/'source_snapshot',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','historical','reference_preview'))
        from full_worker import audit_data
        audit_cfg=configuration(jobs[0],config,env,output,ledger.deadline)
        raw_audit=audit_data({'output_root':str(output),'deadline':ledger.deadline},audit_cfg)
        reference_audit=read(root/'data_audit.json')
        if reference_audit is None or reference_audit['data_sha256']!=raw_audit['data_sha256']:raise ValueError('Raw dataset differs from completed reference')
        def launch(job,action,gpu,extra=None):
            home=output/'jobs'/job['id'];home.mkdir(parents=True,exist_ok=True)
            cfg=configuration(job,config,env,output,ledger.deadline);cp=home/'configuration.json';atomic(cp,cfg)
            payload={'config_path':str(cp),'home':str(home),'action':action,'deadline':ledger.deadline,'source_root':str(ROOT),'official_root':env['sam2_root'] if 'sam2_root' in env else env['official_root'],**(extra or {})}
            jp=home/(action+'_job.json');atomic(jp,payload);log=home/(action+'.log');f=open(log,'a',buffering=1)
            e=os.environ.copy();e['CUDA_VISIBLE_DEVICES']=str(gpu);e['OMP_NUM_THREADS']='2';e['MKL_NUM_THREADS']='2';e['PYTHONUNBUFFERED']='1'
            proc=subprocess.Popen([sys.executable,str(ROOT/'v5/worker.py'),str(jp)],stdout=f,stderr=subprocess.STDOUT,env=e)
            item=(proc,f,home,action,log);processes.append(item);return item
        def wait(items):
            last_print=time.time()
            while any(p.poll() is None for p,_,_,_,_ in items):
                ledger.tick()
                if time.time()>=ledger.deadline+60:
                    for p,_,_,_,_ in items:
                        if p.poll() is None:p.terminate()
                    raise BudgetStop('Worker deadline grace exceeded; use last atomic checkpoint')
                for p,_,_,a,log in items:
                    if p.poll() not in (None,0):raise RuntimeError(a+' worker failed:\n'+'\n'.join(log.read_text(errors='replace').splitlines()[-35:]))
                if time.time()-last_print>60:print('Workers active:',[h.name+':'+a for p,_,h,a,_ in items if p.poll() is None],flush=True);last_print=time.time()
                time.sleep(3)
            for p,f,h,a,log in items:
                f.close()
                if p.returncode!=0:raise RuntimeError(a+' failed:\n'+'\n'.join(log.read_text(errors='replace').splitlines()[-35:]))
                r=read(h/(a+'_status.json'),{})
                if not r.get('complete'):raise BudgetStop(r.get('reason','Incomplete '+a+'; resume saved state'))
        # Representative profiling per variant. Profiles never become research checkpoints.
        variants={j['variant']:j for j in jobs}
        for v,j in variants.items():
            report=read(output/'jobs'/j['id']/'profile.json')
            if report and report.get('complete'):continue
            ledger.check(600);wait([launch(j,'profile',0)])
        profiles={v:read(output/'jobs'/j['id']/'profile.json') for v,j in variants.items()}
        # Sequential upper forecast and ideal two-GPU lower estimate, both pessimistic max20 epochs.
        from v5.common import folds
        counts={}
        for split in raw_audit['splits'].values():
            for cid,a,b in split['dataset_index']:counts[cid]=counts.get(cid,0)+1
        costs={}
        for j in jobs:
            p=profiles[j['variant']];split=folds()[j['fold']];home=output/'jobs'/j['id'];tr=read(home/'research/training_status.json',{})
            ntrain=sum(counts[c] for c in split['train']);nval=sum(counts[c] for c in split['validation']);ntest=sum(counts[c] for c in split['test'])
            validation_per_slice=max(0,p['estimated_epoch_seconds']-p['train_p75_seconds']*p['train_slices'])/p['validation_slices']
            epochs_left=0 if tr.get('complete') else max(0,20-tr.get('epoch',0))
            train_cost=(p['train_p75_seconds']*ntrain+validation_per_slice*nval)*epochs_left
            test_cost=0 if read(home/'test_status.json',{}).get('complete') else p['estimated_full_test_diagnostic_seconds']*ntest/460
            costs[j['id']]=train_cost+test_cost
        sensitivity_seconds=sum(profiles[j['variant']]['estimated_full_test_diagnostic_seconds']*(10 if j['variant']=='balora_v1' else 2) for j in jobs[:4] if not read(output/'jobs'/j['id']/'sensitivity_status.json',{}).get('complete'))
        if not read(output/'jobs/det_ensemble_f1_s42_123_2026/ensemble_status.json',{}).get('complete'):sensitivity_seconds+=profiles['det_v1']['estimated_full_test_diagnostic_seconds']*3
        sequential=(sum(costs.values())+sensitivity_seconds)*1.25/3600+2;ideal=(max(max(costs.values()),sum(costs.values())/max(1,min(2,gpus)))+sensitivity_seconds)*1.25/3600+2
        forecast={'max_epochs_per_fit':20,'assumes_early_stop':False,'retry_export_margin':.25,'setup_analysis_reserve_hours':2,'remaining_under_current_cap_hours':max(0,ledger.limit-ledger.prior_hours-(time.time()-ledger.start)/3600-.5),'expanded_incremental_hours_ideal_parallel':ideal,'expanded_incremental_hours_sequential':sequential,'per_job_seconds':costs,'forecasts_not_guarantees':True}
        atomic(output/'measured_forecast.json',forecast);print(json.dumps(forecast,indent=2),flush=True)
        if config.get('programme_mode')=='REMAINING_BUDGET':
            raise BudgetStop('Profiles and forecasts complete. To execute the complete internal programme, select EXPANDED_INTERNAL and explicitly allocate the cumulative cap in the configuration cell. Completed reference experiments remain complete.')
        # Same-priority fits execute in pairs, one per physical GPU. All remain full coverage.
        pending=[j for j in jobs if not read(output/'jobs'/j['id']/'train_status.json',{}).get('complete')]
        for j in jobs:
            if read(output/'jobs'/j['id']/'train_status.json',{}).get('complete') and not read(output/'jobs'/j['id']/'test_status.json',{}).get('complete'):
                ledger.check(300);wait([launch(j,'test',0)])
        for start in range(0,len(pending),min(2,gpus)):
            ledger.check(300);group=pending[start:start+min(2,gpus)];wait([launch(j,'train',gpu) for gpu,j in enumerate(group)])
            # Evaluate before moving on, so each complete fit has a complete selected-checkpoint comparison.
            for gpu,j in enumerate(group):
                ledger.check(300);wait([launch(j,'test',gpu)])
        for j in jobs:
            if not read(output/'jobs'/j['id']/'test_status.json',{}).get('complete'):
                ledger.check(300);wait([launch(j,'test',0)])
        # Validation-only sensitivity: never choose a new primary policy on the inspected test cohort.
        for j in jobs[:4]:
            ledger.check(300)
            if not read(output/'jobs'/j['id']/'sensitivity_status.json',{}).get('complete'):wait([launch(j,'sensitivity',0)])
        ensemble_job={'id':'det_ensemble_f1_s42_123_2026','variant':'det_ensemble','seed':42,'fold':1}
        if not read(output/'jobs'/ensemble_job['id']/'ensemble_status.json',{}).get('complete'):
            ledger.check(300);wait([launch(ensemble_job,'ensemble',0,{'reference_root':str(root)})])
        status.update(status='FULL_INTERNAL_PROGRAMME_COMPLETE',complete_internal_programme=True)
    except (Exception,KeyboardInterrupt) as exc:
        status.update(status='PAUSED' if isinstance(exc,(BudgetStop,KeyboardInterrupt)) or type(exc).__name__=='BudgetExpired' else 'FAILED',reason=str(exc))
        atomic(output/'exception.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        print(status['status'],status['reason'],flush=True)
    finally:
        for p,f,_,_,_ in processes:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            if not f.closed:f.close()
        from v5.report import report
        try:
            report(output);status['figures_and_tables_complete']=True
        except Exception:
            status['figures_and_tables_complete']=False
            if status.get('complete_internal_programme'):status.update(status='ANALYSIS_EXPORT_FAILED',complete_internal_programme=False,reason='Fits completed but analysis/figures failed; inspect report_error.json')
            atomic(output/'report_error.json',{'traceback':traceback.format_exc()})
    return status,archive(output,ledger,status)
