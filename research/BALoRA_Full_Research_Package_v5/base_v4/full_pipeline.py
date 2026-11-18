"""Full raw-BTCV -> fresh training -> selection -> held-out tests -> figures."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,signal,subprocess,sys,time,traceback,zipfile
from auditlib.io_budget import atomic_json,digest_file,digest_json,BudgetStop
from run_budget import Ledger,ORIGINAL_ACCOUNTED_HOURS,safe_extract
from protocol import ROOT,METHODS,identity,verify_source,configuration,status_path,assert_training_complete
from data_root import resolve_btcv_root

def get_status(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else {'complete':False}

def restore(config,output):
    resume=config.get('resume_root','').strip()
    if not resume:
        if (output/'study_identity.json').is_file():return str(output)
        candidates=[];inputs=Path(config['input_root'])
        for p in inputs.rglob('study_identity.json') if inputs.is_dir() else []:
            if json.loads(p.read_text()).get('study_id')==config['study_id']:candidates.append(p.parent)
        if not candidates and inputs.is_dir():
            for p in sorted(inputs.rglob('*.zip')):
                with zipfile.ZipFile(p) as z:
                    names=[n for n in z.namelist() if n.endswith('study_identity.json')]
                    if any(json.loads(z.read(n)).get('study_id')==config['study_id'] for n in names):candidates.append(p)
        if len(candidates)>1:raise ValueError('Several v4 resumes match. Set resume_root to the intended latest output')
        if not candidates:return ''
        resume=str(candidates[0])
    path=Path(resume).expanduser().resolve()
    if path.is_file():
        unpack=output.parent/'balora_resume_unpacked';safe_extract(path,unpack)
        candidates=[p.parent for p in unpack.rglob('study_identity.json')]
        if len(candidates)!=1:raise ValueError('Resume ZIP must contain one full-run output')
        path=candidates[0]
    if not (path/'study_identity.json').is_file():raise FileNotFoundError('Set resume_root to a v4 results ZIP or its extracted full-run output folder')
    if path.resolve()!=output.resolve():
        if (output/'study_identity.json').is_file():
            # Never combine two independently progressing research directories.
            raise FileExistsError('Output already contains a study. Resume it directly or choose an empty output directory')
        shutil.copytree(path,output,dirs_exist_ok=True)
    return str(path)

def worker_failure(item,stage):
    path=Path(item['log'])
    tail=path.read_text(errors='replace')[-18000:] if path.is_file() else '(worker log unavailable)'
    report={}
    status=item.get('status')
    if status and Path(status).is_file():
        try:report=json.loads(Path(status).read_text())
        except (ValueError,OSError):pass
    message=(stage+' '+item['method']+' failed (exit '+str(item['process'].returncode)+').\n'
             +'Worker status: '+json.dumps(report,ensure_ascii=False)+'\n'
             +'--- worker log tail ---\n'+tail+'\n--- end worker log ---\nLog: '+str(path))
    print(message,flush=True)
    return RuntimeError(message)

def wait_workers(processes,ledger,stage):
    active=list(processes)
    try:
        while active:
            ledger.tick(stage)
            for item in processes:
                code=item['process'].poll()
                if code is not None and code!=0:
                    raise worker_failure(item,stage)
            active=[item for item in active if item['process'].poll() is None]
            if not active:break
            if time.time()>ledger.deadline+60:
                raise BudgetStop('Worker exceeded cooperative deadline; last atomic states retained')
            time.sleep(.5)
        for item in processes:
            if item['process'].returncode:
                raise worker_failure(item,stage)
    finally:
        for item in processes:
            p=item['process']
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
            item['handle'].close()

def launch(method,action,gpu,config,output,environment,ledger,source_hash):
    ledger.check(15)
    cfg_path=configuration(method,config,output,environment,ledger.deadline)
    target=output/'data_status.json' if action=='data' else status_path(output,method,action)
    job={'action':action,'method':method,'configuration':cfg_path,'output_root':str(output),
        'status_path':str(target),'official_root':environment['official_root'],
        'source_sha256':source_hash,'deadline':ledger.deadline,
        'runtime_identity':{'gpu':environment['gpu'],'versions':environment['versions'],'cuda':environment['cuda']}}
    job_path=output/'jobs'/(method+'_'+action+'.json');atomic_json(job_path,job)
    log=output/'logs'/(method+'_'+action+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    handle=log.open('a');env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']=str(gpu)
    env.setdefault('OMP_NUM_THREADS','2');env.setdefault('MKL_NUM_THREADS','2')
    env.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');env['PYTHONUNBUFFERED']='1'
    process=subprocess.Popen([sys.executable,str(ROOT/'full_worker.py'),str(job_path)],stdout=handle,
        stderr=subprocess.STDOUT,env=env,start_new_session=True)
    print('Started:',method,action,'physical GPU',gpu,'log',log,flush=True)
    return {'method':method,'process':process,'handle':handle,'log':log,'status':target}

def stage(methods,action,config,output,environment,ledger,source_hash):
    atomic_json(output/'active_stage.json',{'action':action,'methods':methods,'time_unix':time.time()})
    count=int(environment['gpu_count']);parallel=bool(config.get('parallel_on_two_gpus',True) and count>=2 and action in ('train','profile'))
    pending=[m for m in methods if action=='data' or not get_status(status_path(output,m,action)).get('complete')]
    width=2 if parallel else 1
    for start in range(0,len(pending),width):
        group=pending[start:start+width]
        jobs=[launch(m,action,i,config,output,environment,ledger,source_hash) for i,m in enumerate(group)]
        wait_workers(jobs,ledger,action)
        if not all(get_status(j['status']).get('complete') for j in jobs):
            raise BudgetStop('Incomplete '+action+' stage; resume the saved output without restarting completed work')

def live_plan(output,environment,ledger):
    reports={m:get_status(status_path(output,m,'profile')) for m in ['deterministic','balora']}
    training={m:20*r['estimated_epoch_seconds']*1.2 for m,r in reports.items()}
    # Final inference, regional uncertainty diagnostics, two refinement passes
    # and figures require their own allowance; this conservative proxy is not
    # a claim of matched inference speed or a guarantee of completion.
    sequential=sum(training.values());parallel=max(training.values())
    estimate=(parallel if environment['gpu_count']>=2 else sequential)+6*3600
    value={'complete':True,'requested_max_epochs_per_model':20,'minimum_epochs':8,'patience':6,
        'observed_profiles':reports,'max_target_training_forecast_seconds':training,
        'final_inference_audit_export_proxy_hours':6,'full_run_proxy_hours':estimate/3600,
        'remaining_cumulative_working_hours':max(0,ledger.working_ceiling-ledger.prior_hours-(time.time()-ledger.start)/3600),
        'estimated_to_fit_active_cumulative_allowance':estimate<max(0,(ledger.working_ceiling-ledger.prior_hours)*3600-(time.time()-ledger.start)),
        'estimated_to_fit_original_remaining_allowance':estimate<max(0,(29.5-ledger.prior_hours)*3600-(time.time()-ledger.start)),
        'original_budget_cap_hours':30,'active_cumulative_cap_hours':ledger.limit,
        'all_requested_training_retained':True,'on_insufficient_time':'Train/resume while allowed, then save PARTIAL; never declare a shortened experiment complete',
        'early_stopping_not_assumed_to_repeat':True,'gpu_parallelism':'independent model processes on distinct GPUs, not distributed training'}
    atomic_json(output/'live_runtime_plan.json',value)
    print('Full-target proxy hours:',round(estimate/3600,2),'remaining working allowance:',round(value['remaining_cumulative_working_hours'],2),flush=True)
    return value

def checkpoint_export(output,ledger,state):
    atomic_json(output/'run_status.json',state)
    # Checkpoints are essential outputs in this training notebook.
    # Raw caches and the 323 MB shared pretrained backbone stay out of the ZIP.
    archive=output.parent/'BALoRA_Full_Run_Results.zip';temporary=archive.with_suffix('.partial.zip')
    skip={'setup','__pycache__','.pytest_cache','raw_volume_cache','tensorboard'}
    hashes={}
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for path in sorted(output.rglob('*')):
            if not path.is_file() or path==ledger.path or path.name=='SHA256_manifest.json' or set(path.relative_to(output).parts)&skip or path.suffix=='.tmp':continue
            rel=path.relative_to(output);hashes[str(rel)]=digest_file(path)
            z.write(path,Path(output.name)/rel,compress_type=zipfile.ZIP_STORED if path.suffix in ('.pt','.npz') else zipfile.ZIP_DEFLATED)
        atomic_json(output/'SHA256_manifest.json',hashes)
        z.write(output/'SHA256_manifest.json',Path(output.name)/'SHA256_manifest.json')
    ledger.close(state['status'])
    with zipfile.ZipFile(temporary,'a',zipfile.ZIP_DEFLATED) as z:z.write(ledger.path,Path(output.name)/ledger.path.name)
    os.replace(temporary,archive)
    print('Saved full-state results:',archive,flush=True)
    return str(archive)

def run(config,started=None):
    config=dict(config);output=Path(config['output_root']).expanduser().resolve();output.mkdir(parents=True,exist_ok=True)
    resumed=restore(config,output)
    source_hash=verify_source();ident=identity(config,source_hash)
    identity_path=output/'study_identity.json'
    if identity_path.is_file() and json.loads(identity_path.read_text())!=ident:
        old=json.loads(identity_path.read_text())
        old_core={k:v for k,v in old.items() if k!='source_sha256'}
        new_core={k:v for k,v in ident.items() if k!='source_sha256'}
        fitted=any(output.rglob('*.pt')) or any(output.glob('*/test/records/*.json'))
        profiled=any(get_status(status_path(output,m,'profile')).get('complete') for m in ['deterministic','balora'])
        if old_core!=new_core or fitted or profiled:
            raise ValueError('Output is a different study/source/seed or contains fitted/profiled state. Use the matching notebook or a distinct output directory')
        atomic_json(output/'source_transition_before_training.json',{'previous':old,'updated':ident,
            'reason':'Bootstrap/diagnostic fix applied before any fitted weights, test records or completed profile existed',
            'accounting_preserved':True})
    atomic_json(identity_path,ident)
    ledger=Ledger(output,config['input_root'],resumed,config,started)
    state={'status':'RUNNING','full_new_protocol_complete':False,'historical_three_method_comparison':'COMPLETE; unchanged',
        'new_training_methods':['deterministic','balora'],'raw_data_only_required':True,'selected_policy':'none',
        'study_id':config['study_id'],'source_sha256':source_hash,'stage':'setup'}
    failure=None
    try:
        config['btcv_root'],input_resolution=resolve_btcv_root(config['input_root'],config.get('btcv_root',''))
        atomic_json(output/'resolved_btcv_root.json',input_resolution)
        print('Selected labeled BTCV cohort:',config['btcv_root'],'| 30 cases, fixed 19/5/6 split',flush=True)
        atomic_json(output/'requested_configuration.json',config)
        atomic_json(output/'run_status.json',state)
        from auditlib.setup_inputs import setup_sam2
        ledger.check(120)
        environment=setup_sam2(output/'setup',config['input_root'],deadline=ledger.deadline)
        import torch
        environment['gpu_count']=torch.cuda.device_count()
        if not environment['gpu_count']:raise RuntimeError('Enable a CUDA GPU')
        atomic_json(output/'environment.json',environment)
        atomic_json(output/'environment_history'/('session_'+ledger.sid+'.json'),environment)
        print('Historical hours carried:',ledger.prior_hours,'active total cap:',ledger.limit,'GPUs:',environment['gpu_count'],flush=True)
        state['stage']='data_audit';stage(['frozen'],'data',config,output,environment,ledger,source_hash)
        state['stage']='disposable_profiles';stage(['deterministic','balora'],'profile',config,output,environment,ledger,source_hash)
        live_plan(output,environment,ledger)
        state['stage']='fresh_research_training';stage(['deterministic','balora'],'train',config,output,environment,ledger,source_hash)
        assert_training_complete(output)
        state['stage']='validation_refinement_audit';stage(['balora'],'refinement',config,output,environment,ledger,source_hash)
        state['stage']='heldout_evaluation';stage(['frozen','deterministic','balora'],'test',config,output,environment,ledger,source_hash)
        state['stage']='figures_and_tables';ledger.check(90)
        from summarize_full_run import summarize
        report=summarize(output)
        state.update(status='FULL_NEW_FOLD1_PROTOCOL_COMPLETE',stage='completed',full_new_protocol_complete=True,
            new_test_coverage=report['coverage'],figure_sets=report['figure_sets'])
    except BudgetStop as exc:
        state.update(status='PARTIAL_SESSION_OR_CUMULATIVE_BUDGET_PAUSED',pause_reason=str(exc))
    except Exception as exc:
        failure=exc
        state.update(status='FAILED_SEE_FAILURE_REPORT',exception=type(exc).__name__,message=str(exc))
        atomic_json(output/'failure.json',{'stage':state['stage'],'exception':type(exc).__name__,
            'message':str(exc),'traceback':traceback.format_exc(),'full_new_protocol_complete':False})
    finally:
        for method in ['deterministic','balora']:
            state[method+'_training']=get_status(status_path(output,method,'train'))
        # Partial training curves are useful diagnostics, explicitly labeled.
        try:
            from summarize_full_run import training_figure
            training_figure(output,partial=not state['full_new_protocol_complete'])
        except Exception as exc:
            atomic_json(output/'partial_figure_note.json',{'message':str(exc),'not_a_research_completion':True})
        archive=checkpoint_export(output,ledger,state)
    if failure is not None:raise failure
    return state,archive

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('configuration');a=p.parse_args();print(run(json.loads(Path(a.configuration).read_text())))
