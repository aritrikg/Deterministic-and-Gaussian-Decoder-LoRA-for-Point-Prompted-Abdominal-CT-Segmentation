"""Engineering fixtures only; no BTCV research run or performance is simulated."""
from pathlib import Path
import json,sys,time,zipfile
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'source'))
from auditlib.io_budget import atomic_json,BudgetStop,AccountingError
from run_budget import Ledger,ORIGINAL_ACCOUNTED_HOURS,merge,safe_extract
from protocol import assert_training_complete,status_path
from summarize_full_run import rollup,volume_analysis

def configuration(tmp_path,**updates):
    inputs=tmp_path/'inputs';inputs.mkdir(exist_ok=True)
    value={'study_id':'fixture-only','seed':42,'input_root':str(inputs),'output_root':str(tmp_path/'output'),
           'session_work_hours':10.5,'total_cumulative_hours':30,'parallel_on_two_gpus':True,'reconciliations':{}}
    value.update(updates);return value

def test_budget_preserves_original_usage_and_configured_session_cap(tmp_path):
    clock=[1000.];cfg=configuration(tmp_path);ledger=Ledger(cfg['output_root'],cfg['input_root'],'',cfg,started=clock[0],clock=lambda:clock[0])
    assert ledger.prior_hours==ORIGINAL_ACCOUNTED_HOURS
    assert abs(ledger.deadline-1000-(29.5-ORIGINAL_ACCOUNTED_HOURS)*3600+180)<1e-7
    clock[0]+=123;ledger.close('PARTIAL')
    resumed=Ledger(cfg['output_root'],cfg['input_root'],'',cfg,clock=lambda:clock[0])
    assert abs(resumed.prior_hours-ORIGINAL_ACCOUNTED_HOURS-123/3600)<1e-10
    resumed.close('TEST_ONLY')

def test_expanded_cap_is_recorded_without_resetting_original_ledger(tmp_path):
    cfg=configuration(tmp_path,total_cumulative_hours=60)
    ledger=Ledger(cfg['output_root'],cfg['input_root'],'',cfg,clock=lambda:1000.)
    value=json.loads(ledger.path.read_text())
    assert value['baseline_hours']==ORIGINAL_ACCOUNTED_HOURS and value['original_declared_cap_hours']==30
    assert value['active_total_cumulative_cap_hours']==60 and value['explicit_expansion_over_original_hours']==30
    ledger.close('TEST_ONLY')

def test_unclosed_ledger_needs_documented_full_duration(tmp_path):
    path=tmp_path/'full_run_ledger_x.json'
    atomic_json(path,{'schema':'balora_full_run_ledger_v4','baseline_hours':ORIGINAL_ACCOUNTED_HOURS,'sessions':{'x':{'elapsed_seconds':100,'closed':False}}})
    with pytest.raises(AccountingError):merge([path])
    with pytest.raises(AccountingError):merge([path],{'x':{'full_elapsed_seconds':90,'host_log_evidence':'host'}})
    merged=merge([path],{'x':{'full_elapsed_seconds':120,'host_log_evidence':'full host log 120s'}})
    assert merged['x']['elapsed_seconds']==120 and merged['x']['closed']

def test_resume_zip_blocks_traversal(tmp_path):
    archive=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('../outside.txt','fixture')
    with pytest.raises(ValueError):safe_extract(archive,tmp_path/'extract')

def test_full_heldout_comparison_requires_both_new_fits(tmp_path):
    with pytest.raises(ValueError):assert_training_complete(tmp_path)
    atomic_json(status_path(tmp_path,'deterministic','train'),{'complete':True})
    p=tmp_path/'deterministic/research/best.pt';p.parent.mkdir();p.write_bytes(b'engineering-fixture-not-research-weights')
    with pytest.raises(ValueError):assert_training_complete(tmp_path)

def test_pair_weighting_does_not_treat_unequal_slice_counts_as_volumes():
    records={}
    for method in ['frozen','deterministic','balora']:
        rows=[]
        for cid,score,count in [('a',.2,1),('b',.8,3)]:
            for i in range(count):
                rows.append({'case_id':cid,'organ':'organ','seg':{k:score for k in ['dice','iou','precision','recall','f1','hd95','assd']},
                             'cal':{k:.1 for k in ['ece','brier','nll']}})
        records[method]=rows
    value=rollup(records)
    dice=next(r for r in value['metrics'] if r['method']=='balora' and r['metric']=='dice')
    assert dice['mean']==.5 and dice['total_pairs']==2

def test_undefined_surface_denominators_are_retained():
    r={'case_id':'a','organ':'x','seg':{k:.5 for k in ['dice','iou','precision','recall','f1','hd95','assd']},'cal':{k:.1 for k in ['ece','brier','nll']}}
    r['seg']['hd95']=None;value=rollup({'balora':[r]})
    metric=next(x for x in value['metrics'] if x['metric']=='hd95')
    assert metric['mean'] is None and metric['defined_pairs']==0 and metric['undefined_slice_values']==1

def test_bootstrap_pairs_whole_volume_vectors_and_deletion():
    rows=[]
    for i in range(6):
        for method,score in [('frozen',.4),('deterministic',.8),('balora',.7)]:rows.append({'method':method,'case_id':str(i),'metric':'dice','mean':score+i*.001})
    arrays,contrasts,loo=volume_analysis(rows)
    value=next(x for x in contrasts if x['comparison']=='deterministic_minus_balora')
    assert value['independent_volume_units']==6 and value['ordered_resamples']==46656
    assert np.isclose(value['mean_difference_points'],10) and np.isclose(value['exploratory_lower_points'],10)
    assert all(np.isclose(r['difference_points'],10) for r in loo)

def test_models_use_two_gpus_for_training_but_one_for_matched_tests(tmp_path,monkeypatch):
    import full_pipeline as pipeline
    cfg=configuration(tmp_path);output=Path(cfg['output_root']);output.mkdir()
    calls=[]
    def launch(method,action,gpu,*args):
        calls.append(('launch',method,action,gpu));return {'method':method,'status':status_path(output,method,action)}
    def wait(jobs,ledger,action):
        calls.append(('wait',len(jobs)))
        for job in jobs:atomic_json(job['status'],{'complete':True})
    monkeypatch.setattr(pipeline,'launch',launch);monkeypatch.setattr(pipeline,'wait_workers',wait)
    pipeline.stage(['deterministic','balora'],'train',cfg,output,{'gpu_count':2},None,'fixture')
    assert calls[:3]==[('launch','deterministic','train',0),('launch','balora','train',1),('wait',2)]
    calls.clear();pipeline.stage(['frozen','deterministic','balora'],'test',cfg,output,{'gpu_count':2},None,'fixture')
    assert [c for c in calls if c[0]=='wait']==[('wait',1)]*3
    assert all(c[-1]==0 for c in calls if c[0]=='launch')

def test_missing_cuda_exports_failure_and_closed_ledger(tmp_path,monkeypatch):
    import full_pipeline as pipeline
    import auditlib.setup_inputs as setup
    cfg=configuration(tmp_path)
    monkeypatch.setattr(pipeline,'verify_source',lambda:'fixture-source-only')
    monkeypatch.setattr(pipeline,'resolve_btcv_root',lambda *a:(cfg['input_root'],{}))
    def unavailable(*args,**kwargs):raise RuntimeError('fixture CUDA unavailable')
    monkeypatch.setattr(setup,'setup_sam2',unavailable)
    with pytest.raises(RuntimeError,match='CUDA unavailable'):pipeline.run(cfg)
    state=json.loads((Path(cfg['output_root'])/'run_status.json').read_text())
    assert state['status']=='FAILED_SEE_FAILURE_REPORT' and not state['full_new_protocol_complete']
    with zipfile.ZipFile(tmp_path/'BALoRA_Full_Run_Results.zip') as z:
        ledger=json.loads(z.read(next(n for n in z.namelist() if Path(n).name.startswith('full_run_ledger_'))))
        assert all(s['closed'] for s in ledger['sessions'].values())
        assert not any(n.endswith('.pt') for n in z.namelist())

def test_auto_resume_discovers_new_run_archive_not_historical_weights(tmp_path):
    from full_pipeline import restore
    cfg=configuration(tmp_path);archive=Path(cfg['input_root'])/'renamed.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('saved/study_identity.json',json.dumps({'study_id':cfg['study_id']}))
        z.writestr('saved/deterministic/research/last.pt','fixture full-state placeholder not real weights')
    output=Path(cfg['output_root']);output.mkdir()
    path=restore(cfg,output)
    assert path and (output/'deterministic/research/last.pt').is_file()

def test_deterministic_prediction_has_no_epistemic_map():
    import torch
    from inference_common import predict_one
    class PointModel:
        def decode_one(self,embedding,points,labels):return torch.ones(1,1,64,64)*.7
    target=torch.zeros(1,64,64);target[:,20:40,22:44]=1
    p,u,ranks,xy=predict_one(PointModel(),'deterministic',{},target,700043,'fixture',21,3)
    assert p.shape==(64,64) and u is None and ranks=={} and target[0,xy[1],xy[0]]

def test_real_embedded_templates_build_all_three_runtime_configs(tmp_path):
    from protocol import configuration as make_config
    from auditlib.setup_inputs import COMMIT,BACKBONE
    cfg=configuration(tmp_path)
    env={'backbone_checkpoint':'/fixture/backbone.pt','official_commit':COMMIT,'backbone_sha256':BACKBONE}
    for method in ['frozen','deterministic','balora']:
        path=make_config(method,cfg,Path(cfg['output_root']),env,12345.)
        actual=json.loads(Path(path).read_text())
        assert actual['training']['num_epochs']==20 and actual['training']['min_epochs']==8
        assert actual['training']['early_stopping_patience']==6 and actual['data']['image_size']==1024
        assert actual['refinement']['enabled'] is False
        assert actual['runtime']['training_deadline_unix']==12345.
        if method=='deterministic':
            from evaluation.control_inputs import protocol_digest
            assert len(protocol_digest(actual))==64 and actual['lora']['encoder_rank']==actual['lora']['decoder_rank']==16
            assert actual['loss']['lambda_kl']==actual['loss']['lambda_rank']==0
        elif method=='balora':assert actual['bayesian']['k_train']==2 and actual['bayesian']['k_infer']==4

def test_worker_failure_stops_the_other_process_promptly(tmp_path):
    import subprocess
    from full_pipeline import wait_workers
    cfg=configuration(tmp_path);ledger=Ledger(cfg['output_root'],cfg['input_root'],'',cfg)
    jobs=[]
    for method,script in [('bad','raise RuntimeError("engineering fixture failure")'),('waiting','import time; time.sleep(30)')]:
        path=tmp_path/(method+'.log');handle=path.open('w')
        process=subprocess.Popen([sys.executable,'-c',script],stdout=handle,stderr=handle,start_new_session=True)
        jobs.append({'method':method,'process':process,'handle':handle,'log':path})
    started=time.time()
    try:
        with pytest.raises(RuntimeError,match='bad failed'):wait_workers(jobs,ledger,'train')
        assert time.time()-started<7 and all(j['process'].poll() is not None for j in jobs)
    finally:ledger.close('ENGINEERING_FIXTURE_ONLY')

def test_parent_error_contains_worker_traceback(tmp_path):
    from types import SimpleNamespace
    from full_pipeline import worker_failure
    log=tmp_path/'data.log';log.write_text('Traceback\nValueError: exact fixture cause\n')
    item={'method':'frozen','log':log,'process':SimpleNamespace(returncode=1)}
    assert 'ValueError: exact fixture cause' in str(worker_failure(item,'data'))

def test_source_fix_migrates_only_before_research_state(tmp_path,monkeypatch):
    import full_pipeline as pipeline
    import auditlib.setup_inputs as setup
    from protocol import identity
    cfg=configuration(tmp_path);output=Path(cfg['output_root']);output.mkdir()
    atomic_json(output/'study_identity.json',identity(cfg,'old-source'))
    monkeypatch.setattr(pipeline,'verify_source',lambda:'new-diagnostic-source')
    monkeypatch.setattr(pipeline,'resolve_btcv_root',lambda *a:(cfg['input_root'],{}))
    def stop(*a,**k):raise RuntimeError('fixture stop before data')
    monkeypatch.setattr(setup,'setup_sam2',stop)
    with pytest.raises(RuntimeError,match='fixture stop'):pipeline.run(cfg)
    migration=json.loads((output/'source_transition_before_training.json').read_text())
    assert migration['accounting_preserved'] and migration['previous']['source_sha256']=='old-source'

def test_source_fix_refuses_to_migrate_fitted_weights(tmp_path,monkeypatch):
    import full_pipeline as pipeline
    from protocol import identity
    cfg=configuration(tmp_path);output=Path(cfg['output_root']);output.mkdir()
    atomic_json(output/'study_identity.json',identity(cfg,'old-source'))
    path=output/'balora/research/last.pt';path.parent.mkdir(parents=True);path.write_bytes(b'test-only-placeholder')
    monkeypatch.setattr(pipeline,'verify_source',lambda:'new-source')
    with pytest.raises(ValueError,match='fitted/profiled'):pipeline.run(cfg)
