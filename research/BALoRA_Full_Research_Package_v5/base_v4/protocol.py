"""Fresh Fold-1 protocol; no historical scores or fitted adapters are inputs."""
from pathlib import Path
import copy,hashlib,json
from auditlib.io_budget import atomic_json,digest_json,digest_file

ROOT=Path(__file__).resolve().parent
METHODS=('frozen','deterministic','balora')

def split_ids():return json.loads((ROOT/'assets/original_config.json').read_text())['experiment']['split_case_ids']

def verify_source():
    manifest=json.loads((ROOT/'source_hashes.json').read_text())
    for rel,expected in manifest.items():
        if digest_file(ROOT/rel)!=expected:raise ValueError('Research source changed: '+rel)
    return digest_json(manifest)

def identity(config,source_hash):
    return {'workflow':'full_train_select_test_figures_v4','method_version':'v1 unchanged',
        'study_id':config['study_id'],'seed':int(config['seed']),'split_case_ids':split_ids(),
        'validation_prompt_seed':90043,'test_prompt_seed':700043,'max_epochs':20,'min_epochs':8,'patience':6,
        'image_size':1024,'ct_window':[-175.,250.],'k_train':2,'k_infer':4,'deterministic_k':1,
        'policy':'none fixed as primary before new fitting','refinement_audit':True,
        'source_sha256':source_hash,'historical_adapters_used':False}

def configuration(method,config,output,environment,deadline):
    name='control_config.json' if method=='deterministic' else 'original_config.json'
    cfg=copy.deepcopy(json.loads((ROOT/'assets'/name).read_text()))
    output=Path(output)
    cfg['experiment'].update(seed=int(config['seed']),device='cuda:0',output_dir=str(output/method/'research'))
    cfg['experiment'].pop('runtime_budget',None)
    cfg['sam2']['checkpoint']=environment['backbone_checkpoint']
    cfg['sam2']['resolved_commit']=environment['official_commit']
    cfg['sam2']['checkpoint_sha256']=environment['backbone_sha256']
    cfg['data'].update(root_dir=config.get('btcv_root') or config['input_root'],cache_dir='/tmp/balora_full_run_v4_cache',foreground_only=True)
    for short,role in [('train','train'),('val','validation'),('test','test')]:
        path=output/'splits'/(role+'.json');atomic_json(path,{'cases':split_ids()[role]});cfg['data'][short+'_split']=str(path)
    cfg['training'].update(num_epochs=20,min_epochs=8,early_stopping_patience=6,batch_size=1,num_workers=0,checkpoint_seconds=300)
    cfg['training'].pop('continuation_lr',None)
    cfg.setdefault('refinement',{})['enabled']=False
    cfg['runtime']={'training_deadline_unix':deadline,'training_pause_reason':'SESSION_LIMIT_PAUSED',
                    'evaluation_pause_reason':'SESSION_LIMIT_PAUSED','encoder_checkpointing':'off' if method=='deterministic' else 'per_block'}
    chosen=output/method/'chosen_runtime.json'
    if method=='deterministic' and chosen.is_file():
        cfg['runtime']['encoder_checkpointing']=json.loads(chosen.read_text())['encoder_checkpointing']
    cfg['full_run_v4']={'seed':int(config['seed']),'fresh_adapters':True,'primary_policy':'none',
        'validation_seed':90043,'test_seed':700043,'history_not_expected_to_match_previous_fit':True}
    if method=='deterministic':
        # The hash implementation is imported only in isolated source workers.
        cfg['control'].pop('reference_checkpoint_sha256',None)
        cfg['control']['policy']='none'
    path=output/method/'configuration.json';atomic_json(path,cfg)
    return str(path)

def status_path(output,method,action):return Path(output)/method/(action+'_status.json')

def assert_training_complete(output):
    for method in ['deterministic','balora']:
        p=status_path(output,method,'train')
        if not p.is_file() or not json.loads(p.read_text()).get('complete'):
            raise ValueError('Full held-out comparison requires completed new '+method+' training first')
        if not (Path(output)/method/'research/best.pt').is_file():raise FileNotFoundError('Selected new checkpoint missing')
