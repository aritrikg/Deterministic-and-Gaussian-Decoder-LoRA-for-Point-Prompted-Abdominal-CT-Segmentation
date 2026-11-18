"""Runtime-only import routing and recorded failed-session carry-forward."""
import copy,hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def _resume_path(config):
 if config.get('resume_path'):return
 base=Path(config['input_root'])
 roots=sorted({p.parent for p in base.rglob('programme_identity.json') if 'source_snapshot' not in p.parts})
 if len(roots)==1:config['resume_path']=str(roots[0]);return
 if len(roots)>1:raise ValueError('Multiple v5 resume directories: set RESUME_PATH to the latest one')
 archives=sorted(base.rglob('BALoRA_Strengthening_v5_Results*.zip'))
 unique={}
 for path in archives:
  h=hashlib.sha256()
  with path.open('rb') as stream:
   for block in iter(lambda:stream.read(2**20),b''):h.update(block)
  unique.setdefault(h.hexdigest(),path)
 if len(unique)==1:config['resume_path']=str(next(iter(unique.values())))
 elif len(unique)>1:raise ValueError('Multiple different v5 result ZIPs: set RESUME_PATH to the latest one')

def run_with_import_guard(config,started):
 from v5 import orchestrator
 from v5.common import atomic
 run=orchestrator.run
 config=copy.deepcopy(config);output=Path(config['output_root']);output.mkdir(parents=True,exist_ok=True)
 recent=ROOT/'recent_accounting'
 if recent.is_dir():shutil.copytree(recent,output/'recent_accounting',dirs_exist_ok=True)
 _resume_path(config)
 original=subprocess.Popen;original_archive=orchestrator.archive;entry=ROOT/'worker_entry.py'
 attempt=config.get('_notebook_attempt_id','direct_launch')
 receipt={'execution_attempt_id':attempt,'started_unix':started,'mode':'RUN_PENDING','worker_commands_launched':0,'complete_internal_programme':False}
 def save_receipt():atomic(output/'execution_receipt.json',receipt)
 save_receipt()
 def pinned(command,*args,**kwargs):
  if isinstance(command,(list,tuple)) and len(command)>=3:
   script=Path(str(command[1]))
   if script.name=='worker.py' and script.parent.name=='v5':
    receipt['worker_commands_launched']+=1
    job=json.loads(Path(command[2]).read_text())
    receipt['last_action']=job.get('action');receipt['last_job']=Path(job.get('home','')).name
    print('EXECUTING',receipt['last_action'],receipt['last_job'],flush=True);save_receipt()
    command=[command[0],str(entry),*command[2:]]
  return original(command,*args,**kwargs)
 def marked_archive(output_root,ledger,status):
  status['execution_attempt_id']=attempt;status['worker_commands_launched']=receipt['worker_commands_launched']
  receipt['status']=status.get('status');receipt['complete_internal_programme']=bool(status.get('complete_internal_programme'));save_receipt()
  return original_archive(output_root,ledger,status)
 subprocess.Popen=pinned;orchestrator.archive=marked_archive
 try:return run(config,started=started)
 finally:subprocess.Popen=original;orchestrator.archive=original_archive
