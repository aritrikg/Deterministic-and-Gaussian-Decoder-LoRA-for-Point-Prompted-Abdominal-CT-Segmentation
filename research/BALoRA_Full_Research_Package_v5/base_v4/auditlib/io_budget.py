"""Atomic records and cumulative ledger; no PyTorch dependency."""
from pathlib import Path
import hashlib,json,os,tempfile,time,uuid
BASELINE_HOURS=20.884743572863048
class BudgetStop(RuntimeError):pass
class AccountingError(RuntimeError):pass
def digest_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
 return h.hexdigest()
def digest_json(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def atomic_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.tmp')
 try:
  with os.fdopen(fd,'w') as f:json.dump(value,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def merge_ledgers(paths):
 sessions={}
 for path in paths:
  x=json.loads(Path(path).read_text())
  if x.get('schema')!='balora_followup_ledger_v2' or abs(x['baseline_hours']-BASELINE_HOURS)>1e-8:raise AccountingError('Ledger baseline/schema mismatch')
  for sid,s in x['sessions'].items():
   if s['elapsed_seconds']<0:raise AccountingError('Negative elapsed time')
   if sid not in sessions or (s['elapsed_seconds'],bool(s.get('closed')))>(sessions[sid]['elapsed_seconds'],bool(sessions[sid].get('closed'))):sessions[sid]=s
 for sid,s in sessions.items():
  if not s.get('closed'):raise AccountingError('Unclosed prior session '+sid+' requires full-duration reconciliation from the Kaggle log; a heartbeat is only a lower bound')
 return sessions
class Budget:
 def __init__(self,root,prior_paths=(),start_unix=None,work_hours=3,clock=None):
  self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.clock=clock or time.time;self.start=self.clock() if start_unix is None else float(start_unix)
  if not 0<work_hours<=10.5:raise ValueError('Work cap must be in (0,10.5] hours')
  self.sessions=merge_ledgers(prior_paths);self.sid=uuid.uuid4().hex
  self.prior_hours=BASELINE_HOURS+sum(s['elapsed_seconds'] for s in self.sessions.values())/3600
  available=min(work_hours,29.5-self.prior_hours)
  if available<=0:raise BudgetStop('No working allowance remains')
  self.deadline=self.start+available*3600-120
  self.path=self.root/('followup_ledger_'+self.sid+'.json');self.sessions[self.sid]={'started_unix':self.start,'elapsed_seconds':0,'closed':False,'status':'RUNNING'};self.tick()
 def tick(self,status='RUNNING',closed=False):
  self.sessions[self.sid].update(elapsed_seconds=max(0,self.clock()-self.start),closed=closed,status=status,last_update_unix=self.clock())
  atomic_json(self.path,{'schema':'balora_followup_ledger_v2','baseline_hours':BASELINE_HOURS,'sessions':self.sessions,'accounted_hours':BASELINE_HOURS+sum(s['elapsed_seconds'] for s in self.sessions.values())/3600})
 def check(self,needed_seconds=0):
  self.tick()
  if self.clock()+needed_seconds>=self.deadline:raise BudgetStop('Cooperative deadline reached; resume saved records')
 def admit(self,seconds,margin=1.30):self.check(seconds*margin);return {'estimate_seconds':seconds,'margin':margin,'deadline':self.deadline}
 def close(self,status):self.tick(status,True)
class WorkerBudget:
 def __init__(self,path,sid,deadline):self.path=Path(path);self.sid=sid;self.deadline=deadline
 def check(self,needed_seconds=0):
  x=json.loads(self.path.read_text());s=x['sessions'][self.sid];now=time.time();s.update(elapsed_seconds=max(s['elapsed_seconds'],now-s['started_unix']),last_update_unix=now)
  x['accounted_hours']=x['baseline_hours']+sum(v['elapsed_seconds'] for v in x['sessions'].values())/3600;atomic_json(self.path,x)
  if now+needed_seconds>=self.deadline:raise BudgetStop('Worker deadline reached')

def discover_ledgers(input_root,output_root,resume_root=''):
 """Recover follow-up ledgers from attached folders or ZIPs before admission."""
 import zipfile
 output_root=Path(output_root);paths=list(output_root.rglob('followup_ledger_*.json'))
 roots=[Path(input_root)]+([Path(resume_root)] if resume_root else [])
 imported=output_root/'imported_ledgers';imported.mkdir(parents=True,exist_ok=True)
 for root in roots:
  if not root.is_dir():continue
  paths.extend(root.rglob('followup_ledger_*.json'))
  for archive in root.rglob('*.zip'):
   with zipfile.ZipFile(archive) as z:
    for item in z.infolist():
     name=Path(item.filename).name
     if not (name.startswith('followup_ledger_') and name.endswith('.json')):continue
     if item.file_size>10*2**20:raise AccountingError('Unexpectedly large ledger in '+str(archive))
     raw=z.read(item);x=json.loads(raw)
     if x.get('schema')!='balora_followup_ledger_v2':raise AccountingError('Unexpected follow-up ledger schema')
     target=imported/('followup_ledger_'+hashlib.sha256(raw).hexdigest()+'.json');atomic_json(target,x);paths.append(target)
 return sorted(set(paths))
