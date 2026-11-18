"""One wall-clock ledger for a complete, possibly multi-session research run."""
from pathlib import Path
import hashlib,json,math,os,time,uuid,zipfile
from auditlib.io_budget import atomic_json,digest_file,BudgetStop,AccountingError

ORIGINAL_ACCOUNTED_HOURS=20.884743572863048

def collect(input_root,output_root,resume_root=''):
    paths=set()
    for root in [input_root,output_root,resume_root]:
        root=Path(root) if root else None
        if root is None or not root.is_dir():continue
        paths.update(root.rglob('followup_ledger_*.json'))
        paths.update(root.rglob('full_run_ledger_*.json'))
    imported=Path(output_root)/'imported_ledgers';imported.mkdir(parents=True,exist_ok=True)
    root=Path(input_root)
    for archive in sorted(root.rglob('*.zip')) if root.is_dir() else []:
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                name=Path(item.filename).name
                if not name.endswith('.json') or not name.startswith(('followup_ledger_','full_run_ledger_')):continue
                if item.file_size>10*2**20:raise AccountingError('Unexpectedly large session ledger')
                raw=z.read(item);value=json.loads(raw)
                path=imported/('full_run_ledger_'+hashlib.sha256(raw).hexdigest()+'.json')
                atomic_json(path,value);paths.add(path)
    return sorted(paths)

def merge(paths,reconciliations=None):
    sessions={}
    for path in paths:
        x=json.loads(Path(path).read_text())
        if x.get('schema') not in ('balora_followup_ledger_v2','balora_full_run_ledger_v4'):
            raise AccountingError('Unsupported ledger schema: '+str(path))
        if abs(x['baseline_hours']-ORIGINAL_ACCOUNTED_HOURS)>1e-8:
            raise AccountingError('Historical experiment accounting was changed')
        for sid,s in x['sessions'].items():
            elapsed=float(s['elapsed_seconds'])
            if not math.isfinite(elapsed) or elapsed<0:raise AccountingError('Invalid recorded duration')
            if sid not in sessions or (elapsed,bool(s.get('closed')))>(sessions[sid]['elapsed_seconds'],bool(sessions[sid].get('closed'))):
                sessions[sid]=dict(s)
    for sid,value in (reconciliations or {}).items():
        if sid not in sessions or sessions[sid].get('closed'):raise AccountingError('Reconcile an existing unclosed session only')
        seconds=float(value['full_elapsed_seconds']);evidence=value.get('host_log_evidence','').strip()
        if not math.isfinite(seconds) or seconds<sessions[sid]['elapsed_seconds'] or not evidence:
            raise AccountingError('Reconciliation needs the full host duration and its evidence')
        sessions[sid].update(elapsed_seconds=seconds,closed=True,status='RECONCILED_HOST_LOG',host_log_evidence=evidence)
    for sid,s in sessions.items():
        if not s.get('closed'):raise AccountingError('Unclosed session '+sid+' requires the full host-log duration; a heartbeat is only a lower bound')
    return sessions

class Ledger:
    def __init__(self,output,input_root,resume_root,config,started=None,clock=None):
        self.clock=clock or time.time;self.start=self.clock() if started is None else float(started)
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.sessions=merge(collect(input_root,output,resume_root),config.get('reconciliations'))
        self.sid=uuid.uuid4().hex
        self.prior_hours=ORIGINAL_ACCOUNTED_HOURS+sum(s['elapsed_seconds'] for s in self.sessions.values())/3600
        self.limit=float(config.get('total_cumulative_hours',30));work=float(config.get('session_work_hours',10.5))
        if not math.isfinite(self.limit) or self.limit<=0 or not 0<work<=10.5:raise ValueError('Invalid cumulative/session allowance')
        self.working_ceiling=self.limit-.5
        available=min(work,self.working_ceiling-self.prior_hours)
        if available<=0:raise BudgetStop('Cumulative allowance exhausted; preserve ledgers and explicitly allocate more compute before continuing')
        self.deadline=self.start+available*3600-180
        self.path=self.output/('full_run_ledger_'+self.sid+'.json')
        self.sessions[self.sid]={'started_unix':self.start,'elapsed_seconds':0.,'closed':False,'status':'RUNNING'}
        self.tick()
    def tick(self,status='RUNNING',closed=False):
        self.sessions[self.sid].update(elapsed_seconds=max(0,self.clock()-self.start),last_update_unix=self.clock(),status=status,closed=closed)
        atomic_json(self.path,{'schema':'balora_full_run_ledger_v4','baseline_hours':ORIGINAL_ACCOUNTED_HOURS,
            'original_declared_cap_hours':30,'active_total_cumulative_cap_hours':self.limit,
            'explicit_expansion_over_original_hours':max(0,self.limit-30),
            'sessions':self.sessions,'accounted_hours':ORIGINAL_ACCOUNTED_HOURS+sum(s['elapsed_seconds'] for s in self.sessions.values())/3600,
            'time_unit':'pipeline wall time; concurrent GPUs share the same pipeline clock; not GPU-hours'})
    def check(self,needed=0):
        self.tick()
        if self.clock()+needed>=self.deadline:raise BudgetStop('Session/cumulative work deadline reached; full-state files are resumable')
    def close(self,status):self.tick(status,True)

def safe_extract(archive,destination):
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate ZIP paths')
        for name in names:
            if not (destination/name).resolve().is_relative_to(destination) or '\\' in name:raise ValueError('Unsafe ZIP path')
        if z.testzip() is not None:raise ValueError('Corrupt ZIP')
        z.extractall(destination)
    return destination
