"""Safe repeatable notebook launch, outside the unchanged research code."""
import copy,fcntl,hashlib,json,math,os,tempfile,time,traceback,uuid,zipfile
from pathlib import Path

def _read(path,default=None):
 path=Path(path)
 return json.loads(path.read_text()) if path.is_file() else default

def _write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.')
 try:
  with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)

def _host_info():
 try:
  boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
  ticks=float(Path('/proc/1/stat').read_text().rsplit(')',1)[1].split()[19])
  btime=next(float(s.split()[1]) for s in Path('/proc/stat').read_text().splitlines() if s.startswith('btime '))
  return hashlib.sha256(f'{boot}:{ticks}'.encode()).hexdigest()[:24],btime+ticks/os.sysconf('SC_CLK_TCK')
 except (OSError,ValueError,StopIteration,IndexError):return 'local_container',0.

def _paths(output):
 output=Path(output).resolve()
 return output.parent/(output.name+'_session_guard.json'),output.parent/(output.name+'_session_guard.lock')

def _sessions(output,reconciliations):
 sessions={}
 paths=sorted(set(Path(output).rglob('full_run_ledger_*.json'))|set(Path(output).rglob('followup_ledger_*.json')))
 for path in paths:
  for sid,record in _read(path).get('sessions',{}).items():
   item=dict(record);elapsed=float(item['elapsed_seconds']);started=float(item['started_unix'])
   if not math.isfinite(elapsed+started) or elapsed<0:raise ValueError('Invalid existing accounting: '+str(path))
   item['elapsed_seconds']=elapsed;item['started_unix']=started
   if sid not in sessions or (elapsed,bool(item.get('closed')))>(sessions[sid]['elapsed_seconds'],bool(sessions[sid].get('closed'))):sessions[sid]=item
 for sid,fix in reconciliations.items():
  if sid not in sessions or sessions[sid].get('closed'):continue
  elapsed=float(fix['full_elapsed_seconds'])
  if math.isfinite(elapsed) and elapsed>=sessions[sid]['elapsed_seconds'] and str(fix.get('host_log_evidence','')).strip():sessions[sid]={**sessions[sid],'elapsed_seconds':elapsed,'closed':True}
 return sessions

def _workers_launched(output,host_start):
 for path in Path(output).glob('jobs/*/*_job.json'):
  if float(_read(path,{}).get('deadline',0))>=host_start:return True
 return False

def _archive(output,hint=None):
 path=Path(hint) if hint else Path(output).parent/'BALoRA_Strengthening_v5_Results.zip'
 return path if path.is_file() else None

def _notice(status,message,archive):
 if status.get('status') in ('RUNNING','SESSION_LIMIT_REACHED','RECOVERY_REQUIRED') and archive is not None:
  status={**status,'previous_archive_preserved':str(archive)};archive=None
 return {'execution_called_this_click':False,**status,'notebook_notice':message},archive

def _current_archive(archive,attempt):
 if archive is None:return False
 try:
  with zipfile.ZipFile(archive) as z:
   record=json.loads(z.read('BALoRA_Strengthening_v5_Results/execution_receipt.json'))
   return record.get('execution_attempt_id')==attempt
 except (OSError,KeyError,ValueError,zipfile.BadZipFile):return False

def launch_or_show(config,notebook_start,runner,*,clock=None,host_info=None):
 """Run pending work by default; cache only a completed full programme.

 Retry intervals start after the latest closed interval and retain the original
 host deadline. No research file, ledger or checkpoint is deleted or edited.
 """
 clock=clock or time.time;now=float(clock());output=Path(config['output_root']).resolve();output.mkdir(parents=True,exist_ok=True)
 guard_path,lock_path=_paths(output)
 with lock_path.open('a+') as lock:
  try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return _notice({'status':'RUNNING','complete_internal_programme':False},'An execution is active for this output. No duplicate jobs were launched.',_archive(output))
  try:
   host,birth=host_info or _host_info();requested=float(config.get('session_work_hours',10.5))
   if not 0<requested<=10.5:raise ValueError('session_work_hours must be in (0,10.5]')
   sessions=_sessions(output,config.get('reconciliations',{}));unclosed=[sid for sid,s in sessions.items() if not s.get('closed')]
   if unclosed:return _notice({'status':'RECOVERY_REQUIRED','complete_internal_programme':False,'reason':'Unclosed accounting requires the actual full host-log duration: '+', '.join(unclosed)},'Saved files were preserved. Reconcile the recorded session before resuming; no new jobs were launched.',_archive(output))
   state=_read(guard_path,{})
   current=[s for s in sessions.values() if birth<=s['started_unix']<=now+1]
   if state.get('host')!=host:
    anchor=min([float(notebook_start)]+[s['started_unix'] for s in current])
    state={'host':host,'host_start':anchor,'work_hours':requested,'attempted':bool(current),'phase':'IDLE','cached_status':None,'archive_path':None}
   else:
    state['host_start']=min(float(state['host_start']),float(notebook_start));state['work_hours']=min(float(state['work_hours']),requested)
   anchor=state['host_start'];deadline=anchor+state['work_hours']*3600;launched=_workers_launched(output,anchor)
   saved=state.get('cached_status') or (_read(output/'run_status.json') if state.get('attempted') else None)
   action=config.get('execution_action','RUN_PENDING')
   if action not in ('RUN_PENDING','RESULTS_ONLY'):raise ValueError('execution_action must be RUN_PENDING or RESULTS_ONLY')
   if action=='RESULTS_ONLY' or (saved and saved.get('complete_internal_programme') is True):
    saved=saved or {'status':'NOT_STARTED','complete_internal_programme':False}
    message='Showing saved results because RESULTS_ONLY was selected or the full programme is complete.'
    if action=='RESULTS_ONLY' and not saved.get('complete_internal_programme'):message+=' Select RUN_PENDING to execute unfinished stages.'
    _write(guard_path,state);return _notice({**saved,'execution_called_this_click':False},message,_archive(output,state.get('archive_path')))
   if now>=deadline-300:return _notice({'status':'SESSION_LIMIT_REACHED','complete_internal_programme':False,'reason':'The original session deadline is reached; repeated clicks do not extend it.'},'Save available output and resume in a new Kaggle execution session.',_archive(output,state.get('archive_path')))
   attempt_start=max([anchor]+[s['started_unix']+s['elapsed_seconds'] for s in current])
   if attempt_start>now+1:raise ValueError('Saved accounting ends after the current clock')
   effective=copy.deepcopy(config);effective['session_work_hours']=(deadline-attempt_start)/3600
   effective['_notebook_attempt_id']=uuid.uuid4().hex
   print('RUN_PENDING: executing unfinished stages; completed checkpoints and records will be reused.',flush=True)
   state.update(attempted=True,phase='RUNNING',last_attempt_start=attempt_start,host_deadline=deadline);_write(guard_path,state)
   try:
    status,archive=runner(effective,started=attempt_start)
    if not isinstance(status,dict):raise TypeError('Research runner returned no status dictionary')
   except BaseException as exc:
    post=_sessions(output,config.get('reconciliations',{}));open_after=any(not s.get('closed') for s in post.values())
    status={'status':'RECOVERY_REQUIRED' if open_after else 'FAILED_STARTUP','complete_internal_programme':False,'reason':str(exc),'error_type':type(exc).__name__}
    _write(output/'notebook_launch_error.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()});archive=_archive(output)
   state.update(phase='RETURNED',cached_status=status,archive_path=str(archive) if archive else None);_write(guard_path,state)
   fresh=_current_archive(archive,effective['_notebook_attempt_id'])
   status={**status,'execution_called_this_click':True,'execution_attempt_id':effective['_notebook_attempt_id'],'archive_is_current_attempt':fresh}
   if archive is not None and not fresh:status['previous_archive_preserved']=str(archive)
   message='Execution attempted pending work. Read its current status and reason; a failed or partial run is not completion.'
   if status.get('status') in ('FAILED','FAILED_STARTUP','RECOVERY_REQUIRED') and not _workers_launched(output,anchor):message='Startup stopped before workers launched. Correct the reported cause and rerun this cell; accounting and the original deadline are retained.'
   return _notice(status,message,_archive(output,archive) if fresh else None)
  except (ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:return _notice({'status':'RECOVERY_REQUIRED','complete_internal_programme':False,'reason':str(exc)},'Existing accounting/session metadata needs inspection. No new jobs were launched.',_archive(output))
  finally:fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
