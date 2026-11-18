import fcntl,importlib.util,json,tempfile,unittest,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('entry',ROOT/'notebook_entry.py');entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
HOST=('fixture',900.)
class EntryChecks(unittest.TestCase):
 def cfg(self,d):return {'output_root':str(Path(d)/'run'),'input_root':str(Path(d)/'input'),'session_work_hours':1.,'total_cumulative_hours':45.}
 def runner(self,clock,status='FULL_INTERNAL_PROGRAMME_COMPLETE',workers=False):
  calls=[]
  def run(cfg,started):
   calls.append((cfg,started));root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True);sid=str(len(calls))
   (root/('full_run_ledger_'+sid+'.json')).write_text(json.dumps({'sessions':{sid:{'started_unix':started,'elapsed_seconds':clock[0]-started,'closed':True}}}))
   if workers:
    p=root/'jobs/fixture/profile_job.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps({'deadline':4600.}))
   result={'status':status,'complete_internal_programme':status=='FULL_INTERNAL_PROGRAMME_COMPLETE','reason':'underlying fixture reason'}
   (root/'run_status.json').write_text(json.dumps(result));archive=root.parent/'BALoRA_Strengthening_v5_Results.zip';archive.write_bytes(b'fixture');return result,archive
  return run,calls
 def launch(self,cfg,start,run,clock):return entry.launch_or_show(cfg,start,run,clock=lambda:clock[0],host_info=HOST)
 def test_complete_repeat_does_not_relaunch(self):
  with tempfile.TemporaryDirectory() as d:
   cfg=self.cfg(d);clock=[1020.];run,calls=self.runner(clock);self.launch(cfg,1000.,run,clock);r=self.launch(cfg,1000.,run,clock)
   self.assertEqual(len(calls),1);self.assertTrue(r[0]['complete_internal_programme'])
 def test_import_failure_does_not_latch(self):
  with tempfile.TemporaryDirectory() as d:
   cfg=self.cfg(d);clock=[1020.]
   def bad(*a,**k):raise ImportError('fixture dependency')
   self.assertEqual(self.launch(cfg,1000.,bad,clock)[0]['status'],'FAILED_STARTUP');run,calls=self.runner(clock);self.launch(cfg,1000.,run,clock);self.assertEqual(len(calls),1)
 def test_failed_startup_retains_deadline_and_intervals(self):
  with tempfile.TemporaryDirectory() as d:
   cfg=self.cfg(d);clock=[1020.];run,calls=self.runner(clock,status='FAILED');self.launch(cfg,1000.,run,clock);clock[0]=1110.;self.launch(cfg,1000.,run,clock)
   self.assertEqual([x[1] for x in calls],[1000.,1020.])
   for c,s in calls:self.assertAlmostEqual(s+c['session_work_hours']*3600,4600.)
   self.assertAlmostEqual(sum(x['elapsed_seconds'] for x in entry._sessions(cfg['output_root'],{}).values()),110.)
   self.assertEqual(cfg['session_work_hours'],1.)
 def test_failed_worker_closed_state_can_resume(self):
  with tempfile.TemporaryDirectory() as d:
   cfg=self.cfg(d);clock=[1020.];run,calls=self.runner(clock,status='FAILED',workers=True);self.launch(cfg,1000.,run,clock);r=self.launch(cfg,1000.,run,clock)
   self.assertEqual(len(calls),2);self.assertEqual(r[0]['reason'],'underlying fixture reason')
 def test_pause_can_continue_before_original_deadline(self):
  with tempfile.TemporaryDirectory() as d:
   cfg=self.cfg(d);clock=[1020.];run,calls=self.runner(clock,status='PAUSED');self.launch(cfg,1000.,run,clock);self.launch(cfg,1000.,run,clock);self.assertEqual(len(calls),2)
 def test_kernel_restart_does_not_reset_deadline(self):
  with tempfile.TemporaryDirectory() as d:
   cfg=self.cfg(d);clock=[1020.];run,calls=self.runner(clock,status='FAILED');self.launch(cfg,1000.,run,clock);clock[0]=1200.;self.launch(cfg,1150.,run,clock)
   c,s=calls[1];self.assertAlmostEqual(s+c['session_work_hours']*3600,4600.)
 def test_unclosed_ledger_is_preserved(self):
  with tempfile.TemporaryDirectory() as d:
   cfg=self.cfg(d);p=Path(cfg['output_root'])/'full_run_ledger_x.json';p.parent.mkdir();raw=json.dumps({'sessions':{'x':{'started_unix':1000.,'elapsed_seconds':10.,'closed':False}}});p.write_text(raw)
   def forbidden(*a,**k):self.fail('must not launch')
   self.assertEqual(self.launch(cfg,1000.,forbidden,[1020.])[0]['status'],'RECOVERY_REQUIRED');self.assertEqual(p.read_text(),raw)
 def test_concurrent_lock_blocks_duplicate(self):
  with tempfile.TemporaryDirectory() as d:
   cfg=self.cfg(d);Path(cfg['output_root']).mkdir();_,p=entry._paths(cfg['output_root'])
   with p.open('a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def forbidden(*a,**k):self.fail('duplicate launch')
    self.assertEqual(self.launch(cfg,1000.,forbidden,[1020.])[0]['status'],'RUNNING')
 def test_legacy_completed_output_is_reused(self):
  with tempfile.TemporaryDirectory() as d:
   cfg=self.cfg(d);clock=[1020.];run,calls=self.runner(clock);run(cfg,1000.)
   def forbidden(*a,**k):self.fail('legacy duplicate')
   self.assertTrue(self.launch(cfg,1000.,forbidden,clock)[0]['complete_internal_programme'])
 def test_expired_session_is_not_extended(self):
  with tempfile.TemporaryDirectory() as d:
   cfg=self.cfg(d)
   def bad(*a,**k):raise ImportError('fixture')
   self.launch(cfg,1000.,bad,[1020.])
   def forbidden(*a,**k):self.fail('session extended')
   self.assertEqual(self.launch(cfg,4500.,forbidden,[4500.])[0]['status'],'SESSION_LIMIT_REACHED')
 def test_actual_research_ledger_keeps_historical_hours(self):
  sys.path[:0]=[str(ROOT),str(ROOT/'base_v4')]
  from run_budget import Ledger
  with tempfile.TemporaryDirectory() as d:
   cfg=self.cfg(d);out=Path(cfg['output_root']);shutil.copytree(ROOT/'evidence',out/'carried_accounting');clock=[1020.];calls=[]
   def run(c,started):
    ledger=Ledger(out,c['input_root'],'',c,started=started,clock=lambda:clock[0]);calls.append((ledger.start,ledger.deadline,ledger.prior_hours));clock[0]+=10;ledger.close('FAILED_STARTUP');return {'status':'FAILED','complete_internal_programme':False,'reason':'fixture'},None
   self.launch(cfg,1000.,run,clock);clock[0]=1110.;self.launch(cfg,1000.,run,clock)
   self.assertEqual([c[0] for c in calls],[1000.,1030.]);self.assertEqual([c[1] for c in calls],[4420.,4420.]);self.assertAlmostEqual(calls[0][2],35.726407254264196);self.assertAlmostEqual(calls[1][2]-calls[0][2],30/3600)
 def test_results_only_never_launches(self):
  with tempfile.TemporaryDirectory() as d:
   cfg=self.cfg(d);cfg['execution_action']='RESULTS_ONLY'
   def forbidden(*a,**k):self.fail('RESULTS_ONLY must not launch')
   r=self.launch(cfg,1000.,forbidden,[1020.]);self.assertEqual(r[0]['status'],'NOT_STARTED');self.assertFalse(r[0]['execution_called_this_click'])
 def test_failed_attempt_never_returns_old_zip_as_new(self):
  import zipfile
  with tempfile.TemporaryDirectory() as d:
   cfg=self.cfg(d);archive=Path(d)/'BALoRA_Strengthening_v5_Results.zip'
   with zipfile.ZipFile(archive,'w') as z:z.writestr('BALoRA_Strengthening_v5_Results/execution_receipt.json',json.dumps({'execution_attempt_id':'old'}))
   def broken(*a,**k):raise ImportError('fixture startup failure')
   r=self.launch(cfg,1000.,broken,[1020.]);self.assertIsNone(r[1]);self.assertFalse(r[0]['archive_is_current_attempt']);self.assertTrue(archive.exists())
if __name__=='__main__':unittest.main(verbosity=2)
