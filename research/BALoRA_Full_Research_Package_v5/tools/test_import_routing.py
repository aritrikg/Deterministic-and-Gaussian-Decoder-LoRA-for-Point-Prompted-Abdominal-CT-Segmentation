"""Reproduce the uploaded wrong-training-package failure without CUDA."""
import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Routing(unittest.TestCase):
 def test_official_training_collision_and_cached_module(self):
  with tempfile.TemporaryDirectory() as d:
   official=Path(d)/'official';(official/'training').mkdir(parents=True);(official/'sam2').mkdir()
   (official/'training/__init__.py').write_text('')
   (official/'training/train.py').write_text('raise RuntimeError("WRONG_OFFICIAL_TRAINER")\n')
   (official/'sam2/__init__.py').write_text('')
   job={'source_root':str(ROOT),'official_root':str(official)};p=Path(d)/'job.json';p.write_text(json.dumps(job))
   script='''import sys,json,importlib
from pathlib import Path
job=json.loads(Path(sys.argv[1]).read_text())
sys.path[:0]=[job['official_root'],job['source_root']]
import training
assert Path(training.__file__).is_relative_to(Path(job['official_root']))
from worker_entry import prepare_worker
root=prepare_worker(job)
# The unchanged worker again prepends official_root; the pinned package must persist.
sys.path.insert(0,job['official_root'])
t=importlib.import_module('training.train')
assert Path(t.__file__).resolve().is_relative_to(root/'base_v4/source')
assert hasattr(t,'train_one_epoch') and hasattr(t,'_make_grad_scaler')
print('Correct BALoRA trainer bound despite the official SAM2 training package')
'''
   result=subprocess.run([sys.executable,'-c',script,str(p)],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
   self.assertEqual(result.returncode,0,result.stdout)
 def test_actual_worker_launcher_import_check(self):
  with tempfile.TemporaryDirectory() as d:
   official=Path(d)/'official';(official/'training').mkdir(parents=True);(official/'sam2').mkdir()
   (official/'training/__init__.py').write_text('raise RuntimeError("WRONG_TRAINER_SELECTED")')
   (official/'sam2/__init__.py').write_text('')
   p=Path(d)/'job.json';p.write_text(json.dumps({'source_root':str(ROOT),'official_root':str(official)}))
   result=subprocess.run([sys.executable,str(ROOT/'worker_entry.py'),'--check-imports',str(p)],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
   self.assertEqual(result.returncode,0,result.stdout);self.assertIn('BALoRA training import:',result.stdout)
 def test_resume_copy_suffix_is_discovered(self):
  sys.path.insert(0,str(ROOT));from runtime_launch import _resume_path
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'BALoRA_Strengthening_v5_Results (1).zip';p.write_bytes(b'fixture')
   cfg={'input_root':d,'resume_path':''};_resume_path(cfg);self.assertEqual(cfg['resume_path'],str(p))
 def test_extracted_resume_is_discovered(self):
  sys.path.insert(0,str(ROOT));from runtime_launch import _resume_path
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'dataset/BALoRA_Strengthening_v5_Results';root.mkdir(parents=True);(root/'programme_identity.json').write_text('{}')
   cfg={'input_root':d,'resume_path':''};_resume_path(cfg);self.assertEqual(cfg['resume_path'],str(root))
 def test_recorded_failed_hours_are_carried(self):
  sys.path[:0]=[str(ROOT),str(ROOT/'base_v4')]
  from run_budget import merge,ORIGINAL_ACCOUNTED_HOURS
  files=list((ROOT/'evidence').glob('full_run_ledger_*.json'))+list((ROOT/'recent_accounting').glob('full_run_ledger_*.json'))
  sessions=merge(files+files);total=ORIGINAL_ACCOUNTED_HOURS+sum(s['elapsed_seconds'] for s in sessions.values())/3600
  self.assertAlmostEqual(total,35.9096370237325)
 def test_identical_resume_copies_are_not_ambiguous(self):
  sys.path.insert(0,str(ROOT));from runtime_launch import _resume_path
  with tempfile.TemporaryDirectory() as d:
   a=Path(d)/'BALoRA_Strengthening_v5_Results (1).zip';b=Path(d)/'BALoRA_Strengthening_v5_Results (2).zip'
   a.write_bytes(b'identical fixture');b.write_bytes(a.read_bytes());cfg={'input_root':d,'resume_path':''};_resume_path(cfg);self.assertIn(cfg['resume_path'],[str(a),str(b)])
if __name__=='__main__':unittest.main(verbosity=2)
