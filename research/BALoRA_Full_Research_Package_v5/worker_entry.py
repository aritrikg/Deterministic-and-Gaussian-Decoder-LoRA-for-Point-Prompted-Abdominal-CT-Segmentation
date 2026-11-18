"""Bind project namespaces before executing the unchanged v5 worker.

The official SAM2 repository also has a top-level `training` package. Binding
BALoRA's packages first prevents `training.train` resolving to SAM2's launcher.
"""
import importlib,json,runpy,sys
from pathlib import Path

def prepare_worker(job):
 root=Path(job['source_root']).resolve();source=root/'base_v4/source';official=Path(job['official_root']).resolve()
 sys.path[:0]=[str(source),str(root/'base_v4'),str(root),str(official)]
 for name in ('training','models','datasets','losses','utils','evaluation'):
  old=sys.modules.get(name);location=getattr(old,'__file__',None)
  if old is not None and (not location or not Path(location).resolve().is_relative_to(source)):
   for key in list(sys.modules):
    if key==name or key.startswith(name+'.'):del sys.modules[key]
  module=importlib.import_module(name)
  if not Path(module.__file__).resolve().is_relative_to(source):raise ImportError('Wrong project namespace: '+name)
 training=importlib.import_module('training.train')
 if not Path(training.__file__).resolve().is_relative_to(source):raise ImportError('Wrong BALoRA training implementation')
 sam2=importlib.import_module('sam2')
 if not Path(sam2.__file__).resolve().is_relative_to(official):raise ImportError('Wrong official SAM2 provenance')
 print('BALoRA training import:',training.__file__,flush=True)
 print('Official SAM2 import:',sam2.__file__,flush=True)
 return root

if __name__=='__main__':
 check=len(sys.argv)>1 and sys.argv[1]=='--check-imports';path=sys.argv[2] if check else sys.argv[1]
 job=json.loads(Path(path).read_text());root=prepare_worker(job)
 if not check:
  sys.argv=[str(root/'v5/worker.py'),path]
  runpy.run_path(str(root/'v5/worker.py'),run_name='__main__')
