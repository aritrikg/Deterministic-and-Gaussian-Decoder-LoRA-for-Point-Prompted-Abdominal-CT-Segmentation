"""Pinned input resolution; import official SAM2 before original model modules."""
from pathlib import Path
import hashlib,importlib,importlib.metadata,importlib.util,json,os,subprocess,sys,urllib.request,zipfile,time
from .io_budget import atomic_json,digest_file,digest_json,BudgetStop
COMMIT='2b90b9f5ceec907a1c18123530e92e794ad901a4'
BACKBONE='a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5'
SELECTED={'balora':'8536f08e85dfd94eaa364513ee366b10b090e8948e00853e1a7bbd1f24113436','deterministic':'1ca0c623a3db41033710960c86dc493113e1e5705c4f92a27eb42b975fa5ee9d'}
def verify_bundle(root):
 root=Path(root);manifest=json.loads((root/'source_hashes.json').read_text())
 for rel,expected in manifest.items():
  if digest_file(root/rel)!=expected:raise ValueError('Bundle source/reference mismatch: '+rel)
 return digest_json(manifest)
def selected_checkpoint(root,kind,destination,override=''):
 expected=SELECTED[kind];dest=Path(destination)/(kind+'_selected.pt');dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.is_file() and digest_file(dest)==expected:return str(dest)
 root=Path(override) if override else Path(root)
 if root.is_file() and root.suffix=='.pt':
  if digest_file(root)!=expected:raise ValueError('Checkpoint checksum mismatch')
  return str(root)
 for f in root.rglob('best.pt') if root.is_dir() else []:
  if digest_file(f)==expected:return str(f)
 archives=[root] if root.is_file() else sorted(root.rglob('*.zip'),key=lambda p:(not ('fold1_20epochs' in p.name if kind=='balora' else 'Deterministic_LoRA_Control_Results' in p.name),str(p)))
 for archive in archives:
  with zipfile.ZipFile(archive) as z:
   for member in z.namelist():
    if not member.endswith('best.pt'):continue
    h=hashlib.sha256();tmp=dest.with_suffix('.tmp')
    with z.open(member) as src,tmp.open('wb') as target:
     for block in iter(lambda:src.read(2**20),b''):h.update(block);target.write(block)
    if h.hexdigest()==expected:os.replace(tmp,dest);return str(dest)
    tmp.unlink(missing_ok=True)
 raise FileNotFoundError('Attach the completed '+kind+' results archive/checkpoint with SHA256 '+expected)
def setup_sam2(work,input_root,checkpoint_override='',install_missing=True,deadline=None):
 def remaining_timeout(maximum):
  left=float('inf') if deadline is None else deadline-time.time()
  if left<=0:raise BudgetStop('Setup deadline reached')
  return max(.1,min(maximum,left))
 import torch
 if not torch.cuda.is_available():raise RuntimeError('Select a CUDA GPU for this notebook; use the CPU reanalysis notebook for saved records')
 dependencies={'hydra':'hydra-core==1.3.2','iopath':'iopath==0.1.10','nibabel':'nibabel==5.3.2','yaml':'PyYAML==6.0.2','tensorboard':'tensorboard==2.20.0','scipy':'scipy==1.16.3','matplotlib':'matplotlib==3.10.8'}
 missing=[v for k,v in dependencies.items() if importlib.util.find_spec(k) is None]
 if missing:
  if not install_missing:raise ImportError('Missing non-core dependencies: '+str(missing))
  subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check',*missing],check=True,timeout=remaining_timeout(600));importlib.invalidate_caches()
 work=Path(work);work.mkdir(parents=True,exist_ok=True);root=work/'sam2_pinned'
 if not (root/'.git').is_dir():subprocess.run(['git','clone','--filter=blob:none','https://github.com/facebookresearch/sam2.git',str(root)],check=True,timeout=remaining_timeout(300))
 got=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
 if got!=COMMIT:
  subprocess.run(['git','fetch','origin',COMMIT],cwd=root,check=True,timeout=remaining_timeout(300));subprocess.run(['git','checkout','--detach',COMMIT],cwd=root,check=True,timeout=remaining_timeout(60))
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==COMMIT
 subprocess.run(['git','diff','--exit-code','HEAD','--'],cwd=root,check=True,timeout=remaining_timeout(30))
 if 'sam2' in sys.modules and not Path(sys.modules['sam2'].__file__).resolve().is_relative_to(root.resolve()):raise RuntimeError('Different sam2 already imported; restart this notebook with setup first')
 sys.path.insert(0,str(root));import sam2
 from sam2.build_sam import build_sam2
 assert Path(sam2.__file__).resolve().is_relative_to(root.resolve()) and callable(build_sam2)
 target=Path(checkpoint_override) if checkpoint_override else work/'sam2.1_hiera_base_plus.pt'
 if checkpoint_override and digest_file(target)!=BACKBONE:raise ValueError('Backbone checksum mismatch')
 if not target.is_file() or digest_file(target)!=BACKBONE:
  found=next((f for f in Path(input_root).rglob('sam2.1_hiera_base_plus.pt') if digest_file(f)==BACKBONE),None)
  if found:target=found
  else:
   tmp=target.with_suffix('.download')
   with urllib.request.urlopen('https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt',timeout=remaining_timeout(30)) as response,tmp.open('wb') as f:
    while True:
     remaining_timeout(30);block=response.read(2**20)
     if not block:break
     f.write(block)
   if digest_file(tmp)!=BACKBONE:tmp.unlink(missing_ok=True);raise ValueError('Downloaded backbone checksum mismatch')
   os.replace(tmp,target)
 versions={}
 for package in ['torch','numpy','hydra-core','iopath','nibabel','scipy','matplotlib','tensorboard']:
  try:versions[package]=importlib.metadata.version(package)
  except importlib.metadata.PackageNotFoundError:versions[package]='unavailable'
 return {'official_root':str(root),'backbone_checkpoint':str(target),'official_commit':COMMIT,'backbone_sha256':BACKBONE,'versions':versions,'python':sys.version,'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'sam2_import':sam2.__file__}
def make_splits(reference,work):
 result={}
 for role,ids in reference['split_case_ids'].items():
  path=Path(work)/'splits'/(role+'.json');atomic_json(path,{'cases':ids});result[role]=str(path)
 assert len(set(sum(reference['split_case_ids'].values(),[])))==30
 return result
