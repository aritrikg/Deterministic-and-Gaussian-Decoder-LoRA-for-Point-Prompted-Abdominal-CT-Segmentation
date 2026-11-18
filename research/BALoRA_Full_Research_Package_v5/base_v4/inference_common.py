from pathlib import Path
import hashlib,json,os,tempfile
import numpy as np
from auditlib.io_budget import digest_file

def seed_for(base,cid,z,organ):return int.from_bytes(hashlib.sha256(f'{base}:{cid}:{z}:{organ}'.encode()).digest()[:4],'little')%(2**31-1)

def fingerprint(model):
 h=hashlib.sha256()
 for name,p in model.named_parameters():
  if p.requires_grad:h.update(name.encode());h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def predict_one(model,method,embedding,target,base_seed,cid,z,organ,expected_point=None):
 import torch
 from utils.prompting import sample_foreground_points
 torch.manual_seed(seed_for(base_seed,cid,z,organ));points=sample_foreground_points(target)
 xy=[int(x) for x in points[0,0].detach().cpu().tolist()]
 if expected_point is not None and xy!=[int(v) for v in expected_point]:raise ValueError('Saved prompt-coordinate mismatch for '+f'{cid}:{z}:{organ}')
 labels=torch.ones((1,1),dtype=torch.int32,device=target.device)
 if method=='balora':
  from models.confidence import estimate_uncertainty
  logits,probe,ranks=model.decode_with_adaptive_rank(embedding,points,labels,k_samples=4);u=estimate_uncertainty(logits.squeeze(2));p=u.mean_prob[0].detach();epi=u.epistemic[0].detach()
 else:
  logits=model.decode_one(embedding,points,labels);p=logits.sigmoid().reshape(*target.shape[-2:]).detach();epi=None;ranks={}
 return p,epi,ranks,xy

def save_arrays(path,image,target,p,epi,identity):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.tmp')
 try:
  with os.fdopen(fd,'wb') as f:
   values={'windowed_CT':np.asarray(image,dtype=np.float16),'ground_truth':np.asarray(target,dtype=np.uint8),'prediction_FP32_threshold':(p>=.5).astype(np.uint8),'mean_probability_FP16':p.astype(np.float16),'identity_json':np.asarray(json.dumps(identity,sort_keys=True))}
   if epi is not None:values['epistemic_MI_FP16']=epi.astype(np.float16)
   np.savez_compressed(f,**values);f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
 return digest_file(path)
