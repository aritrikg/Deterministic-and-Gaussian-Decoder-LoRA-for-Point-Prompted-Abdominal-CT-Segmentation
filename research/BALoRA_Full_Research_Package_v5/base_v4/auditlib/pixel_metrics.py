"""Histogram uncertainty diagnostics with explicit approximation/conditioning."""
import numpy as np
def entropy(p):
 q=np.clip(np.asarray(p,dtype=np.float64),np.finfo(np.float32).eps,1-np.finfo(np.float32).eps)
 return -(q*np.log(q)+(1-q)*np.log1p(-q))
def calibration(p,y,mask=None,bins=15):
 p=np.asarray(p,dtype=np.float64);y=np.asarray(y,dtype=bool)
 if mask is not None:p=p[mask];y=y[mask]
 else:p=p.ravel();y=y.ravel()
 if not p.size:return {'n':0,'ece':None,'brier':None,'nll':None}
 ix=np.minimum((p*bins).astype(int),bins-1);n=np.bincount(ix,minlength=bins);sp=np.bincount(ix,weights=p,minlength=bins);sy=np.bincount(ix,weights=y,minlength=bins);q=np.clip(p,np.finfo(np.float32).eps,1-np.finfo(np.float32).eps)
 return {'n':int(p.size),'ece':float(np.abs(sp-sy).sum()/p.size),'brier':float(np.mean((p-y)**2)),'nll':float(np.mean(-(y*np.log(q)+(1-y)*np.log1p(-q)))),'bin_count':n.tolist(),'bin_sum_p':sp.tolist(),'bin_sum_y':sy.tolist()}
def ranking(score,error,mask=None,bins=256):
 s=np.asarray(score,dtype=np.float64);e=np.asarray(error,dtype=bool)
 if mask is not None:s=s[mask];e=e[mask]
 else:s=s.ravel();e=e.ravel()
 if not s.size:return {'n':0,'auroc_hist':None,'average_precision_hist':None,'risk_curve':[],'mean_uncertainty':None,'point_biserial':None}
 ix=np.minimum((np.clip(s/np.log(2),0,1)*bins).astype(int),bins-1);pos=np.bincount(ix,weights=e,minlength=bins);neg=np.bincount(ix,weights=~e,minlength=bins);P=float(pos.sum());N=float(neg.sum());tp=np.cumsum(pos[::-1]);fp=np.cumsum(neg[::-1]);trap=getattr(np,'trapezoid',None) or np.trapz
 auc=float(trap(np.r_[0,tp/P],np.r_[0,fp/N])) if P and N else None
 ap=float(np.sum(tp/np.maximum(tp+fp,1)*pos[::-1])/P) if P else None
 count=pos+neg;cum=np.cumsum(count);errors=np.cumsum(pos);curve=[]
 for coverage in np.linspace(.1,1,10):
  keep=coverage*s.size;i=min(int(np.searchsorted(cum,keep)),bins-1);pc=cum[i-1] if i else 0;pe=errors[i-1] if i else 0;fraction=(keep-pc)/count[i] if count[i] else 0
  curve.append({'coverage':float(coverage),'pixel_error_risk':float((pe+fraction*pos[i])/keep)})
 sd=float(s.std());ed=float(e.std());corr=float(np.mean((s-s.mean())*(e-e.mean()))/(sd*ed)) if sd>1e-15 and ed>0 else None
 return {'n':int(s.size),'errors':int(P),'error_prevalence':P/s.size,'auroc_hist':auc,'average_precision_hist':ap,'mean_uncertainty':float(s.mean()),'point_biserial':corr,'risk_curve':curve,'normalized_aurc_0p1_to_1':float(trap([x['pixel_error_risk'] for x in curve],[x['coverage'] for x in curve])/.9),'hist_errors':pos.astype(int).tolist(),'hist_correct':neg.astype(int).tolist(),'bins':bins,'approximation':'grouped histogram AUROC/AP; linear within-bin tie interpolation for partial coverage'}
def summarize_prediction(prob,target,epistemic=None,bins=256):
 from scipy.ndimage import binary_dilation,binary_erosion
 p=np.asarray(prob,dtype=np.float32);y=np.asarray(target,dtype=bool)
 if p.shape!=y.shape or p.ndim!=2 or not np.isfinite(p).all() or np.min(p)<0 or np.max(p)>1:raise ValueError('Invalid probability/target arrays')
 band=binary_dilation(y,iterations=2)^binary_erosion(y,iterations=2);error=(p>=.5)!=y
 out={'calibration':{n:calibration(p,y,m) for n,m in [('all',None),('GT_foreground',y),('GT_boundary_2px',band)]},'ranking':{},'epistemic_available':epistemic is not None}
 scores=[('predictive_entropy',entropy(p))]
 if epistemic is not None:scores.append(('epistemic_MI',np.asarray(epistemic,dtype=np.float32)))
 for name,score in scores:
  if score.shape!=p.shape or not np.isfinite(score).all() or score.min()<-1e-6 or score.max()>np.log(2)+1e-5:raise ValueError('Invalid uncertainty map')
  out['ranking'][name]={n:ranking(score,error,m,bins) for n,m in [('all',None),('GT_boundary_2px',band)]}
 return out
