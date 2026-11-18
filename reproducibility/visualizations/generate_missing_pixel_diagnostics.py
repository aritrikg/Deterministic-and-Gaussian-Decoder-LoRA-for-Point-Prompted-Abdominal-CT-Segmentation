"""Exact per-pixel BALoRA confusion and reliability figures.

Requires true saved inference artifacts, not mean Dice scores.
Layout:
  input_root/
      frozen/*.npz
      deterministic/*.npz
      balora/*.npz
Each NPZ must contain binary pred_mask, binary gt_mask, and foreground probability_map
with identical shapes (each example is one organ mask from one test prompt).

Only run on the SAME held-out split and matched organ-prompt definition used
for reported figures. Predictions must be the exact chosen inference output.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS=['frozen','deterministic','balora']
MODEL_NAMES=['Frozen SAM2','Deterministic LoRA','BALoRA']

def compute(folder,bins=15):
 n=np.zeros(bins,dtype=np.int64);sp=np.zeros(bins,dtype=np.float64);sy=np.zeros(bins,dtype=np.float64)
 tp=fp=tn=fn=total=0
 paths=sorted(folder.glob('*.npz'))
 if not paths:raise ValueError(f'No NPZ inference files in {folder}')
 for path in paths:
  with np.load(path,allow_pickle=False) as data:
   need=['pred_mask','gt_mask','probability_map']
   absent=[k for k in need if k not in data]
   if absent:raise ValueError(f'{path}: missing {absent}')
   raw_p,raw_y,raw_prob=(np.asarray(data[k]) for k in need)
   if raw_p.shape!=raw_y.shape or raw_p.shape!=raw_prob.shape:raise ValueError(f'{path}: incompatible array shapes')
   if not np.all(np.isin(raw_p,[0,1])) or not np.all(np.isin(raw_y,[0,1])):raise ValueError(f'{path}: mask arrays are not binary')
   if not np.isfinite(raw_prob).all() or np.any((raw_prob<0)|(raw_prob>1)):raise ValueError(f'{path}: probabilities outside [0,1]')
   p=raw_p.ravel().astype(bool);y=raw_y.ravel().astype(bool);q=raw_prob.ravel().astype(float)
   tp+=int((p&y).sum());fp+=int((p&~y).sum());tn+=int((~p&~y).sum());fn+=int((~p&y).sum());total+=len(p)
   idx=np.minimum((q*bins).astype(int),bins-1)
   n+=np.bincount(idx,minlength=bins);sp+=np.bincount(idx,weights=q,minlength=bins);sy+=np.bincount(idx,weights=y,minlength=bins)
 return dict(counts=np.array([[tn,fp],[fn,tp]],dtype=np.int64),n=n,sp=sp,sy=sy,n_targets=len(paths),total=total)

def main():
 pa=argparse.ArgumentParser();pa.add_argument('input_root',type=Path);pa.add_argument('output_dir',type=Path);pa.add_argument('--bins',type=int,default=15)
 args=pa.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
 runs=[compute(args.input_root/m,args.bins) for m in MODELS]
 colors=['#73808D','#D97B28','#247B81']
 fig,axs=plt.subplots(1,3,figsize=(12,4.2),layout='constrained')
 for name,d,ax in zip(MODEL_NAMES,runs,axs):
  c=d['counts'];m=ax.imshow(np.log10(1+c),cmap='Oranges');ax.set_title(f'{name}\n{d["n_targets"]} matched targets')
  ax.set_xticks([0,1],['Pred 0','Pred 1']);ax.set_yticks([0,1],['True 0','True 1']);ax.set_xlabel('Predicted label')
  for i in range(2):
   for j in range(2):ax.text(j,i,f'{c[i,j]:,}',ha='center',va='center',fontsize=12)
 fig.savefig(args.output_dir/'pixel_confusion_three_models.png',dpi=220);plt.close(fig)

 fig,ax=plt.subplots(figsize=(7.5,6.3),layout='constrained')
 ax.plot([0,1],[0,1],ls='--',color='gray',label='Perfect calibration')
 for d,name,color in zip(runs,MODEL_NAMES,colors):
  nz=d['n']>0;conf=d['sp'][nz]/d['n'][nz];freq=d['sy'][nz]/d['n'][nz]
  ax.plot(conf,freq,marker='o',markersize=4,lw=1.8,label=f'{name} (n={d["total"]:,} px)',color=color)
 ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xlabel('Mean predicted foreground probability');ax.set_ylabel('Observed foreground frequency');ax.set_title('Pixel-level reliability (all pixels)');ax.grid(alpha=.2);ax.legend(loc='upper left',fontsize=9)
 fig.savefig(args.output_dir/'reliability_diagram_three_models.png',dpi=220);plt.close(fig)
 for model,d in zip(MODELS,runs):
  np.savez(args.output_dir/f'{model}_counts_and_calibration_bins.npz',confusion=d['counts'],bin_count=d['n'],bin_sum_probability=d['sp'],bin_sum_foreground=d['sy'])
 print('Saved exact plots derived solely from provided prediction arrays.')

if __name__=='__main__':main()
