"""Create paper figures from archived validation histories and saved CT arrays."""
from pathlib import Path
from zipfile import ZipFile
import argparse
import io
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

BLUE = '#145b8a'
TEAL = '#08816e'
INK = '#243240'
LIGHT = '#d7dce1'
plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':8.6,'axes.labelsize':8.6,
    'axes.titlesize':9.3,'axes.titleweight':'bold',
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
    'pdf.fonttype':42,'figure.dpi':200,'savefig.dpi':340,
})


def plot_method(figures):
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.99)
    ax.set(xlim=(0,1), ylim=(0,1))
    ax.axis('off')

    def box(x, y, heading, details, color='#eef4f7'):
        width, height = .446, .179
        ax.add_patch(FancyBboxPatch((x,y),width,height,boxstyle='round,pad=.008,rounding_size=.012',
                                   fc=color,ec='#6e8191',lw=.8))
        ax.text(x+width/2,y+.123,heading,ha='center',va='center',color=INK,weight='bold',size=9)
        ax.text(x+width/2,y+.064,details,ha='center',va='center',color=INK,size=8.2,linespacing=1.22)

    def arrow(x, y, x_end, y_end):
        ax.annotate('',(x_end,y_end),(x,y),
                    arrowprops={'arrowstyle':'-|>','color':'#465e70','lw':1,'shrinkA':0,'shrinkB':0})

    left, right = .012, .54
    box(left,.79,'CT slice','HU window and 1024 × 1024 resize\nGrayscale repeated in three channels')
    box(right,.79,'SAM2.1 image encoder','Frozen backbone and 96 deterministic adapters\nFixed rank 16')
    box(left,.54,'Positive point','Uniform sample from the reference organ mask\nSame point at both decoder stages','#fcf1e8')
    box(right,.54,'Gaussian decoder probes','Two rank-16 draws\nMean pixel mutual information')
    box(right,.29,'Rank selection','Detached activation × gradient attribution\n32 layers select from ranks {4, 8, 16}')
    box(left,.29,'Gaussian decoder at selected ranks','Fresh factor draws\nTwo during training and four at inference')
    box(left,.04,'Final prediction','Mean foreground probability and 0.5 threshold\nNo refinement','#e8f4ee')
    arrow(.469,.882,.528,.882)
    arrow(.763,.778,.763,.730)
    arrow(.469,.630,.528,.630)
    arrow(.763,.528,.763,.480)
    arrow(.235,.528,.235,.480)
    arrow(.529,.380,.469,.380)
    arrow(.235,.278,.235,.230)
    ax.text(.54,.131,'The gate receives a sparsity gradient.\nThe image embedding is reused across draws.',
            color=INK,va='center',size=8.2,linespacing=1.35)
    fig.savefig(figures/'gaussian_decoder_schematic.pdf')
    plt.close(fig)


def plot_failure_increments(data, figures):
    scores=pd.read_csv(data/'failure_auroc_by_volume.csv')
    intervals=json.loads((data/'analysis_summary.json').read_text())['within_volume_intervals']
    fig, axes=plt.subplots(1,2,figsize=(7.15,2.5),sharey=True,sharex=True)
    y=np.arange(len(scores))
    for ax,title,col,interval in [
        (axes[0],'A  Add MI to organ and area','paired_added_mi','added_mi_t_interval'),
        (axes[1],'B  Add MI to organ, area, and entropy','paired_added_mi_given_entropy',
         'added_mi_given_entropy_t_interval'),
    ]:
        mean=scores[col].mean()
        lo,hi=intervals[interval]
        ax.axvline(0,color='#8f9aa6',lw=.85,ls='--')
        ax.scatter(scores[col],y,color=BLUE,s=24,zorder=3)
        ax.errorbar(mean,len(y)+.30,xerr=[[mean-lo],[hi-mean]],color=TEAL,fmt='D',
                    markersize=4.1,lw=1.25,capsize=3)
        ax.set(title=title,xlabel='Within-volume AUROC change',xlim=(-.029,.079),
               xticks=[-.02,0,.02,.04,.06],yticks=[*y,len(y)+.30],
               yticklabels=[*scores.volume,'Mean (95% t CI)'])
        ax.grid(axis='x',color=LIGHT,alpha=.5)
    axes[0].invert_yaxis()
    fig.tight_layout(w_pad=2)
    fig.savefig(figures/'failure_ranking_increments.pdf',bbox_inches='tight')
    plt.close(fig)


def plot_checkpoints(data, figures):
    h=json.loads((data/'validation_histories.json').read_text())
    d=pd.read_csv(data/'paired_dice_by_volume.csv')
    s=json.loads((data/'analysis_summary.json').read_text())
    fig, (a,b) = plt.subplots(1,2,figsize=(7.15,2.52),gridspec_kw={'width_ratios':[1.14,1]})
    for method,name,c in [('deterministic','Deterministic LoRA',BLUE),('balora','Gaussian decoder LoRA',TEAL)]:
        a.plot(h[method]['epoch'],h[method]['val_dice'],color=c,lw=1.35,marker='o',ms=3,label=name)
        a.scatter([7],[h[method]['val_dice'][6]],s=40,facecolor='white',edgecolor=c,zorder=4)
    a.axvline(7,color='#8f9aa6',lw=.75,ls='--')
    a.set(title='A  Validation Dice by epoch',xlabel='Completed epoch',ylabel='Volume–organ mean Dice')
    a.set_xticks([1,4,7,10,13])
    a.legend(frameon=False,fontsize=7.2,loc='lower right')
    a.grid(axis='y',alpha=.28,zorder=0)
    y=np.arange(len(d))
    b.axvline(0,color='#8f9aa6',lw=.8,ls='--')
    b.scatter(d.det_minus_gaussian_pp,y,color=BLUE,s=25,zorder=3)
    c=s['paired_dice']
    lo,hi=c['t_interval_pp']
    b.errorbar(c['mean_pp'],len(d)+.35,xerr=[[c['mean_pp']-lo],[hi-c['mean_pp']]],
               fmt='D',color=TEAL,capsize=3,markersize=4,lw=1.2)
    b.set(yticks=np.r_[y,len(d)+.35],yticklabels=list(d.volume)+['Mean (95% t CI)'],
          xlabel='Deterministic − Gaussian Dice (points)',title='B  Paired test-volume contrasts')
    b.invert_yaxis()
    b.grid(axis='x',alpha=.25,zorder=0)
    fig.tight_layout(w_pad=2)
    fig.savefig(figures/'verified_validation_and_paired_dice.pdf',bbox_inches='tight')
    plt.close(fig)


def plot_ct(data, figures, archive):
    audit=json.loads((data/'archive_audit.json').read_text())
    df=pd.read_csv(data/'per_prompt_fold1_seed42.csv')
    pool=audit['qualitative_targets']
    best=max(pool,key=lambda v:v['dice'])
    worst=min(pool,key=lambda v:v['dice'])
    assert best['dice']>.95 and worst['dice']<.5
    chosen=[]
    fig,axs=plt.subplots(2,4,figsize=(7.12,3.37))
    titles=['Reference and point','Deterministic LoRA','Gaussian decoder LoRA','Decoder MI']
    with ZipFile(archive) as z:
        root=z.namelist()[0].split('/')[0]+'/'
        for row, target in enumerate((best,worst)):
            key=target['key']
            stem=key.replace(':','_')
            m={}
            for method in ('deterministic','balora'):
                name=root+f'{method}/test/arrays/{stem}.npz'
                with np.load(io.BytesIO(z.read(name)),allow_pickle=False) as a:
                    m[method]={k:a[k] for k in a.files if k!='identity_json'}
            rec=json.loads(z.read(root+f'balora/test/records/{stem}.json'))
            x,y=rec['prompt_xy']
            g=m['balora']['ground_truth'].astype(bool)
            assert g[y,x]
            det=m['deterministic']['prediction_FP32_threshold'].astype(bool)
            bayes=m['balora']['prediction_FP32_threshold'].astype(bool)
            v,u=np.where(g|det|bayes)
            center=((u.min()+u.max())/2,(v.min()+v.max())/2)
            radius=max(96,(max(u.max()-u.min(),v.max()-v.min())+50)/2)
            def bounds(c):
                lo=max(0,int(c-radius))
                hi=min(1024,int(c+radius))
                return lo,hi
            x0,x1=bounds(center[0]); y0,y1=bounds(center[1])
            ct=m['balora']['windowed_CT']
            for col,ax in enumerate(axs[row]):
                ax.imshow(ct,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
                ax.contour(g,levels=[.5],colors=['#63e7b8'],linewidths=.8)
                if col==0:
                    ax.plot(x,y,'+',color='#ffdc65',ms=6,mew=1.3)
                elif col in (1,2):
                    ax.contour([det,bayes][col-1],levels=[.5],colors=['#f4aa48'],linewidths=1.05)
                else:
                    v=m['balora']['epistemic_MI_FP16'].astype(float)
                    ax.imshow(np.ma.masked_where(v<.005,v),cmap='magma',vmin=0,vmax=.15,
                              alpha=.9,interpolation='nearest')
                ax.set_xlim(x0,x1); ax.set_ylim(y1,y0); ax.axis('off')
                if row==0:
                    ax.set_title(titles[col],fontsize=8.3,pad=4)
            name=rec['organ'].replace('_',' ').capitalize()
            score_det=float(df[(df.method=='deterministic')&(df.key==key)].dice.iloc[0])
            vol=rec['case_id']
            axs[row,0].text(.04,.04,f"{vol} · {name}\nSlice {rec['slice_index']}",fontsize=6.8,
                            color='white',ha='left',va='bottom',transform=axs[row,0].transAxes,
                            bbox={'facecolor':'black','alpha':.75,'edgecolor':'none','pad':2})
            for col,val in [(1,score_det),(2,target['dice'])]:
                axs[row,col].text(.04,.04,f'Dice {val:.3f}',fontsize=6.8,
                                  color='white',transform=axs[row,col].transAxes,
                                  bbox={'facecolor':'black','alpha':.75,'edgecolor':'none','pad':2})
            b=df[(df.method=='balora')&(df.volume==vol)].mi
            chosen.append({'key':key,'organ':name,'deterministic_dice':score_det,
                           'gaussian_dice':target['dice'],'mean_mi_nats':target['mi'],
                           'within_volume_mi_percentile':float((b<=target['mi']).mean())})
    fig.subplots_adjust(left=.006,right=.994,bottom=.02,top=.925,wspace=.032,hspace=.075)
    fig.savefig(figures/'verified_ct_examples.png',bbox_inches='tight',pad_inches=.015)
    plt.close(fig)
    (data/'main_qualitative_examples.json').write_text(json.dumps(chosen,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('archive',type=Path)
    p.add_argument('source',type=Path)
    args=p.parse_args()
    data=args.source/'analysis'/'data'
    figs=args.source/'figures'
    figs.mkdir(parents=True, exist_ok=True)
    plot_method(figs)
    plot_checkpoints(data,figs)
    plot_failure_increments(data,figs)
    plot_ct(data,figs,args.archive)


if __name__=='__main__':
    main()
