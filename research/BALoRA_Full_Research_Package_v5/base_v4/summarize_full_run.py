"""Generate tables and figures only from this full run's actual outputs."""
from pathlib import Path
import collections,csv,itertools,json,math,shutil,statistics
import numpy as np
from auditlib.io_budget import atomic_json
from protocol import ROOT,METHODS,assert_training_complete,status_path

LABEL={'frozen':'Frozen SAM2','deterministic':'Deterministic LoRA','balora':'BALoRA no refinement'}
COLOR={'frozen':'#b84c23','deterministic':'#648147','balora':'#17679b'}
KEYS=['dice','iou','precision','recall','f1','hd95','assd','ece','brier','nll']

def mean(values):
    x=[float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(statistics.mean(x)) if x else None

def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def plotting():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    return plt

def save(fig,root,name):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    for ext in ['png','svg','pdf']:fig.savefig(root/(name+'.'+ext),dpi=300,bbox_inches='tight',pad_inches=.15,facecolor='white')
    plotting().close(fig)

def histories(root):
    output={}
    for method in ['deterministic','balora']:
        path=Path(root)/method/'research/training_history.json'
        if not path.is_file():continue
        h=json.loads(path.read_text())
        if isinstance(h,dict):h=[{k:v[i] for k,v in h.items() if isinstance(v,list) and len(v)>i} for i in range(len(h['epoch']))]
        output[method]=h
    return output

def training_figure(root,partial=False):
    root=Path(root);h=histories(root)
    if not any(h.values()):return
    plt=plotting();fig,axes=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
    for method,rows in h.items():
        if not rows:continue
        epochs=[r['epoch']+1 for r in rows]
        axes[0].plot(epochs,[r['train_loss'] for r in rows],'o-',color=COLOR[method],ms=3,label=LABEL[method])
        axes[1].plot(epochs,[r['val_dice'] for r in rows],'o-',color=COLOR[method],ms=3,label=LABEL[method])
        best=max(rows,key=lambda r:r['val_dice'])
        axes[1].scatter(best['epoch']+1,best['val_dice'],s=100,marker='*',c=COLOR[method],edgecolor='black',zorder=5)
    axes[0].set(xlabel='Completed epoch',ylabel='Mean objective penalties differ',title='Training objective')
    axes[1].set(xlabel='Completed epoch',ylabel='Fixed-prompt validation Dice',title='Validation-selected checkpoint')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    axes[0].legend(frameon=False,fontsize=8)
    if partial:fig.suptitle('PARTIAL training histories no complete new comparison')
    save(fig,root/'figures','figure_02_training')

def rollup(records):
    """Slice -> volume-organ -> pair summary. Undefined values are counted."""
    pairs=[];volumes=[];metrics=[];organs=[];regional=[];ranking=[]
    for method,rows in records.items():
        groups=collections.defaultdict(list)
        for row in rows:groups[row['case_id'],row['organ']].append(row)
        for (cid,organ),rr in sorted(groups.items()):
            for metric in KEYS:
                prefix='cal' if metric in ['ece','brier','nll'] else 'seg'
                values=[r[prefix][metric] for r in rr];n=sum(v is not None and math.isfinite(float(v)) for v in values)
                pairs.append({'method':method,'case_id':cid,'organ':organ,'metric':metric,'mean':mean(values),
                    'defined_slices':n,'total_slices':len(rr),'undefined_slices':len(rr)-n})
            if not all('diagnostics' in r for r in rr):continue
            for region in ['all','GT_foreground','GT_boundary_2px']:
                for metric in ['ece','brier','nll']:
                    values=[r['diagnostics']['calibration'][region][metric] for r in rr]
                    regional.append({'method':method,'case_id':cid,'organ':organ,'region':region,'metric':metric,
                        'mean':mean(values),'defined_slices':sum(v is not None for v in values),'total_slices':len(rr)})
            for score in rr[0]['diagnostics']['ranking']:
                for region in ['all','GT_boundary_2px']:
                    for metric in ['auroc_hist','average_precision_hist','point_biserial','normalized_aurc_0p1_to_1']:
                        values=[r['diagnostics']['ranking'][score][region][metric] for r in rr]
                        ranking.append({'method':method,'case_id':cid,'organ':organ,'score':score,'region':region,'metric':metric,
                            'mean':mean(values),'defined_slices':sum(v is not None for v in values),'total_slices':len(rr)})
        for cid in sorted({r['case_id'] for r in rows}):
            for metric in KEYS:
                vv=[r['mean'] for r in pairs if r['method']==method and r['case_id']==cid and r['metric']==metric]
                volumes.append({'method':method,'case_id':cid,'metric':metric,'mean':mean(vv),
                                'defined_organs':sum(v is not None for v in vv),'total_organs':len(vv)})
        for metric in KEYS:
            rr=[r for r in pairs if r['method']==method and r['metric']==metric];values=[r['mean'] for r in rr];valid=[v for v in values if v is not None]
            metrics.append({'method':method,'metric':metric,'mean':mean(values),
                'sd_across_volume_organ_pairs':float(statistics.stdev(valid)) if len(valid)>1 else None,
                'defined_pairs':len(valid),'total_pairs':len(values),'undefined_slice_values':sum(r['undefined_slices'] for r in rr)})
        for organ in sorted({r['organ'] for r in rows}):
            vv=[r['mean'] for r in pairs if r['method']==method and r['organ']==organ and r['metric']=='dice']
            organs.append({'method':method,'organ':organ,'mean_dice':mean(vv),'defined_volumes':sum(v is not None for v in vv)})
    return {'pairs':pairs,'volumes':volumes,'metrics':metrics,'organs':organs,'regional_calibration':regional,'uncertainty_pair_metrics':ranking}

def volume_analysis(volumes):
    cases=sorted({r['case_id'] for r in volumes});n=len(cases)
    arrays={m:np.array([next(r['mean'] for r in volumes if r['method']==m and r['case_id']==cid and r['metric']=='dice') for cid in cases]) for m in METHODS}
    ix=np.asarray(list(itertools.product(range(n),repeat=n))) if n<=6 else np.random.default_rng(4321).integers(0,n,(20000,n))
    contrasts=[]
    for a,b in [('deterministic','balora'),('balora','frozen'),('deterministic','frozen')]:
        d=arrays[a]-arrays[b];lo,hi=np.quantile(d[ix].mean(1),[.025,.975])
        contrasts.append({'comparison':a+'_minus_'+b,'mean_difference_points':float(100*d.mean()),
            'exploratory_lower_points':float(100*lo),'exploratory_upper_points':float(100*hi),
            'independent_volume_units':n,'ordered_resamples':len(ix),'multiplicity_adjusted':False})
    d=arrays['deterministic']-arrays['balora']
    loo=[{'omitted_volume':cid,'remaining_volumes':n-1,'difference_points':float(100*np.delete(d,i).mean())} for i,cid in enumerate(cases)]
    return arrays,contrasts,loo

def summarize(root):
    root=Path(root);assert_training_complete(root)
    audit=json.loads((root/'data_audit.json').read_text());expected=set(audit['expected_test_keys'])
    records={};identities={}
    for method in METHODS:
        s=json.loads(status_path(root,method,'test').read_text())
        if not s.get('complete'):raise ValueError('Incomplete full test '+method)
        rows=[json.loads(p.read_text()) for p in sorted((root/method/'test/records').glob('*.json'))]
        if len(rows)!=2466 or {r['key'] for r in rows}!=expected:raise ValueError('Missing or duplicate full test keys')
        records[method]=rows
        identities[method]=json.loads((root/method/'test/identity.json').read_text())['identity']
        selected=root/method/'research/best.pt'
        if method!='frozen':
            from auditlib.io_budget import digest_file
            if identities[method]['weights_sha256']!=digest_file(selected):raise ValueError('Test weights differ from the validation-selected new checkpoint')
    if not all(identities[m]['data']==identities['frozen']['data'] and identities[m]['runtime']==identities['frozen']['runtime'] for m in METHODS):
        raise ValueError('Do not silently combine test reconstructions from different datasets or runtime environments')
    maps={m:{r['key']:r for r in rows} for m,rows in records.items()}
    for key in expected:
        if not all(maps[m][key]['prompt_xy']==maps['frozen'][key]['prompt_xy'] for m in METHODS):raise ValueError('Unmatched positive point')
    summary=rollup(records);tables=root/'tables'
    for name,rows in summary.items():write_csv(tables/(name+'.csv'),rows)
    arrays,contrasts,loo=volume_analysis(summary['volumes'])
    write_csv(tables/'paired_volume_bootstrap.csv',contrasts);write_csv(tables/'leave_one_volume_out.csv',loo)
    failure_detection(root,records);plots(root,records,summary,arrays,contrasts,loo,audit)
    report={'complete':True,'fresh_research_run':True,'historical_scores_injected':False,
        'coverage':{'volumes':6,'foreground_slices_per_method':460,'predictions_per_method':2466,'dependent_pairs_per_method':78},
        'training_seed_count':1,'seed':json.loads((root/'study_identity.json').read_text())['seed'],'fold_count':1,
        'primary_aggregation':'slice metric mean within volume-organ, then equal pair mean',
        'primary_policy':'none fixed before this new run','distance_units':'resized 2D pixels, not physical 3D millimetres',
        'metrics':summary['metrics'],'volume_contrasts':contrasts,
        'bootstrap_scope':'exploratory, whole volumes jointly resampled, conditional on one seed/split/prompt set; no confirmatory significance claim',
        'figure_sets':len(list((root/'figures').glob('figure_*.png'))),
        'qualitative_panels':len(list((root/'figures/qualitative').glob('*.png'))),
        'uncertainty_caveat':'Histogram AUROC/AP approximations; pixel/organ observations within a volume dependent; no artificial deterministic epistemic MI',
        'prediction_array_precision':'FP16 probabilities/MI for figures; primary metrics and stored threshold masks use FP32'}
    atomic_json(root/'final_research_report.json',report)
    text='# Completed new Fold 1 results\n\nThis report describes the new run only. Previous scores are not copied. One seed and one fold completed; oracle positive-point assistance remains part of the task.\n\n| Method | Mean Dice |\n| --- | ---: |\n'
    for m in METHODS:text+='| '+LABEL[m]+' | '+format(next(r['mean'] for r in summary['metrics'] if r['method']==m and r['metric']=='dice'),'.6f')+' |\n'
    text+='\nSee tables for all denominators, undefined values, conditional probability diagnostics and paired volume effects. Intervals are exploratory and do not cover retraining or external generalization. Refinement findings are reported from this run without assuming the historical negative result recurs.\n'
    (root/'RESULTS.md').write_text(text)
    return report

def failure_detection(root,records):
    from scipy.stats import rankdata
    rows=[]
    for method,rr in records.items():
        for cid in sorted({r['case_id'] for r in rr}):
            vv=[r for r in rr if r['case_id']==cid];bad=np.array([r['dice']<.5 for r in vv]);p=int(bad.sum());n=len(vv)-p
            for score in vv[0]['diagnostics']['ranking']:
                u=np.array([r['diagnostics']['ranking'][score]['all']['mean_uncertainty'] for r in vv]);q=rankdata(u);d=rankdata([1-r['dice'] for r in vv])
                auc=float((q[bad].sum()-p*(p+1)/2)/(p*n)) if p and n else None
                corr=float(np.corrcoef(q,d)[0,1]) if q.std()>0 and d.std()>0 else None
                rows.append({'method':method,'case_id':cid,'score':score,'n_predictions':len(vv),'n_failures':p,
                    'failure_rule':'Dice below 0.5 exploratory','mask_failure_AUROC':auc,'spearman_uncertainty_vs_error':corr,
                    'inference':'within-volume descriptive; no slice-level p value'})
    write_csv(Path(root)/'tables/failure_detection.csv',rows)

def plots(root,records,s,arrays,contrasts,loo,audit):
    plt=plotting();figures=Path(root)/'figures';figures.mkdir(exist_ok=True);captions={}
    for ext in ['png','svg','pdf']:shutil.copy2(ROOT/'assets'/('figure_1_method.'+ext),figures/('figure_01_method.'+ext))
    captions['figure_01_method']='Implemented v1 method schematic, not a patient image. Two full-rank posterior probes precede hard rank selection. Deterministic adaptation uses fixed rank 16 and a single output; frozen SAM2 has no adapters.'
    training_figure(root)
    captions['figure_02_training']='Actual completed epochs and fixed-prompt validation scores of the two new fits; stars mark their own selected checkpoints. Objective penalty terms differ. No seed replication is implied.'
    organs=sorted({r['organ'] for r in s['organs']});cases=sorted(audit['splits']['test']['cases']);lookup={(r['method'],r['organ']):r['mean_dice'] for r in s['organs']}
    fig,ax=plt.subplots(figsize=(9,5.8),layout='constrained')
    for method,mark in zip(METHODS,['s','^','o']):ax.scatter([lookup[method,o] for o in organs],range(13),label=LABEL[method],c=COLOR[method],marker=mark,s=30)
    ax.set(yticks=range(13),yticklabels=[o.replace('_',' ') for o in organs],xlim=(0,1),xlabel='Dice mean of six volume-organ means per organ');ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.legend(ncol=3,loc='lower center',bbox_to_anchor=(.5,1.01),frameon=False,fontsize=8)
    save(fig,figures,'figure_03_organs');captions['figure_03_organs']='Every organ point averages six volume-organ means after slice averaging. Six volumes are independent only under the stated subject assumption.'
    fig,ax=plt.subplots(figsize=(9,3.5),layout='constrained')
    for i,cid in enumerate(cases):ax.plot(range(3),[arrays[m][i] for m in METHODS],'o-',label=cid)
    ax.set(xticks=range(3),xticklabels=[LABEL[m] for m in METHODS],ylabel='Dice mean of 13 organ means');ax.legend(ncol=3,loc='lower center',bbox_to_anchor=(.5,1.01),frameon=False);ax.grid(axis='y',alpha=.2)
    save(fig,figures,'figure_04_volumes');captions['figure_04_volumes']='Each line is one of six paired test volumes; its 13 organ means remain dependent.'
    metric={(r['method'],r['metric']):r for r in s['metrics']}
    for stem,keys,unit in [('figure_05_probability',['ece','brier','nll'],'All-pixel score'),('figure_06_surfaces',['hd95','assd'],'Resized 2D pixels')]:
        fig,axes=plt.subplots(1,len(keys),figsize=(9,3.5),layout='constrained')
        for ax,key in zip(np.atleast_1d(axes),keys):
            vals=[metric[m,key]['mean'] for m in METHODS];bars=ax.bar(range(3),[v or 0 for v in vals],color=[COLOR[m] for m in METHODS])
            ax.set(xticks=range(3),xticklabels=['Frozen','Det. LoRA','BALoRA'],title=key.upper(),ylabel=unit);ax.tick_params(axis='x',rotation=20)
            for bar,value,m in zip(bars,vals,METHODS):
                label='undefined' if value is None else f'{value:.4g}'
                ax.annotate(label,(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
            ax.margins(y=.18);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        save(fig,figures,stem)
        captions[stem]=('Primary right-edge-bin all-pixel probability scores average defined volume-organ means. Background imbalance limits interpretation; not evidence of epistemic usefulness.' if keys[0]=='ece' else 'HD95 and ASSD are resized 2D pixels. Undefined slice/pair counts appear in tables/metrics.csv; finite summaries are conditional on those denominators.')
    ref=json.loads((Path(root)/'validation_refinement_audit/validation_comparison.json').read_text())
    if not ref.get('complete'):raise ValueError('Full new validation refinement audit is incomplete')
    base=ref['reports']['legacy']['same_checkpoint_no_refinement']['per_organ_segmentation']
    fig,ax=plt.subplots(figsize=(9,5.8),layout='constrained')
    for policy,mark,col in [('none','o','#17679b'),('legacy','s','#b84c23'),('guarded','^','#648147')]:
        source=base if policy=='none' else ref['reports'][policy]['per_organ_segmentation']
        ax.scatter([source[o]['dice']['mean'] for o in organs],range(13),marker=mark,c=col,label=policy,s=30)
    ax.set(yticks=range(13),yticklabels=[o.replace('_',' ') for o in organs],xlim=(0,1),xlabel='Validation organ mean Dice');ax.invert_yaxis();ax.legend(frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.01));ax.grid(axis='x',alpha=.2)
    save(fig,figures,'figure_07_validation_refinement');captions['figure_07_validation_refinement']='New validation-only audit on five volumes and 371 foreground slices with the same selected BALoRA weights and paired initial predictions. Results may differ from the historical audit. The main new comparison retains its predeclared no-refinement policy.'
    counts=collections.Counter(rank for r in records['balora'] for rank in r['ranks'].values())
    fig,ax=plt.subplots(figsize=(8,3),layout='constrained');bars=ax.bar([str(r) for r in [4,8,16]],[counts[r] for r in [4,8,16]],color='#17679b')
    for bar,rank in zip(bars,[4,8,16]):ax.annotate(f'{counts[rank]:,} ({100*counts[rank]/sum(counts.values()):.2f}%)',(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
    ax.set(xlabel='Selected prefix rank',ylabel='Dependent layer-level choices');ax.margins(y=.2)
    save(fig,figures,'figure_08_ranks');captions['figure_08_ranks']='Rank choices at the new selected checkpoint across all test prompts. Active prefixes are descriptive and do not establish storage or latency savings.'
    pair={(r['method'],r['case_id'],r['organ']):r['mean'] for r in s['pairs'] if r['metric']=='dice'}
    grids=[np.array([[100*(pair['balora',cid,o]-pair[m,cid,o]) for cid in cases] for o in organs]) for m in ['frozen','deterministic']]
    lim=max(1,math.ceil(max(np.abs(g).max() for g in grids)/10)*10);fig,axes=plt.subplots(1,2,figsize=(9,5.7),layout='constrained')
    for ax,grid,title in zip(axes,grids,['BALoRA minus frozen','BALoRA minus deterministic']):
        im=ax.imshow(grid,vmin=-lim,vmax=lim,cmap='RdBu',aspect='auto');ax.set(xticks=range(6),xticklabels=cases,yticks=range(13),yticklabels=[o.replace('_',' ') for o in organs] if ax is axes[0] else [],title=title);ax.tick_params(axis='x',rotation=45)
    fig.colorbar(im,ax=list(axes),label='Dice difference percentage points',shrink=.75)
    save(fig,figures,'figure_09_pair_differences');captions['figure_09_pair_differences']='All 78 dependent volume-organ differences; positive favors BALoRA. One unclipped symmetric color scale is used. This is not 78 independent patients.'
    d=100*(arrays['deterministic']-arrays['balora']);fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    axes[0].scatter(range(6),d,c='#17679b');axes[1].scatter(range(6),[r['difference_points'] for r in loo],c='#17679b')
    for ax in axes:ax.axhline(d.mean(),ls='--',c='gray');ax.axhline(0,c='black',lw=.7);ax.set(xticks=range(6),xticklabels=cases,ylabel='Deterministic minus BALoRA Dice points');ax.tick_params(axis='x',rotation=45);ax.grid(axis='y',alpha=.2)
    axes[0].set_title('Observed volume effects');axes[1].set_title('Mean after deleting the labeled volume')
    save(fig,figures,'figure_10_sensitivity');captions['figure_10_sensitivity']='Post hoc observed-case influence. Deletion estimates are not confidence intervals, seed replications or external confirmation.'
    risk_reliability_runtime(root,records,s,captions)
    qualitative(root,records,audit)
    atomic_json(Path(root)/'figure_captions.json',captions)
    (Path(root)/'FIGURES.md').write_text('# Figures generated from this run\n\n'+'\n\n'.join('## '+k+'\n\n'+v for k,v in captions.items())+'\n')

def risk_reliability_runtime(root,records,s,captions):
    plt=plotting();figures=Path(root)/'figures'
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained');risk_rows=[]
    for method,rr in records.items():
        for score in rr[0]['diagnostics']['ranking']:
            for ax,region in zip(axes,['all','GT_boundary_2px']):
                curve=[]
                for j in range(10):
                    group=collections.defaultdict(list)
                    for r in rr:group[r['case_id'],r['organ']].append(r['diagnostics']['ranking'][score][region]['risk_curve'][j]['pixel_error_risk'])
                    volumes=collections.defaultdict(list)
                    for (cid,organ),v in group.items():volumes[cid].append(mean(v))
                    value=mean([mean(v) for v in volumes.values()]);curve.append(value)
                    risk_rows.append({'method':method,'score':score,'region':region,'coverage':(j+1)/10,'pixel_error_risk':value,'defined_volumes':sum(mean(v) is not None for v in volumes.values())})
                ax.plot(np.linspace(.1,1,10),curve,c=COLOR[method],ls='--' if score=='epistemic_MI' else '-',label=method+' '+score)
    for ax,region in zip(axes,['All pixels','Ground truth boundary band']):ax.set(xlabel='Retained pixel coverage',ylabel='Pixel error risk',title=region);ax.grid(alpha=.2)
    axes[1].legend(fontsize=6);save(fig,figures,'figure_11_risk_coverage')
    write_csv(Path(root)/'tables/risk_coverage.csv',risk_rows)
    captions['figure_11_risk_coverage']='Exploratory pixel misclassification risk, not Dice, from 256-bin ranking with within-bin interpolation. Slice diagnostics average within volume-organ pairs then volumes. Deterministic entropy and Bayesian MI are distinct comparators; no deterministic MI is manufactured.'
    fig,axes=plt.subplots(1,3,figsize=(10,3.5),layout='constrained');reliability=[]
    for ax,region in zip(axes,['all','GT_foreground','GT_boundary_2px']):
        ax.plot([0,1],[0,1],'--',c='gray',lw=.8)
        for method,rr in records.items():
            by=collections.defaultdict(lambda:[np.zeros(15),np.zeros(15),np.zeros(15)])
            for r in rr:
                cal=r['diagnostics']['calibration'][region];n,p,y=by[r['case_id']]
                n+=np.asarray(cal['bin_count']);p+=np.asarray(cal['bin_sum_p']);y+=np.asarray(cal['bin_sum_y'])
            xs=[];ys=[]
            for j in range(15):
                valid=[v for v in by.values() if v[0][j]>0]
                if not valid:continue
                xp=mean([p[j]/n[j] for n,p,y in valid]);yp=mean([y[j]/n[j] for n,p,y in valid]);xs.append(xp);ys.append(yp)
                reliability.append({'method':method,'region':region,'bin':j,'mean_probability':xp,'foreground_frequency':yp,'defined_volumes':len(valid),'bins':'15 left-inclusive diagnostic bins; each curve point equally averages volumes having that bin'})
            ax.plot(xs,ys,'o-',ms=3,c=COLOR[method],label=LABEL[method])
        ax.set(xlim=(0,1),ylim=(0,1),title=region.replace('GT_','GT '),xlabel='Mean foreground probability',ylabel='Foreground frequency');ax.grid(alpha=.2)
    axes[2].legend(fontsize=6);save(fig,figures,'figure_12_conditional_reliability')
    write_csv(Path(root)/'tables/conditional_reliability.csv',reliability)
    captions['figure_12_conditional_reliability']='Diagnostic reliability curves pool bin counts within each volume, then equally average defined volume-bin means. Region masks use ground truth; foreground-only frequency is necessarily one. These left-inclusive diagnostic bins are separate from the primary metric implementation. Bin-specific volume denominators are reported; no pixel-independence confidence bands are drawn.'
    timings=[]
    for method,rr in records.items():
        enc=sum({(r['case_id'],r['slice_index']):r['encoder_seconds_for_slice'] for r in rr}.values());dec=sum(r['decode_seconds'] for r in rr)
        status=json.loads((Path(root)/method/'test/status.json').read_text())
        timings.append({'method':method,'encoder_ms_per_unique_slice':1000*enc/460,'decoder_ms_per_prompt':1000*dec/2466,
            'peak_reserved_GiB_across_test_sessions':status['peak_reserved_bytes_all_sessions']/2**30,
            'stored_trainable_parameters':status['stored_trainable_parameters'],
            'scope':'Synchronized model calls, one recorded encoder call per unique slice; extra resume encodes, CPU metrics, transfer, setup and export excluded; warm-up retained'})
    write_csv(Path(root)/'tables/matched_inference_measurements.csv',timings)
    fig,axes=plt.subplots(1,3,figsize=(10,3.5),layout='constrained')
    for ax,key,title in zip(axes,['encoder_ms_per_unique_slice','decoder_ms_per_prompt','peak_reserved_GiB_across_test_sessions'],['Encoder ms per slice','Decoder ms per prompt','Peak reserved GPU GiB']):
        ax.bar(range(3),[r[key] for r in timings],color=[COLOR[m] for m in METHODS]);ax.set(xticks=range(3),xticklabels=['Frozen','Det. LoRA','BALoRA'],title=title);ax.tick_params(axis='x',rotation=20);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    save(fig,figures,'figure_13_inference_measurements')
    captions['figure_13_inference_measurements']='New measured inference boundaries only. Encoder is counted once per unique slice and decoder once per prompt; BALoRA includes two probes and four posterior outputs. Warm-up remains included; resume encodes, transfers, diagnostics and export are excluded. Peak reserved memory is not training memory or active-rank parameter savings.'

def qualitative(root,records,audit):
    plt=plotting();destination=Path(root)/'figures/qualitative';destination.mkdir(exist_ok=True)
    maps={m:{r['key']:r for r in rr} for m,rr in records.items()}
    for key in audit['qualitative_keys']:
        arrays={m:np.load(Path(root)/m/'test'/maps[m][key]['array_file'],allow_pickle=False) for m in METHODS}
        b=arrays['balora'];fig,axes=plt.subplots(2,3,figsize=(10,7),layout='constrained')
        views=[(b['windowed_CT'],'Windowed CT','gray'),(b['ground_truth'],'Ground truth','gray'),(b['epistemic_MI_FP16'],'BALoRA epistemic MI nats','magma')]
        views.extend((arrays[m]['prediction_FP32_threshold'],LABEL[m]+' Dice '+format(maps[m][key]['dice'],'.3f'),'gray') for m in METHODS)
        for ax,(data,title,cmap) in zip(axes.ravel(),views):
            im=ax.imshow(data,cmap=cmap,vmin=0,vmax=np.log(2) if cmap=='magma' else 1);ax.set_title(title,fontsize=9);ax.axis('off')
            if cmap=='magma':fig.colorbar(im,ax=ax,fraction=.046,pad=.03,label='MI nats')
        point=maps['balora'][key]['prompt_xy']
        for ax in [axes[0,0],axes[0,1]]:ax.scatter([point[0]],[point[1]],marker='+',c='#00cc44',s=40)
        fig.suptitle(key+' | predeclared anatomy-based example selection',fontsize=10)
        stem=key.replace(':','_');fig.savefig(destination/(stem+'.png'),dpi=250,bbox_inches='tight');fig.savefig(destination/(stem+'.pdf'),bbox_inches='tight');plt.close(fig)
        for value in arrays.values():value.close()
    (destination/'CAPTION.md').write_text('Examples are the predeclared maximum-area ground-truth liver, pancreas and right adrenal slices per test volume, selected before fitting. They are genuine outputs and do not represent an independently sampled accuracy cohort. Green crosses mark the same oracle positive point. Maps store FP16 probability/MI for display; masks and reported metrics use the original FP32 threshold. Interpret successes and failures using the displayed Dice values; no synthetic masks are used.\n')
