"""Figures and tables from complete measurements only. No synthetic result curves."""
import csv,json,math
from pathlib import Path
import numpy as np
from v5.common import read,atomic,programme

def pair_values(rows,metric='dice'):
    groups={}
    for r in rows:
        x=r.get('dice') if metric=='dice' else r.get('seg',{}).get(metric,r.get('cal',{}).get(metric))
        if x is not None and math.isfinite(float(x)):groups.setdefault((r['case_id'],r['organ_class']),[]).append(x)
    return {k:float(np.mean(v)) for k,v in groups.items()}

def volume_values(rows):
    groups={}
    for (c,o),v in pair_values(rows).items():groups.setdefault(c,[]).append(v)
    return {c:float(np.mean(v)) for c,v in groups.items()}

def paired(a,b):
    if a and b and all('key' in r and 'prompt_xy' in r for r in a+b):
        if {r['key']:r['prompt_xy'] for r in a}!={r['key']:r['prompt_xy'] for r in b}:raise ValueError('Paired targets/prompt coordinates differ')
    pa,pb=pair_values(a),pair_values(b)
    if set(pa)!=set(pb):raise ValueError('Paired volume-organ sets differ')
    cases=sorted({c for c,o in pa});sums=[];counts=[]
    for c in cases:
        values=[pa[k]-pb[k] for k in pa if k[0]==c];sums.append(sum(values));counts.append(len(values))
    sums=np.asarray(sums);counts=np.asarray(counts);ds=sums/counts
    rng=np.random.default_rng(42019);ix=rng.integers(0,len(ds),(20000,len(ds)))
    est=sums[ix].sum(1)/counts[ix].sum(1)
    return {'mean_difference':float(sums.sum()/counts.sum()),'mean_volume_macro_difference':float(ds.mean()),'volume_differences':dict(zip(cases,ds.tolist())),
        'aggregation':'primary equal volume-organ pairs; bootstrap whole volumes retaining all organ pairs and their counts','organ_pairs_per_volume':dict(zip(cases,counts.tolist())),
        'exploratory_percentile_interval':np.quantile(est,[.025,.975]).tolist(),'independent_units':len(ds),
        'bootstrap_draws':20000,'bootstrap_rng_seed':42019,
        'qualification':'Exploratory volume-cluster interval; seed/split/prompt uncertainty excluded. Only six independent test volumes per fold. No confirmatory significance or slice/pair independence.'}

def binary_auc(score,labels):
    score=np.asarray(score,float);labels=np.asarray(labels,bool);P=labels.sum();N=len(labels)-P
    if not P or not N:return None
    order=np.argsort(score,kind='stable');ranks=np.empty(len(score),float);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and score[order[j]]==score[order[i]]:j+=1
        ranks[order[i:j]]=(i+j+1)/2;i=j
    return float((ranks[labels].sum()-P*(P+1)/2)/(P*N))

def table(path,rows):
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def report(output):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    output=Path(output);dest=output/'analysis';dest.mkdir(exist_ok=True);figures=dest/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none','pdf.fonttype':42})
    def save(fig,name,caption):
        fig.tight_layout()
        for ext in ('png','pdf','svg'):fig.savefig(figures/(name+'.'+ext),dpi=240,bbox_inches='tight')
        plt.close(fig);(figures/(name+'_caption.txt')).write_text(caption)
    register=[];runs={};configs={}
    for j in programme():
        home=output/'jobs'/j['id'];tr=read(home/'research/training_status.json',{});te=read(home/'test/status.json',{})
        complete=bool(tr.get('complete') and te.get('complete') and not te.get('bounded_profile'))
        register.append({**j,'status':'COMPLETE' if complete else '[PENDING EXPERIMENT]','completed_epochs':tr.get('epoch'),'selected_epoch':tr.get('best_epoch'),'checkpoint_sha256':tr.get('selected_checkpoint_sha256'),'test_records':te.get('records')})
        if complete:
            runs[j['id']]=[read(p) for p in sorted((home/'test/records').glob('*.json'))];configs[j['id']]=j
    table(dest/'experiment_register.csv',register)
    # Add verified historical current comparison without confusing a rerun with another seed.
    ref=read(output/'reference_audit.json',{}).get('source_root')
    if ref:
        for m in ('frozen','deterministic','balora'):
            p=Path(ref)/m/'test/records'
            if p.is_dir():
                name={'frozen':'frozen_ref','deterministic':'det_v1_f1_s42','balora':'balora_v1_f1_s42'}[m]
                runs[name]=[read(x) for x in sorted(p.glob('*.json'))];configs[name]={'variant':m+'_ref','seed':42,'fold':1}
    ensemble_home=output/'jobs/det_ensemble_f1_s42_123_2026'
    if read(ensemble_home/'ensemble_status.json',{}).get('complete'):
        runs[ensemble_home.name]=[read(p) for p in (ensemble_home/'test/records').glob('*.json')]
    metrics=[];organs=[]
    for name,rows in runs.items():
        for metric in ('dice','iou','hd95','assd','ece','brier','nll'):
            pairs=pair_values(rows,metric)
            metrics.append({'run':name,'metric':metric,'mean_over_defined_pairs':float(np.mean(list(pairs.values()))) if pairs else None,'defined_pairs':len(pairs),'total_pairs':len({(r['case_id'],r['organ_class']) for r in rows}),'undefined_predictions':sum((r.get('seg',{}).get(metric,r.get('cal',{}).get(metric))) is None for r in rows) if metric!='dice' else 0,'units':'resized 2D pixels' if metric in ('hd95','assd') else 'unitless'})
        groups={}
        for (c,o),v in pair_values(rows).items():groups.setdefault(o,[]).append(v)
        for o,v in groups.items():organs.append({'run':name,'organ':o,'mean_dice':float(np.mean(v)),'volume_pairs':len(v)})
    table(dest/'metrics.csv',metrics);table(dest/'per_organ.csv',organs)
    # Mechanism schematic explicitly has no experimental measurements.
    fig,ax=plt.subplots(figsize=(9,4));ax.axis('off')
    boxes=[(.01,.62,.24,.22,'Windowed CT\n1024 x 1024'),(.34,.62,.26,.22,'SAM2.1 Hiera Base+\nDeterministic encoder LoRA'),(.71,.62,.27,.22,'Decoder adapter\npoint estimate or Gaussian'),(.05,.08,.39,.25,'v1: MI gradient x activation\ndetached score -> integer rank\ngate learns penalty prior only'),(.55,.08,.4,.25,'v2: entropy attribution\nsoftmax + hard forward rank\nstraight-through task gradient')]
    for x,y,w,h,label in boxes:ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.01',facecolor='#edf2f6',edgecolor='#36556d'));ax.text(x+w/2,y+h/2,label,ha='center',va='center')
    for a,b in [((.25,.73),(.34,.73)),((.6,.73),(.71,.73)),((.83,.62),(.25,.33)),((.83,.62),(.75,.33))]:ax.annotate('',b,a,arrowprops={'arrowstyle':'->'})
    ax.text(.27,.48,'Oracle positive point from GT\nFrozen prompt encoder',ha='center',va='center',fontsize=8)
    ax.annotate('',(.71,.65),(.44,.49),arrowprops={'arrowstyle':'->','connectionstyle':'arc3,rad=-.12'})
    save(fig,'01_method_schematic','Schematic of the preserved v1 and separately trained v2. Branches are distinct implementations; v2 is not a correction applied to historical results. Oracle ground-truth point assistance is required. No measured result is depicted.')
    if runs:
        fig,ax=plt.subplots(figsize=(9,max(3,len(runs)*.28)));names=list(runs);means=[np.mean(list(pair_values(runs[n]).values())) for n in names]
        ax.barh(names,means,color='#287da1');ax.set(xlabel='Dice: equal mean over volume–organ slice means',xlim=(0,1));save(fig,'02_completed_comparisons','Only complete selected-checkpoint evaluations are shown. Each test fold has six independent volumes; organ pairs remain dependent. Reference frozen/adapted fits are labeled separately.')
        matrix=np.full((len(names),13),np.nan)
        for i,n in enumerate(names):
            for r in organs:
                if r['run']==n:matrix[i,int(r['organ'])-1]=r['mean_dice']
        fig,ax=plt.subplots(figsize=(10,max(3,len(names)*.3)));im=ax.imshow(matrix,vmin=0,vmax=1,aspect='auto',cmap='viridis');ax.set_yticks(range(len(names)),names);ax.set_xticks(range(13),range(1,14));ax.set_xlabel('BTCV organ label (mapping in source)');fig.colorbar(im,ax=ax,label='Mean Dice');save(fig,'03_per_organ','Slice means within each volume–organ pair, then means over available volume pairs per organ. Missing values are blank; no invented values.')
    comparisons=[]
    for seed in (42,123,2026):
        a=f'det_v1_f1_s{seed}';b=f'balora_v1_f1_s{seed}'
        if a in runs and b in runs:comparisons.append({'seed':seed,'fold':1,'contrast':'det_v1 minus balora_v1',**paired(runs[a],runs[b])})
    for fold in range(2,6):
        a=f'det_v1_f{fold}_s42';b=f'balora_v1_f{fold}_s42'
        if a in runs and b in runs:comparisons.append({'seed':42,'fold':fold,'contrast':'det_v1 minus balora_v1',**paired(runs[a],runs[b])})
    atomic(dest/'paired_volume_effects.json',comparisons)
    if comparisons:
        fig,ax=plt.subplots(figsize=(8,3.5))
        for i,c in enumerate(comparisons):
            d=np.array(list(c['volume_differences'].values()))*100;ax.scatter(np.full(len(d),i),d,alpha=.65);ax.plot([i-.2,i+.2],[d.mean()]*2,color='black')
        ax.axhline(0,color='gray',ls='--');ax.set_xticks(range(len(comparisons)),[f"F{c['fold']} S{c['seed']}" for c in comparisons]);ax.set_ylabel('Deterministic − BALoRA Dice (pp)');save(fig,'04_paired_volume_effects','Dots are volume-level paired effects. The same patients recur across training seeds; these dots must not be pooled as independent patients. Fold test partitions are disjoint but training sets overlap. Black lines indicate mean paired volume effects, without confirmatory p-values.')
    histories=[]
    for j in programme():
        h=read(output/'jobs'/j['id']/'research/training_status.json',{}).get('history',[])
        if h:histories.append((j['id'],h))
    if histories:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for n,h in histories:
            xs=[r['epoch_one_based'] for r in h];axes[0].plot(xs,[r['train_loss'] for r in h],label=n);axes[1].plot(xs,[r['val_dice'] for r in h],label=n)
        axes[0].set(xlabel='Completed epoch',ylabel='Training objective (different penalties)');axes[1].set(xlabel='Completed epoch',ylabel='Validation Dice');axes[1].legend(fontsize=5,bbox_to_anchor=(1.02,1));save(fig,'05_actual_learning_curves','Observed completed epochs only, including explicitly partial fits. Training objectives differ across variants and cannot establish superiority by loss magnitude. No interpolation of unobserved epochs.')
    factorial=[f'{v}_f1_s42' for v in ('det_fixed_v2','bayes_fixed_v1','det_adaptive_v2','bayes_adaptive_v2')]
    if all(n in runs for n in factorial):
        contrasts=[('Bayes − det at fixed rank',factorial[1],factorial[0]),('Adaptive − fixed deterministic',factorial[2],factorial[0]),('Adaptive − fixed Bayesian',factorial[3],factorial[1])]
        effects=[{'contrast':n,**paired(runs[a],runs[b])} for n,a,b in contrasts]
        fixed_b={r['key']:r for r in runs[factorial[1]]};fixed_d={r['key']:r for r in runs[factorial[0]]}
        delta_b=[{**r,'dice':r['dice']-fixed_b[r['key']]['dice']} for r in runs[factorial[3]]]
        delta_d=[{**r,'dice':r['dice']-fixed_d[r['key']]['dice']} for r in runs[factorial[2]]]
        effects.append({'contrast':'Interaction: Bayesian adaptive-minus-fixed minus deterministic adaptive-minus-fixed',**paired(delta_b,delta_d)})
        atomic(dest/'v2_factorial_exploratory.json',effects)
        fig,ax=plt.subplots(figsize=(6,4))
        for names,label in [(factorial[:2],'Fixed rank 16'),(factorial[2:],'Adaptive rank v2')]:ax.plot([0,1],[np.mean(list(pair_values(runs[n]).values())) for n in names],'o-',label=label)
        ax.set_xticks([0,1],['Deterministic','Gaussian Bayesian']);ax.set_ylabel('Pair-aggregated test Dice');ax.legend();save(fig,'06_v2_factorial','Fresh fits with a shared v2 allocator definition in adaptive arms; one training seed. The fresh bayes_fixed_v1 control is the identical fixed Bayesian arm and is reused, not counted as another replicate. Effects are exploratory; historical v1 results remain separate. Bayesian loss includes KL; adaptive loss includes expected normalized rank.')
    diag=[]
    for name,rows in runs.items():
        for r in rows:
            for score,regions in r.get('diagnostics',{}).get('ranking',{}).items():
                for region,d in regions.items():
                    diag.append({'run':name,'case_id':r['case_id'],'organ':r['organ_class'],'key':r['key'],'score':score,'region':region,'auroc_hist':d.get('auroc_hist'),'aurc_hist':d.get('normalized_aurc_0p1_to_1'),'dice_failure_below_0p5':r['dice']<.5,'mean_uncertainty':d.get('mean_uncertainty')})
    table(dest/'uncertainty_per_target.csv',diag)
    failure=[];calibration=[];pixel_curves=[]
    for name,rows in runs.items():
        for score in ('predictive_entropy','epistemic_MI','between_seed_MI'):
            for cid in sorted({r['case_id'] for r in rows}):
                rr=[r for r in diag if r['run']==name and r['case_id']==cid and r['score']==score and r['region']=='all' and r['mean_uncertainty'] is not None]
                if rr:failure.append({'run':name,'case_id':cid,'score':score,'mask_failure_definition':'Dice < 0.5 for this method','targets':len(rr),'failures':sum(r['dice_failure_below_0p5'] for r in rr),'within_volume_auroc':binary_auc([r['mean_uncertainty'] for r in rr],[r['dice_failure_below_0p5'] for r in rr])})
        for region in ('all','GT_foreground','GT_boundary_2px'):
            for metric in ('ece','brier','nll'):
                groups={}
                for r in rows:
                    d=r.get('diagnostics',{}).get('calibration',{}).get(region,{})
                    if d.get(metric) is not None:groups.setdefault((r['case_id'],r['organ_class']),[]).append(d[metric])
                if groups:calibration.append({'run':name,'region':region,'metric':metric,'mean_pairs':float(np.mean([np.mean(x) for x in groups.values()])),'defined_pairs':len(groups),'note':'GT-conditioned foreground/boundary diagnostics; all-pixel background imbalance remains important'})
        for score in ('predictive_entropy','epistemic_MI','between_seed_MI'):
            groups={}
            for r in rows:
                curve=r.get('diagnostics',{}).get('ranking',{}).get(score,{}).get('all',{}).get('risk_curve',[])
                if curve:groups.setdefault((r['case_id'],r['organ_class']),[]).append([x['pixel_error_risk'] for x in curve])
            volumes={}
            for (c,o),values in groups.items():volumes.setdefault(c,[]).append(np.mean(values,axis=0))
            if volumes:pixel_curves.append((name+' '+score,np.mean([np.mean(x,axis=0) for x in volumes.values()],axis=0)))
    table(dest/'mask_failure_detection_by_volume.csv',failure);table(dest/'regional_calibration.csv',calibration)
    if failure:
        labels=[];aucs=[]
        for name in runs:
            for score in ('predictive_entropy','epistemic_MI','between_seed_MI'):
                rr=[r['within_volume_auroc'] for r in failure if r['run']==name and r['score']==score and r['within_volume_auroc'] is not None]
                if rr:labels.append(name+' '+score);aucs.append(np.mean(rr))
        fig,ax=plt.subplots(figsize=(9,max(3,len(labels)*.25)));ax.barh(labels,aucs);ax.set(xlabel='Mask failure AUROC; mean of defined volumes',xlim=(0,1));save(fig,'10_mask_failure_detection','Outcome: each method’s target Dice below 0.5. Undefined volumes with no failures or no successes are excluded and denominators reported. Failure outcomes differ across models; this comparison is exploratory and is not a calibrated clinical failure probability.')
    if pixel_curves:
        fig,ax=plt.subplots(figsize=(8,4))
        for name,curve in pixel_curves:ax.plot(np.linspace(.1,1,10),curve,label=name)
        ax.set(xlabel='Retained pixel coverage',ylabel='Pixel error risk');ax.legend(fontsize=5,bbox_to_anchor=(1.02,1));save(fig,'11_selective_prediction','Low uncertainty pixels retained first. Histogram ties use the documented linear within-bin interpolation. Curves averaged slice → volume-organ → volume → volumes. Background dominates all-pixel risk. No independent-pixel confidence intervals.')
    if calibration:
        fig,ax=plt.subplots(figsize=(9,4))
        for region in ('all','GT_foreground','GT_boundary_2px'):
            rr=[r for r in calibration if r['region']==region and r['metric']=='brier'];ax.plot([r['run'] for r in rr],[r['mean_pairs'] for r in rr],'o-',label=region)
        ax.set_ylabel('Brier score');ax.tick_params(axis='x',rotation=90);ax.legend();save(fig,'12_regional_calibration','All-pixel, GT-foreground and two-pixel GT-boundary Brier scores. Foreground-conditioned scores are diagnostics of false-negative confidence, not complete probability calibration. Scores do not alone validate epistemic uncertainty or clinical reliability.')
    if diag:
        # Average within volume-organ then within volume, finally equal across six volumes.
        fig,ax=plt.subplots(figsize=(9,4));labels=[];values=[]
        for name in runs:
            for score in ('predictive_entropy','epistemic_MI','between_seed_MI'):
                rr=[r for r in diag if r['run']==name and r['score']==score and r['region']=='all' and r['auroc_hist'] is not None]
                g={}
                for r in rr:g.setdefault((r['case_id'],r['organ']),[]).append(r['auroc_hist'])
                v={}
                for (c,o),a in g.items():v.setdefault(c,[]).append(np.mean(a))
                if v:labels.append(name+' '+score);values.append(np.mean([np.mean(a) for a in v.values()]))
        ax.barh(labels,values);ax.set(xlabel='Pixel error AUROC, grouped histogram estimate',xlim=(0,1));save(fig,'07_uncertainty_error','Pixel AUROC excludes all-correct/all-error targets with undefined AUROC. Means respect volume-organ nesting. Spatial pixels are not independent observations; no pixel-level confidence intervals. Epistemic MI exists only for genuine posterior draws; entropy is available for deterministic outputs.')
    sens=[]
    for p in (output/'jobs').glob('*/sensitivity/*/status.json'):
        x=read(p)
        if x.get('complete'):sens.append({'run':p.parents[2].name,'setting':p.parent.name,'validation_dice':x['mean_dice'],'records':x['records']})
    table(dest/'validation_sensitivity.csv',sens)
    stability=[]
    for p in (output/'jobs').glob('*/sensitivity/*/status.json'):
        if not read(p).get('complete'):continue
        home=p.parents[2];epoch=read(home/'research/training_status.json',{}).get('best_epoch')
        if epoch is None:continue
        baseline={r['key']:r for r in [read(x) for x in (home/'validation'/f'epoch_{epoch-1:02d}'/'records').glob('*.json')]}
        rr=[read(x) for x in (p.parent/'records').glob('*.json')];agree=[]
        for r in rr:
            ref=baseline.get(r['key'])
            if ref is None:continue
            if r['prompt_xy']!=ref['prompt_xy'] and not p.parent.name.startswith('prompt_draw'):raise ValueError('Sensitivity changed matched prompt unexpectedly')
            for n,rank in r.get('ranks',{}).items():
                if n in ref.get('ranks',{}):agree.append(rank==ref['ranks'][n])
        stability.append({'run':home.name,'setting':p.parent.name,'layer_target_rank_agreement':float(np.mean(agree)) if agree else None,'defined_layer_target_comparisons':len(agree),'note':'Detached attribution probes and subsequent posterior RNG draws are coupled; probe-count effects are not a pure isolated allocator intervention'})
    table(dest/'rank_selection_stability.csv',stability)
    if sens:
        fig,ax=plt.subplots(figsize=(10,max(3,len(sens)*.18)));ax.barh([r['run']+' '+r['setting'] for r in sens],[r['validation_dice'] for r in sens]);ax.set(xlabel='Validation Dice; exploratory sensitivity',xlim=(0,1));save(fig,'08_validation_sensitivity','K, probe count, fixed-rank probes and prompt placement sensitivity use validation only, full coverage. No primary test policy is selected using these analyses.')
    timings=[]
    for name,rows in runs.items():
        dec=[r.get('decode_seconds') for r in rows if r.get('decode_seconds') is not None];enc={}
        for r in rows:
            x=r.get('encoder_seconds',r.get('encoder_seconds_for_slice'))
            if x is not None:enc[(r['case_id'],r['slice_index'])]=x
        if dec:timings.append({'run':name,'median_decoder_ms':np.median(dec)*1000,'median_encoder_ms_per_slice':np.median(list(enc.values()))*1000 if enc else None,'timing_scope':'CUDA synchronized; encoder once/slice and decoder per prompt incl probes; excludes CPU metrics/IO'})
    table(dest/'matched_timing.csv',timings)
    memory=[]
    for p in (output/'jobs').glob('*/profile.json'):
        r=read(p)
        if r.get('complete'):memory.append({'run':p.parent.name,'bounded_disposable_training_peak_GiB':r['peak_reserved_training_bytes']/2**30,'gpu':r['gpu'],'full_training_peak':'[NOT REPORTED]'})
    table(dest/'training_memory_profiles.csv',memory)
    if memory:
        fig,ax=plt.subplots(figsize=(8,max(3,len(memory)*.3)));ax.barh([r['run'] for r in memory],[r['bounded_disposable_training_peak_GiB'] for r in memory]);ax.set_xlabel('Peak reserved training GiB in disposable representative profile');save(fig,'13_bounded_training_memory','Real bounded training profiles only. These are not the peak over all research epochs; per-variant measured GPU is recorded. Encoder checkpointing is common per-block in new fits. Allocator memory and tensor allocation are distinct quantities.')
    if timings:
        fig,ax=plt.subplots(figsize=(9,max(3,len(timings)*.25)));ax.barh([x['run'] for x in timings],[x['median_decoder_ms'] for x in timings]);ax.set_xlabel('Median synchronized decoder ms/prompt, including probes');save(fig,'09_decoder_timing','Comparable decoder timing boundary. Includes Bayesian allocator probes and posterior draws. This is not total clinical latency, training time or notebook cumulative accounting.')
    # Genuine qualitative output, same anatomical selection rule for each completed fit.
    for name in list(runs):
        home=output/'jobs'/name/'test';arrays=sorted((home/'arrays').glob('*.npz'))
        if not arrays:continue
        chosen=arrays[:3]
        with np.load(chosen[0],allow_pickle=False) as z:has_mi='epistemic_MI_FP16' in z
        cols=5 if has_mi else 4;fig,axes=plt.subplots(len(chosen),cols,figsize=(2.25*cols,2.5*len(chosen)),squeeze=False)
        for i,p in enumerate(chosen):
            with np.load(p,allow_pickle=False) as z:
                panels=[('windowed_CT','CT'),('ground_truth','Reference'),('mean_probability_FP16','Probability'),('prediction_FP32_threshold','Prediction')]
                if has_mi:panels.append(('epistemic_MI_FP16','Between-seed MI' if name.startswith('det_ensemble') else 'Posterior MI'))
                for j,(key,title) in enumerate(panels):axes[i,j].imshow(z[key],cmap='gray' if j<2 else 'viridis');axes[i,j].set_title(title);axes[i,j].axis('off')
                xy=json.loads(str(z['identity_json']))['prompt_xy'];axes[i,0].scatter([xy[0]],[xy[1]],s=20,c='red',marker='+')
        save(fig,'qualitative_'+name,'Actual saved outputs. Alphabetically first three anatomy-selected targets among maximum-area liver/pancreas/right-adrenal slices per volume. No score-based cherry-picking. All selected targets and failures are available in arrays/records.')
    atomic(dest/'completion.json',{'completed_new_fits':sum(x['status']=='COMPLETE' for x in register),'planned_new_fits':len(register),'new_results_never_imputed':True,'external_confirmation':'[PENDING EXPERIMENT] requires untouched external data','declarations':'[AUTHOR INPUT]'})
    (dest/'INTERPRETATION.md').write_text('All new tests and sensitivities are exploratory: the original test results were inspected before the strengthening programme. Do not tune variants on these test scores. Volume-organ pairs and slices are not independent patients. Additional folds are internal replication. No publication guarantee or automatic 10/10 rating is possible. Historical original and fresh rerun scores must be kept in separate rows. Report all completed methods regardless of direction. Surface distances are 2D resized pixels. Calibration is strongly affected by background imbalance. A complete internal programme does not complete external validation or author/ethics declarations.\n')
