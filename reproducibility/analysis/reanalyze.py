"""Reproduce the single-split scalar analyses from the 7,398-row export."""
from pathlib import Path
from itertools import product
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import sem, spearmanr, t
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def descriptive_t_interval(x):
    x = np.asarray(x, dtype=float)
    return [float(v) for v in t.interval(.95, len(x)-1, loc=x.mean(), scale=sem(x))]


def paired_dice(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    samples = np.array(list(product(range(n), repeat=n)))
    signs = np.array(list(product([-1, 1], repeat=n)))
    return {
        'mean_pp': float(x.mean()),
        't_interval_pp': descriptive_t_interval(x),
        'percentile_interval_pp': np.percentile(x[samples].mean(1), [2.5, 97.5]).tolist(),
        'signflip_p': float((np.abs(signs @ x) >= abs(x.sum()) - 1e-10).mean()),
        'signflip_patterns': 2**n,
        'leave_one_out_pp': [float(np.delete(x,i).mean()) for i in range(n)],
    }


def within_volume_scores(df, score):
    return df.groupby('volume', sort=True).apply(
        lambda g: roc_auc_score(g['failure'], g[score]),
        include_groups=False,
    )


def risk_model(train, test, columns):
    pipe = make_pipeline(
        ColumnTransformer([
            ('organ', OneHotEncoder(handle_unknown='ignore'), ['organ']),
            ('continuous', StandardScaler(), columns),
        ]),
        LogisticRegression(C=1, max_iter=3000, solver='lbfgs'),
    )
    pipe.fit(train, train.failure)
    return pipe.predict_proba(test)[:, 1]


def validated_area(records):
    # TP is an integer recovered from recall times reference area.
    assert np.isfinite(records[['rec','prec','gt_area','pix_err']]).all().all()
    tp = np.rint(records.rec * records.gt_area).astype(int)
    area = (2 * tp + records.pix_err - records.gt_area).astype(int)
    assert area.between(0, 1024**2).all()
    assert np.allclose(2 * tp / (records.gt_area + area), records.dice, atol=1e-6, rtol=0)
    assert np.allclose(tp / np.maximum(area, 1), records.prec, atol=1e-6, rtol=0)
    return area


def crossfit(records, models, organ_prior=False):
    records = records.copy()
    records['log_pred_area'] = np.log1p(validated_area(records))
    for v in records.volume.unique():
        held_out = records.volume == v
        train, test = records.loc[~held_out], records.loc[held_out]
        if organ_prior:
            prior = train.groupby('organ').failure.mean()
            records.loc[held_out,'organ_only'] = test.organ.map(prior).fillna(train.failure.mean())
        for name, columns in models.items():
            records.loc[held_out,name] = risk_model(train, test, ['log_pred_area', *columns])
    names = list(models) + (['organ_only'] if organ_prior else [])
    return {name: within_volume_scores(records, name) for name in names}


def mi_failure_examples(records):
    """Within-volume, same-organ failure pair chosen by MI percentile gap."""
    examples = records.copy()
    examples['mi_rank'] = examples.groupby('volume').mi.rank(method='max', pct=True)
    eligible = examples.loc[examples.failure == 1].sort_values(['mi_rank','key'])
    groups = eligible.groupby(['volume','organ'])
    pairs = [(g.iloc[0], g.iloc[-1]) for _, g in groups if len(g) > 1]
    low, high = max(pairs, key=lambda pair: pair[1].mi_rank - pair[0].mi_rank)
    return [
        {'role': role, 'key': row.key, 'dice': float(row.dice),
         'mean_mi_nats': float(row.mi), 'within_volume_mi_percentile': float(row.mi_rank)}
        for role, row in [('low_ranked_failure',low), ('high_ranked_failure',high)]
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(args.csv)
    methods = ['frozen','deterministic','balora']
    assert len(d)==7398 and set(d.method)==set(methods)
    assert d.groupby('method').size().eq(2466).all()
    assert d.groupby('key').method.nunique().eq(3).all()
    assert d.groupby(['method','volume','organ']).ngroups==3*78
    d['failure'] = (d.dice < .5).astype(int)
    assert d.groupby(['method','volume']).failure.nunique().eq(2).all()
    pairmeans=d.groupby(['method','volume','organ']).dice.mean()
    byvol=pairmeans.groupby(['method','volume']).mean().unstack(0).sort_index()
    byvol['det_minus_gaussian_pp']=100*(byvol.deterministic-byvol.balora)
    contrast = paired_dice(byvol.det_minus_gaussian_pp)
    byvol.to_csv(args.out/'paired_dice_by_volume.csv')
    failure = d.groupby(['volume','method']).failure.agg(['size','sum'])
    failure.columns=['targets','failures']
    failure.to_csv(args.out/'failure_counts_by_volume.csv')

    scores = {
        'frozen_entropy': within_volume_scores(d[d.method=='frozen'], 'ent'),
        'deterministic_entropy': within_volume_scores(d[d.method=='deterministic'], 'ent'),
        'gaussian_entropy': within_volume_scores(d[d.method=='balora'], 'ent'),
        'gaussian_mi': within_volume_scores(d[d.method=='balora'], 'mi'),
    }
    scores = pd.DataFrame(scores)
    scores['paired_mi_minus_entropy'] = scores.gaussian_mi - scores.gaussian_entropy

    b = d.loc[d.method=='balora']
    det = d.loc[d.method=='deterministic']
    assert np.isfinite(b[['ent','mi']]).all().all() and np.isfinite(det.ent).all()
    gaussian_models = crossfit(b, {
        'organ_area': [], 'organ_area_mi': ['mi'],
        'organ_area_entropy': ['ent'], 'organ_area_entropy_mi': ['ent','mi'],
    }, organ_prior=True)
    deterministic_models = crossfit(det, {
        'det_organ_area': [], 'det_organ_area_entropy': ['ent'],
    })
    scores = scores.assign(**gaussian_models, **deterministic_models)
    scores['paired_added_mi']=scores.organ_area_mi-scores.organ_area
    scores['paired_added_mi_given_entropy']=scores.organ_area_entropy_mi-scores.organ_area_entropy
    scores.to_csv(args.out/'failure_auroc_by_volume.csv')
    examples = mi_failure_examples(b)
    (args.out/'replay_failure_targets.json').write_text(json.dumps(examples,indent=2)+'\n')

    timing=[]
    for m in methods:
        g=d.loc[d.method==m]
        same_slice=g.groupby(['volume','slice'])
        assert same_slice.enc_s.nunique().eq(1).all() and same_slice.ngroups==460
        byslice=same_slice.agg(encoder_s=('enc_s','first'),decoder_s=('dec_s','sum'),prompts=('key','size'))
        byslice['compute_s']=byslice.encoder_s+byslice.decoder_s
        byslice['method']=m
        timing.append(byslice.reset_index())
    timing=pd.concat(timing,ignore_index=True)
    timing.to_csv(args.out/'observed_slice_compute_times.csv',index=False)
    timing_summary={}
    for m,g in timing.groupby('method'):
        timing_summary[m]={
            'mean_encoder_ms':1000*g.encoder_s.mean(),
            'mean_decoder_ms_per_prompt':1000*g.decoder_s.sum()/g.prompts.sum(),
            'mean_compute_s':g.compute_s.mean(),
            'median_compute_s':g.compute_s.median(),
            'iqr_compute_s':g.compute_s.quantile([.25,.75]).tolist(),
        }

    byorgan=d.groupby(['method','volume','organ']).agg(dice=('dice','mean'),ece=('ece','mean')).groupby(['method','organ']).mean().unstack(0)
    byorgan.columns=['_'.join(c) for c in byorgan.columns]
    byorgan['median_reference_pixels']=b.groupby('organ').gt_area.median()
    byorgan['gaussian_minus_deterministic_dice_pp']=100*(byorgan.dice_balora-byorgan.dice_deterministic)
    byorgan.to_csv(args.out/'organ_diagnostics.csv')
    rho=float(spearmanr(byorgan.median_reference_pixels,byorgan.gaussian_minus_deterministic_dice_pp).statistic)
    intervals={
        'mi_minus_entropy_t_interval':descriptive_t_interval(scores.paired_mi_minus_entropy),
        'added_mi_minmax': [float(scores.paired_added_mi.min()),float(scores.paired_added_mi.max())],
        'added_mi_t_interval':descriptive_t_interval(scores.paired_added_mi),
        'added_mi_given_entropy_t_interval':descriptive_t_interval(scores.paired_added_mi_given_entropy),
        'added_mi_given_entropy_minmax': [float(scores.paired_added_mi_given_entropy.min()),
                                        float(scores.paired_added_mi_given_entropy.max())],
    }
    g=d.loc[d.method=='balora']
    assert (g[['n4','n8','n16']].sum(axis=1)==32).all()
    rank_counts = {int(k):int(g[f'n{k}'].sum()) for k in (4,8,16)}
    results={
        'prompt_records':len(d),
        'dice_pair_weighted':pairmeans.groupby('method').mean().to_dict(),
        'dice_prompt_weighted':d.groupby('method').dice.mean().to_dict(),
        'paired_dice':contrast,
        'failure_counts':d.groupby('method').failure.sum().to_dict(),
        'within_volume_auroc_means':scores.mean().to_dict(),
        'within_volume_intervals':intervals,
        'timing':timing_summary,
        'decoder_ratio':timing_summary['balora']['mean_decoder_ms_per_prompt']/timing_summary['deterministic']['mean_decoder_ms_per_prompt'],
        'slice_compute_ratio':timing_summary['balora']['mean_compute_s']/timing_summary['deterministic']['mean_compute_s'],
        'rank_counts':rank_counts,
        'rank16_per_prompt':float(g.n16.mean()),
        'prompts_without_rank16':int((g.n16==0).sum()),
        'mean_active_rank':sum(k*v for k,v in rank_counts.items())/sum(rank_counts.values()),
        'empty_counts':d.groupby('method')['empty'].sum().to_dict(),
        'missing_surface_counts':d.groupby('method').hd95.apply(lambda s: int((~np.isfinite(s)).sum())).to_dict(),
        'organ_area_vs_dice_change_spearman':rho,
    }
    (args.out/'analysis_summary.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
