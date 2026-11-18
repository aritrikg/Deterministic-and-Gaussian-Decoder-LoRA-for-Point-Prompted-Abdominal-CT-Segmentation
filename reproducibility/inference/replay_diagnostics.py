"""Audit SAM2 mask selection, regional uncertainty and matched encoder timing."""
from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter
from unittest.mock import patch

import numpy as np


@contextmanager
def trace_fallback(decoder):
    """Observe the original forward decisions without changing mask selection."""
    calls, scores = [], []
    score_fn = decoder._get_stability_scores
    threshold = decoder.dynamic_multimask_stability_thresh

    def forward_start(module, args, kwargs):
        if not kwargs['multimask_output']:
            calls.append(bool(module.dynamic_multimask_via_stability and not module.training))

    def capture(mask_logits):
        score = score_fn(mask_logits)
        scores.append(score.detach())
        return score

    hook = decoder.register_forward_pre_hook(forward_start, with_kwargs=True)
    try:
        with patch.object(decoder, '_get_stability_scores', capture):
            yield calls, scores, threshold
    finally:
        hook.remove()


def fallback_counts(arm, trace):
    enabled, scores, threshold = trace
    phases = {
        'frozen': [('primary_single', 1)],
        'deterministic': [('primary_single', 1), ('mc_dropout', 4)],
        'gaussian_learned': [('rank_probe', 2), ('final_draw', 4)],
        'gaussian_fixed_sd': [('rank_probe', 2), ('final_draw', 4)],
    }[arm]
    if len(enabled) != sum(n for _, n in phases) or len(scores) != sum(enabled):
        raise ValueError(f'Unexpected SAM2 fallback trace for {arm}: {len(enabled)} calls, {len(scores)} scores')
    values = iter(scores)
    replacements = []
    for active in enabled:
        if active:
            score = next(values)
            value = float(score.item())
            if not np.isfinite(value):
                raise ValueError('Nonfinite SAM2 stability score')
            replacements.append(not bool((score >= threshold).item()))
        else:
            replacements.append(False)
    report, offset = {}, 0
    for phase, n in phases:
        end = offset + n
        report[phase] = {
            'single_output_calls':n, 'enabled_calls':sum(enabled[offset:end]),
            'substitutions':sum(replacements[offset:end]),
        }
        offset = end
    return report


def summarize_fallback(rows):
    counts = defaultdict(lambda: defaultdict(lambda: {
        'targets':0, 'single_output_calls':0, 'enabled_calls':0,
        'substitutions':0, 'targets_with_any_substitution':0,
    }))
    for row in rows:
        for phase, values in row['diagnostics']['fallback_counts'].items():
            for name in ('overall', row['volume']):
                counter = counts[phase][name]
                counter['targets'] += 1
                counter['targets_with_any_substitution'] += int(values['substitutions'] > 0)
                for key, value in values.items():
                    counter[key] += value
    for vols in counts.values():
        for v in vols.values():
            v['substitution_rate_per_call'] = v['substitutions'] / v['single_output_calls']
            v['substitution_rate_per_target'] = v['targets_with_any_substitution'] / v['targets']
    return {phase:dict(values) for phase, values in counts.items()}


def mask_region_scores(mask, entropy, mi):
    from scipy.ndimage import binary_dilation, binary_erosion

    padded = np.pad(mask, 1)
    perimeter = np.count_nonzero(padded[1:, :] != padded[:-1, :])
    perimeter += np.count_nonzero(padded[:, 1:] != padded[:, :-1])
    band = binary_dilation(mask, iterations=2) ^ binary_erosion(mask, iterations=2)
    return {
        'predicted_perimeter_px_edges':int(perimeter),
        'predicted_band_pixels':int(band.sum()),
        'mean_entropy_predicted_band_nats':float(entropy[band].mean()) if band.any() else None,
        'mean_mi_predicted_band_nats':float(mi[band].mean()) if mi is not None and band.any() else None,
    }


def crossfit_regions(rows):
    import pandas as pd
    from replay_risk import risk_model, within_volume_scores

    d = pd.DataFrame(rows)
    d['failure'] = (d.dice < .5).astype(int)
    d['log_area'] = np.log1p(d.predicted_area_pixels)
    d['log_perimeter'] = np.log1p(d.predicted_perimeter_px_edges)
    if d.groupby('volume').failure.nunique().ne(2).any():
        return {'available':False, 'reason':'A volume has no positives or no negatives',
                'failures_by_volume':d.groupby('volume').failure.sum().to_dict()}
    # For an empty prediction, the band does not exist. Encode absent means as
    # zero for the risk-model feature only. Area/perimeter carry the empty state.
    d['band_entropy'] = d.mean_entropy_predicted_band_nats.fillna(0.)
    d['band_mi'] = d.mean_mi_predicted_band_nats.fillna(0.)
    pairs = {
        'full': ['log_area', 'mean_entropy_full_nats'],
        'full_perimeter': ['log_area', 'log_perimeter', 'mean_entropy_full_nats'],
        'predicted_band_perimeter': ['log_area', 'log_perimeter', 'band_entropy'],
    }
    result = {'available':True, 'failures_by_volume':d.groupby('volume').failure.sum().to_dict(),
              'empty_bands':int(d.predicted_band_pixels.eq(0).sum()), 'models':{}}
    for name, base in pairs.items():
        mi = 'band_mi' if name == 'predicted_band_perimeter' else 'mean_mi_full_nats'
        for vol in sorted(d.volume.unique()):
            holdout = d.volume == vol
            train, test = d.loc[~holdout], d.loc[holdout]
            d.loc[holdout, 'base'] = risk_model(train, test, base)
            d.loc[holdout, 'plus_mi'] = risk_model(train, test, base + [mi])
        b, m = within_volume_scores(d,'base'), within_volume_scores(d,'plus_mi')
        result['models'][name] = {
            'base_auroc_by_volume':b.to_dict(), 'with_mi_auroc_by_volume':m.to_dict(),
            'paired_increment_by_volume':(m-b).to_dict(), 'mean_increment':float((m-b).mean()),
        }
    return result


def encoder_timing(configs, weights, dataset, selected, ledger):
    import torch
    from full_worker import make_model

    methods = ('deterministic', 'balora')
    warmups, repeats = 3, 3
    by_volume = defaultdict(list)
    for i, (cid, z, _) in enumerate(dataset._clip_index):
        if f'{cid}:{z}' in selected:
            by_volume[cid].append(i)
    indices = [members[i] for _, members in sorted(by_volume.items())
               for i in sorted({len(members)//4, 3*len(members)//4})]
    ledger.check()
    models = {m:make_model(configs[m],m,weights[m]).eval() for m in methods}
    rows = []
    with torch.no_grad():
        first = dataset[indices[0]].image.to('cuda:0')
        for _ in range(warmups):
            for model in models.values():
                model.encode_image(first)
        torch.cuda.synchronize()
        for sample_number, index in enumerate(indices):
            ledger.check()
            sample = dataset[index]
            image = sample.image.to('cuda:0')
            seconds = {m:[] for m in methods}
            for repeat in range(repeats):
                order = methods if (sample_number + repeat) % 2 == 0 else methods[::-1]
                for m in order:
                    torch.cuda.synchronize()
                    start = perf_counter()
                    result = models[m].encode_image(image)
                    torch.cuda.synchronize()
                    seconds[m].append(perf_counter()-start)
                    del result
            medians = {m:float(np.median(v)) for m,v in seconds.items()}
            rows.append({'volume':sample.case_id, 'slice_index':int(sample.slice_indices[0]),
                         'encoder_seconds_repeats':seconds, 'encoder_seconds_medians':medians,
                         'paired_deterministic_minus_gaussian_seconds':medians['deterministic']-medians['balora']})
    return {'warmups_per_model_excluded':warmups, 'repeats_per_slice':repeats, 'slices':rows,
            'sample_mean_paired_difference_seconds':float(np.mean([
                r['paired_deterministic_minus_gaussian_seconds'] for r in rows]))}
