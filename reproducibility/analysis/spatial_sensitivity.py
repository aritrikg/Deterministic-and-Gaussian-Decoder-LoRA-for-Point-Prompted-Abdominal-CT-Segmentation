"""Retrospective uncertainty sensitivity using the completed F1 archive."""
from pathlib import Path
from zipfile import ZipFile
from hashlib import file_digest
import argparse
import io
import json

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, binary_erosion

from audit_archive import ARCHIVE_HASH
from reanalyze import crossfit


FEATURES = {
    'full_oa': [],
    'full_oam': ['mi'],
    'full_oae': ['ent'],
    'full_oaem': ['ent', 'mi'],
    'full_oae_boundary_mi': ['ent', 'boundary_mi'],
    'boundary_oam': ['boundary_mi'],
    'boundary_oae': ['boundary_entropy'],
    'boundary_oaem': ['boundary_entropy', 'boundary_mi'],
}


def read_archive(archive, keys):
    with archive.open('rb') as stream:
        if file_digest(stream, 'sha256').hexdigest() != ARCHIVE_HASH:
            raise ValueError('Expected the completed F1 archive')
    boundary_rows = []
    array_rows = []
    with ZipFile(archive) as z:
        prefix = z.namelist()[0].split('/')[0] + '/balora/test'
        for key in keys:
            row = json.loads(z.read(f'{prefix}/records/{key.replace(":", "_")}.json'))
            assert row['key'] == key and row['method'] == 'balora'
            ranking = row['diagnostics']['ranking']
            entropy = ranking['predictive_entropy']
            mi = ranking['epistemic_MI']
            band = 'GT_boundary_2px'
            assert entropy[band]['n'] == mi[band]['n'] > 0
            boundary_rows.append({
                'key': key, 'archive_dice': row['dice'],
                'archive_entropy': entropy['all']['mean_uncertainty'],
                'archive_mi': mi['all']['mean_uncertainty'],
                'boundary_pixels': mi[band]['n'],
                'boundary_entropy': entropy[band]['mean_uncertainty'],
                'boundary_mi': mi[band]['mean_uncertainty'],
            })
        for name in sorted(z.namelist()):
            if not name.startswith(prefix + '/arrays/') or not name.endswith('.npz'):
                continue
            case, slice_index, organ = Path(name).stem.rsplit('_', 2)
            key = f'{case}:{slice_index}:{organ}'
            with np.load(io.BytesIO(z.read(name)), allow_pickle=False) as a:
                mask = a['prediction_FP32_threshold'].astype(bool)
                prob = a['mean_probability_FP16'].astype(np.float64)
                mi = a['epistemic_MI_FP16'].astype(np.float64)
            band = binary_dilation(mask, iterations=2) ^ binary_erosion(mask, iterations=2)
            edges = np.pad(mask, 1)
            perimeter = np.count_nonzero(edges[1:, :] != edges[:-1, :])
            perimeter += np.count_nonzero(edges[:, 1:] != edges[:, :-1])
            p = prob.clip(np.finfo(np.float32).eps, 1 - np.finfo(np.float32).eps)
            ent = -p*np.log(p) - (1-p)*np.log1p(-p)
            array_rows.append({
                'key': key, 'predicted_area_px': int(mask.sum()),
                'predicted_perimeter_px_edges': int(perimeter),
                'predicted_band_pixels': int(band.sum()),
                'predicted_band_entropy_nats_fp16': ent[band].mean() if band.any() else None,
                'predicted_band_mi_nats_fp16': mi[band].mean() if band.any() else None,
                'predicted_foreground_mi_nats_fp16': mi[mask].mean() if mask.any() else None,
            })
    return pd.DataFrame(boundary_rows), pd.DataFrame(array_rows)


def scores(records):
    scores = pd.DataFrame(crossfit(records, FEATURES))
    for label in ('full', 'boundary'):
        scores[label+'_mi_increment_given_entropy'] = scores[label+'_oaem'] - scores[label+'_oae']
        scores[label+'_mi_increment_given_area'] = scores[label+'_oam'] - scores['full_oa']
    scores['boundary_mi_increment_given_full_entropy'] = scores['full_oae_boundary_mi'] - scores['full_oae']
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument('archive', type=Path)
    p.add_argument('data', type=Path)
    args = p.parse_args()
    d = pd.read_csv(args.data/'per_prompt_fold1_seed42.csv')
    d = d.loc[d.method == 'balora'].copy()
    assert len(d) == 2466 and d.key.is_unique and d.volume.nunique() == 6
    boundary, arrays = read_archive(args.archive, d.key)
    d = d.merge(boundary, on='key', validate='one_to_one')
    assert len(d) == 2466
    for a, b in [('dice','archive_dice'), ('ent','archive_entropy'), ('mi','archive_mi')]:
        assert np.allclose(d[a], d[b], atol=1e-10, rtol=0)
    d['failure'] = (d.dice < .5).astype(int)
    assert d.failure.sum() == 200
    by_volume = scores(d)
    omissions = pd.DataFrame({v: scores(d.loc[d.volume != v]).mean() for v in sorted(d.volume.unique())}).T
    arrays = arrays.merge(d[['key', 'dice', 'failure']], on='key', validate='one_to_one')
    assert len(arrays) == 18
    by_volume.to_csv(args.data/'boundary_ranking_by_volume.csv')
    omissions.rename_axis('excluded_volume').to_csv(args.data/'failure_ranking_exclude_one_refit.csv')
    boundary.to_csv(args.data/'archived_reference_boundary_scores.csv', index=False)
    arrays.to_csv(args.data/'saved18_predicted_boundary_scores.csv', index=False)
    result = {
        'reference_boundary':'two-pixel binary dilation XOR two-pixel erosion of the ground-truth mask',
        'targets':len(d), 'volumes':d.volume.nunique(), 'failures':int(d.failure.sum()),
        'failure_count_by_volume':d.groupby('volume').failure.sum().to_dict(),
        'mean_auroc':by_volume.mean().to_dict(),
        'refit_mean_auroc_ranges':{c:[float(omissions[c].min()), float(omissions[c].max())] for c in omissions},
        'predicted_region_saved_targets':len(arrays),
        'predicted_region_saved_failures':int(arrays.failure.sum()),
    }
    (args.data/'spatial_sensitivity.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
