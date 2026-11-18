"""Verify the completed Round 3 replay against the archived Fold 1 outputs."""
import argparse
import base64
import csv
from collections import Counter, defaultdict
from hashlib import file_digest, sha256
from io import StringIO
import json
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from reanalyze import descriptive_t_interval, risk_model


ARMS = {
    'frozen': {'frozen_single', 'frozen_multimask_iou', 'frozen_multimask_oracle',
               'frozen_single_or_multimask_oracle'},
    'deterministic': {'deterministic_single', 'decoder_mc_dropout'},
    'gaussian_learned': {'gaussian_learned'},
    'gaussian_fixed_sd': {'gaussian_fixed_sd'},
}
REFERENCES = {'frozen': 'frozen_single', 'deterministic': 'deterministic_single',
              'balora': 'gaussian_learned'}


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def read_json(archive, path):
    return json.loads(archive.read(path))


def close(value, expected):
    np.testing.assert_allclose(value, expected, rtol=0, atol=2e-12)


def verify(original_path, replay_path, output):
    output.mkdir(exist_ok=True, parents=True)
    with ZipFile(replay_path) as replay, ZipFile(original_path) as original:
        assert replay.testzip() is None
        names = replay.namelist()
        assert len(set(names)) == len(names)
        status = read_json(replay, 'status.json')
        identity = read_json(replay, 'provenance/identity.json')
        canonical = json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()
        assert sha256(canonical).hexdigest() == status['identity']
        assert status['status'] == 'COMPLETE' and status['full_cohort_complete']
        assert status['scope'] == identity['scope'] == 'full'
        assert identity['archive_sha256'] == digest(original_path)
        notebook = json.loads((Path(__file__).resolve().parents[1] /
                               'BALoRA_Round3_Inference_Audit.ipynb').read_text())
        source = next(''.join(c['source']) for c in notebook['cells']
                      if 'PAYLOAD_SHA256 = ' in ''.join(c['source']))
        payload = re.search(r'base64\.b64decode\("""(.*?)"""\)', source, re.S)[1]
        assert sha256(base64.b64decode(payload)).hexdigest() == identity['source_sha256']
        reference = {}
        for entry in original.namelist():
            pieces = entry.split('/')
            if len(pieces) == 5 and pieces[1] in REFERENCES and pieces[2:4] == ['test','records']:
                row = read_json(original, entry)
                reference[REFERENCES[pieces[1]], row['key']] = row
        assert len(reference) == 7398
        for method, expected in identity['weight_sha256'].items():
            checkpoint = next(n for n in original.namelist()
                              if n.endswith(f'/{method}/research/best.pt'))
            with original.open(checkpoint) as stream:
                assert file_digest(stream, 'sha256').hexdigest() == expected
        rows = {}
        counts = defaultdict(Counter)
        slices = defaultdict(set)
        targets = defaultdict(set)
        ranks = defaultdict(Counter)
        session_ids = set()
        for entry in names:
            if not entry.startswith('records/'):
                continue
            arm = entry.split('/')[1]
            saved = read_json(replay, entry)
            assert saved['identity'] == status['identity']
            slices[arm].add(saved['slice'])
            session_ids.add(saved['session_id'])
            for row in saved['rows']:
                key = row['key']
                assert key not in targets[arm] and key.rsplit(':',1)[0] == saved['slice']
                targets[arm].add(key)
                assert row['prompt_xy'] == reference['gaussian_learned',key]['prompt_xy']
                assert set(row['scores']) == ARMS[arm]
                volume, z, organ = key.split(':')
                for variant, metrics in row['scores'].items():
                    rows[variant,key] = {'volume':volume, 'slice_index':int(z), 'organ':int(organ),
                                         'x':row['prompt_xy'][0], 'y':row['prompt_xy'][1], **metrics}
                diagnostics = row['diagnostics']
                if 'rank_counts' in diagnostics:
                    ranks[arm].update(diagnostics['rank_counts'])
                if arm == 'frozen':
                    dice = diagnostics['candidate_dice']
                    chosen = int(np.argmax(diagnostics['predicted_iou']))
                    assert diagnostics['iou_candidate_index'] == chosen
                    close(row['scores']['frozen_multimask_iou']['dice'], dice[chosen])
                    close(row['scores']['frozen_multimask_oracle']['dice'], max(dice))
                    close(row['scores']['frozen_single_or_multimask_oracle']['dice'],
                          max(max(dice), row['scores']['frozen_single']['dice']))
                for phase, detail in diagnostics['fallback_counts'].items():
                    assert 0 <= detail['substitutions'] <= detail['enabled_calls'] <= detail['single_output_calls']
                    for group in ('overall', volume):
                        counts[arm,phase,group].update(detail)
                        counts[arm,phase,group]['targets'] += 1
                        counts[arm,phase,group]['targets_with_any_substitution'] += int(detail['substitutions'] > 0)
                if 'array_file' in row:
                    assert sha256(replay.read(row['array_file'])).hexdigest() == row['array_sha256']
        assert len(rows) == 19728 and len(session_ids) == 1
        for arm in ARMS:
            assert len(slices[arm]) == 460 and len(targets[arm]) == 2466
            assert status['coverage'][arm] == {'scope_complete':True,'slices':460,'targets':2466}
            assert targets[arm] == targets['frozen']
        for key,row in reference.items():
            observed = rows[key]
            assert observed['dice'] == row['dice']
            assert [observed['x'],observed['y']] == row['prompt_xy']
        csv_rows = list(csv.DictReader(StringIO(replay.read('per_prompt.csv').decode())))
        assert len(csv_rows) == len(rows)
        assert {(r['variant'], r['key']) for r in csv_rows} == rows.keys()
        for row in csv_rows:
            for col, value in rows[row['variant'],row['key']].items():
                expected = row[col]
                if value is None:
                    assert expected == ''
                elif isinstance(value, (int,float)):
                    assert float(expected) == value
                else:
                    assert expected == value
        variant_rows = []
        all_dice = {}
        for variant, d in status['variant_results'].items():
            df = pd.DataFrame([r for (v,_),r in rows.items() if v==variant])
            pairs = df.groupby(['volume','organ']).dice.mean()
            assert len(pairs) == 78 and len(df) == 2466
            close([pairs.mean(),df.dice.mean(),(df.dice < .5).sum()],
                  [d['pair_weighted_dice'],d['prompt_weighted_dice'],d['failures_dice_lt_0_5']])
            all_dice[variant] = pairs.groupby('volume').mean()
            variant_rows.append({'variant':variant,'pair_weighted_dice':pairs.mean(),
                                 'prompt_weighted_dice':df.dice.mean(),
                                 'failures':int((df.dice < .5).sum())})
        paired_rows = []
        def check_paired(diff, published):
            close(diff.mean(), published['mean'])
            close(descriptive_t_interval(diff), published['descriptive_95pct_t_interval'])
            close(diff, [published['differences_by_volume'][v] for v in diff.index])

        for effect, d in status['paired_dice_effects'].items():
            a, b = effect.split('_minus_')
            delta = all_dice[a] - all_dice[b]
            check_paired(delta, d)
            paired_rows.extend({'comparison':effect,'volume':v,'difference':x} for v,x in delta.items())
        pd.DataFrame(paired_rows).to_csv(output/'paired_dice.csv',index=False)
        auc_rows = []
        for alt in ('decoder_mc_dropout','gaussian_fixed_sd'):
            d = status[f'{alt}_fixed_outcome_comparison']
            ref = d['outcomes_from']
            grouped = defaultdict(list)
            for key in targets['frozen']:
                a, b = rows[alt,key], rows[ref,key]
                grouped[b['volume']].append((b['dice'] < .5,b[d['reference_score']],a['mean_mi_full_nats']))
            diffs = {}
            for v, data in sorted(grouped.items()):
                y, ref_scores, alt_scores = zip(*data)
                ref_auc, alt_auc = roc_auc_score(y,ref_scores), roc_auc_score(y,alt_scores)
                close([ref_auc,alt_auc], [d['reference_auroc_by_volume'][v],d['alternative_mi_auroc_by_volume'][v]])
                diffs[v] = alt_auc-ref_auc
                auc_rows.append({'alternative':alt,'volume':v,'failures':sum(y),
                                 'reference_auroc':ref_auc,'alternative_auroc':alt_auc})
            check_paired(pd.Series(diffs),d['paired_auroc_increment'])
        fallback_rows = []
        for (arm,phase,v), count in sorted(counts.items()):
            d = status['single_output_fallback_by_phase'][arm][phase][v]
            assert all(d[k] == count[k] for k in count)
            close(d['substitution_rate_per_call'],count['substitutions']/count['single_output_calls'])
            close(d['substitution_rate_per_target'],count['targets_with_any_substitution']/count['targets'])
            fallback_rows.append({'arm':arm,'phase':phase,'volume':v,**d})
        regional_rows = []
        for variant, report in status['regional_failure_models'].items():
            # Reuse the row order recorded in the archive for numerical reproducibility.
            df = pd.DataFrame([rows[r['variant'],r['key']] for r in csv_rows if r['variant']==variant])
            df['failure'] = df.dice < .5
            df['log_area'] = np.log1p(df.predicted_area_pixels)
            df['log_perimeter'] = np.log1p(df.predicted_perimeter_px_edges)
            df['band_entropy'] = df.mean_entropy_predicted_band_nats.fillna(0)
            df['band_mi'] = df.mean_mi_predicted_band_nats.fillna(0)
            specs = {'full': (['log_area','mean_entropy_full_nats'],'mean_mi_full_nats'),
                     'full_perimeter': (['log_area','log_perimeter','mean_entropy_full_nats'],'mean_mi_full_nats'),
                     'predicted_band_perimeter': (['log_area','log_perimeter','band_entropy'],'band_mi')}
            for name,(cols,mi) in specs.items():
                for vol, hold in df.groupby('volume',sort=True):
                    train = df.loc[df.volume != vol]
                    a = roc_auc_score(hold.failure, risk_model(train,hold,cols))
                    b = roc_auc_score(hold.failure, risk_model(train,hold,cols+[mi]))
                    m = report['models'][name]
                    close([a,b],[m['base_auroc_by_volume'][vol],m['with_mi_auroc_by_volume'][vol]])
                    regional_rows.append({'variant':variant,'model':name,'volume':vol,
                                          'base_auroc':a,'with_mi_auroc':b,'increment':b-a})
        timing = read_json(replay,'encoder_timing.json')
        assert timing['identity'] == status['identity']
        assert len(timing['slices']) == 12 and timing['repeats_per_slice'] == timing['warmups_per_model_excluded'] == 3
        timing_rows = []
        for row in timing['slices']:
            d = {m:float(np.median(x))*1000 for m,x in row['encoder_seconds_repeats'].items()}
            for m in d:
                close(d[m], row['encoder_seconds_medians'][m]*1000)
            timing_rows.append({'volume':row['volume'],'slice':row['slice_index'],**d,
                                'difference_ms':d['deterministic']-d['balora']})
        timing_df = pd.DataFrame(timing_rows)
        assert timing_df.groupby('volume').size().eq(2).all()
        close(timing_df.difference_ms.mean(),timing['sample_mean_paired_difference_seconds']*1000)
        tables = {'variant_summary':variant_rows,'fixed_outcome_auroc':auc_rows,
                  'fallback_counts':fallback_rows,'regional_failure_models':regional_rows,
                  'encoder_timing':timing_rows}
        for name, table in tables.items():
            pd.DataFrame(table).to_csv(output/f'{name}.csv',index=False)
        audit = {'status':'VERIFIED', 'archive_sha256':digest(replay_path),'identity':identity,
                 'slice_records':sum(map(len,slices.values())), 'variant_prompt_rows':len(rows),
                 'coverage':status['coverage'], 'max_primary_dice_replay_error':0,
                 'rank_counts':ranks, 'encoder_mean_of_slice_medians_ms':timing_df.mean(numeric_only=True).drop('slice').to_dict()}
        (output/'verification.json').write_text(json.dumps(audit,indent=2)+'\n')
        for name in ('status.json','encoder_timing.json'):
            (output/name).write_bytes(replay.read(name))
        print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('original',type=Path)
    cli.add_argument('replay',type=Path)
    cli.add_argument('output',type=Path)
    args = cli.parse_args()
    verify(args.original,args.replay,args.output)
