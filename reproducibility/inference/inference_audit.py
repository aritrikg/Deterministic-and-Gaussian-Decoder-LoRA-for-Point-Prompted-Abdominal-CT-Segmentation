"""Replay the selected F1 checkpoints with mask-selection and regional diagnostics."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import csv
import gc
import json
import math
import os
import shutil
import sys
import time
import traceback

import numpy as np

from auditlib.io_budget import atomic_json, digest_file, digest_json, BudgetStop
from run_budget import Ledger, safe_extract
from replay_diagnostics import (trace_fallback, fallback_counts, summarize_fallback,
                                mask_region_scores, crossfit_regions, encoder_timing)

ROOT = Path(__file__).resolve().parent
ARMS = {
    'gaussian_learned': ('balora', ('gaussian_learned',)),
    'gaussian_fixed_sd': ('balora', ('gaussian_fixed_sd',)),
    'deterministic': ('deterministic', ('deterministic_single', 'decoder_mc_dropout')),
    'frozen': ('frozen', ('frozen_single', 'frozen_multimask_iou', 'frozen_multimask_oracle',
                          'frozen_single_or_multimask_oracle')),
}
SEED = 700043
DRAWS = 4


def read_json(path):
    return json.loads(Path(path).read_text())


def checkpoint_paths(cfg, work, reference):
    """Accept the completed ZIP or extracted weights with the same digests."""
    root = Path(cfg['input_root'])
    explicit = Path(cfg['result_path']) if cfg['result_path'] else None
    hashes = reference['selected_weights']
    work.mkdir(parents=True, exist_ok=True)
    if explicit is not None and not explicit.exists():
        raise FileNotFoundError(f'Results path does not exist: {explicit}')
    if explicit is None:
        zips = sorted(root.rglob('BALoRA_Full_Run_Results*.zip'))
    else:
        zips = [explicit] if explicit.is_file() else []
    for path in zips:
        if digest_file(path) != reference['archive_sha256']:
            continue
        with ZipFile(path) as archive:
            prefix = reference['archive_prefix']
            selected = {}
            for method, sha in hashes.items():
                dest = work / (method + '_best.pt')
                if not dest.is_file() or digest_file(dest) != sha:
                    temp = dest.with_suffix('.tmp')
                    try:
                        with archive.open(f'{prefix}/{method}/research/best.pt') as src, temp.open('wb') as dst:
                            shutil.copyfileobj(src, dst)
                        if digest_file(temp) != sha:
                            raise ValueError(f'Checkpoint checksum mismatch for {method}')
                        os.replace(temp, dest)
                    finally:
                        temp.unlink(missing_ok=True)
                selected[method] = dest
            return selected
    extracted = explicit or root
    selected = {}
    if extracted.is_dir():
        for method, sha in hashes.items():
            for path in sorted(extracted.rglob('best.pt')):
                if path.parent.name == 'research' and path.parent.parent.name == method and digest_file(path) == sha:
                    selected[method] = path
                    break
    if set(selected) != set(hashes):
        raise FileNotFoundError(
            'Attach the completed BALoRA_Full_Run_Results(1).zip or its extracted files. '
            'Set RESULT_PATH to the exact attachment path if the archive was renamed. '
            f'The required archive SHA-256 is {reference["archive_sha256"]}.')
    return selected


def get_model_cfg(reference, method, root, work, environment):
    cfg = deepcopy(reference['configs'][method])
    work.mkdir(parents=True, exist_ok=True)
    split = work / 'test.json'
    atomic_json(split, {'cases': reference['test_cases']})
    cfg['sam2']['checkpoint'] = environment['backbone_checkpoint']
    cfg['data'].update(root_dir=root, test_split=str(split), cache_dir=str(work / 'cache'))
    return cfg


def frozen_multimask(model, embedding, points, labels):
    """Use the pinned SAM2 decoder's three explicit multimask candidates."""
    import torch
    import torch.nn.functional as F

    base = model.backbone
    image_embed = embedding['image_embed']
    encoder = base.sam_prompt_encoder
    sparse, dense = encoder(points=(points.float(), labels.to(torch.int32)), boxes=None, masks=None)
    masks, quality, _, _ = base.sam_mask_decoder(
        image_embeddings=image_embed,
        image_pe=encoder.get_dense_pe().to(device=image_embed.device, dtype=image_embed.dtype),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=True, repeat_image=False, high_res_features=embedding.get('high_res_feats'),
    )
    if tuple(masks.shape[:2]) != (1, 3) or tuple(quality.shape) != (1, 3):
        raise ValueError(f'Unexpected SAM2 candidate shapes {masks.shape} / {quality.shape}')
    if not torch.isfinite(quality).all():
        raise ValueError('Nonfinite SAM2 candidate quality')
    return F.interpolate(masks.float(), size=(1024, 1024), mode='bilinear', align_corners=False)[0].sigmoid(), quality[0]


def get_point(target, key, expected):
    import torch
    from inference_common import seed_for
    from utils.prompting import sample_foreground_points

    cid, z, organ = key.split(':')
    torch.manual_seed(seed_for(SEED, cid, int(z), int(organ)))
    point = sample_foreground_points(target)
    if [int(v) for v in point[0, 0].tolist()] != expected:
        raise ValueError(f'Archived prompt differs at {key}. Check the data and runtime versions.')
    labels = torch.ones((1, 1), dtype=torch.int32, device=target.device)
    return point, labels


def dice_value(probability, truth):
    mask = probability >= np.float32(.5)
    overlap = np.float32(np.count_nonzero(mask & truth))
    denom = np.float32(np.count_nonzero(mask) + np.count_nonzero(truth))
    eps = np.float32(1e-6)
    return float((np.float32(2.) * overlap + eps) / (denom + eps))


def scalar_scores(probability, truth, mi):
    from auditlib.pixel_metrics import entropy

    if probability.shape != truth.shape or not np.isfinite(probability).all():
        raise ValueError('Probability shape or values invalid')
    foreground = probability >= .5
    ent = entropy(probability)
    if mi is not None and (mi.shape != truth.shape or not np.isfinite(mi).all()):
        raise ValueError('MI shape or values invalid')
    return {
        'dice': dice_value(probability, truth),
        'predicted_area_pixels': int(foreground.sum()),
        'mean_entropy_full_nats': float(ent.mean()),
        'mean_entropy_predicted_region_nats': float(ent[foreground].mean()) if foreground.any() else None,
        'mean_mi_full_nats': None if mi is None else float(mi.mean()),
        'mean_mi_predicted_region_nats': None if mi is None or not foreground.any()
        else float(mi[foreground].mean()),
        **mask_region_scores(foreground, ent, mi),
    }


def decode_arm(model, arm, embedding, target, key, prompt, truth):
    import torch
    from inference_common import predict_one, seed_for
    from models.confidence import estimate_uncertainty

    cid, z, organ = key.split(':')
    if arm.startswith('gaussian'):
        p, mi, ranks, _ = predict_one(model, 'balora', embedding, target, SEED, cid, int(z), int(organ), prompt)
        row = {arm: (p, mi)}
        rank_counts = dict(Counter(int(rank) for rank in ranks.values()))
        if sum(rank_counts.values()) != 32:
            raise ValueError('Expected ranks for all 32 decoder adapters')
        return row, {'rank_counts': {str(k): v for k, v in sorted(rank_counts.items())}}

    points, labels = get_point(target, key, prompt)
    single = model.decode_one(embedding, points, labels).sigmoid()[0, 0]
    if arm == 'deterministic':
        from models.bayesian_lora import DeterministicLoRALinear

        adapters = [m.dropout for m in model.backbone.sam_mask_decoder.modules()
                    if isinstance(m, DeterministicLoRALinear)]
        if len(adapters) != 32 or any(not isinstance(d, torch.nn.Dropout) or d.p != .05 for d in adapters):
            raise ValueError('Expected 32 decoder-only dropout modules with p=0.05')
        torch.manual_seed(seed_for(SEED, cid, int(z), int(organ)))
        try:
            for module in adapters:
                module.train(True)
            logits = torch.stack([model.decode_one(embedding, points, labels) for _ in range(DRAWS)])
        finally:
            for module in adapters:
                module.train(False)
        maps = estimate_uncertainty(logits.squeeze(2))
        return {
            'deterministic_single': (single, None),
            'decoder_mc_dropout': (maps.mean_prob[0], maps.epistemic[0]),
        }, {}

    p, quality = frozen_multimask(model, embedding, points, labels)
    scores = [dice_value(candidate, truth) for candidate in p.cpu().numpy()]
    chosen = int(quality.argmax())
    oracle = int(np.argmax(scores))
    single_dice = dice_value(single.cpu().numpy(), truth)
    return {
        'frozen_single': (single, None),
        'frozen_multimask_iou': (p[chosen], None),
        'frozen_multimask_oracle': (p[oracle], None),
        'frozen_single_or_multimask_oracle': (single if single_dice >= scores[oracle] else p[oracle], None),
    }, {'predicted_iou': quality.cpu().tolist(), 'candidate_dice': scores,
        'iou_candidate_index': chosen, 'oracle_candidate_index': oracle}


def change_sd(model):
    import torch
    from models.bayesian_lora import BayesianLoRALinear

    layers = [m for m in model.backbone.sam_mask_decoder.modules() if isinstance(m, BayesianLoRALinear)]
    if len(layers) != 32:
        raise ValueError('Expected 32 Gaussian decoder adapters')
    with torch.no_grad():
        for layer in layers:
            layer.log_sigma_A.fill_(math.log(.05))
            layer.log_sigma_B.fill_(math.log(.05))


def verify_data(dataset, reference, selected_slices, ledger):
    if sorted(dataset.case_ids) != reference['test_cases'] or len(dataset) != 460:
        raise ValueError('Expected the six archived test volumes and 460 foreground slices')
    available = {f'{cid}:{start}' for cid, start, end in dataset._clip_index if end == start + 1}
    if available != set(reference['slices']) or not set(selected_slices) <= available:
        raise ValueError('Test slice identities differ from the completed archive')
    for cid in dataset.case_ids:
        for kind, expected in reference['test_file_sha256'][cid].items():
            ledger.check()
            if digest_file(dataset.pairs[cid][kind]) != expected:
                raise ValueError(f'BTCV {cid} {kind} differs from the completed archive')


def arm_records(output, arm, slice_keys, prompts, identity):
    records = {}
    directory = output / 'records' / arm
    for path in directory.glob('*.json'):
        saved = read_json(path)
        slice_key = saved['slice']
        if (path.stem != slice_key.replace(':', '_') or slice_key not in slice_keys
                or saved['identity'] != identity
                or {r['key'] for r in saved['rows']} != set(prompts[slice_key])
                or len(saved['rows']) != len(prompts[slice_key])):
            raise ValueError(f'Invalid saved progress at {path}')
        for r in saved['rows']:
            if set(r['scores']) != set(ARMS[arm][1]):
                raise ValueError(f'Variant coverage mismatch at {path}')
            if 'array_file' in r:
                arr = output / r['array_file']
                if not arr.is_file() or digest_file(arr) != r['array_sha256']:
                    raise ValueError(f'Replay array checksum mismatch at {arr}')
        records[slice_key] = saved
    return records


def infer_arm(arm, cfg, dataset, selected, saved, weights, reference, output, identity, ledger):
    import torch
    from full_worker import make_model
    from inference_common import save_arrays, fingerprint

    method, variants = ARMS[arm]
    ledger.check()
    model = make_model(cfg, method, weights.get(method))
    model.eval()
    if arm == 'gaussian_fixed_sd':
        change_sd(model)
    before = fingerprint(model)
    decoder = model.backbone.sam_mask_decoder
    policy = {k: getattr(decoder, k, None) for k in (
        'dynamic_multimask_via_stability', 'dynamic_multimask_stability_delta',
        'dynamic_multimask_stability_thresh')}
    example_keys = {row['key'] for row in reference['failure_examples']}
    variants = set(variants)
    written = 0
    with torch.no_grad():
        for index, (cid, z, _) in enumerate(dataset._clip_index):
            slice_key = f'{cid}:{z}'
            if slice_key not in selected or slice_key in saved:
                continue
            ledger.check()
            sample = dataset[index]
            if (sample.case_id, sample.slice_indices[0]) != (cid, z):
                raise ValueError('Dataset slice index changed')
            img = sample.image.to('cuda:0')
            labels = sample.label.to('cuda:0')
            torch.cuda.synchronize()
            begin = time.perf_counter()
            embedding = model.encode_image(img)
            torch.cuda.synchronize()
            encoder_seconds = time.perf_counter() - begin
            rows = []
            for key in reference['slices'][slice_key]:
                if time.time() >= ledger.deadline:
                    raise BudgetStop('Work deadline reached during a slice. Completed slices are saved.')
                prompt = reference['prompts'][key]
                organ = int(key.rsplit(':', 1)[1])
                target = labels[:, organ]
                truth = target[0].cpu().numpy().astype(bool)
                if not truth[prompt[1], prompt[0]]:
                    raise ValueError(f'Recorded point is outside its target at {key}')
                torch.cuda.synchronize()
                begin = time.perf_counter()
                with trace_fallback(decoder) as trace:
                    predictions, details = decode_arm(model, arm, embedding, target, key, prompt, truth)
                torch.cuda.synchronize()
                decode_seconds = time.perf_counter() - begin
                details['fallback_counts'] = fallback_counts(arm, trace)
                if set(predictions) != variants:
                    raise ValueError(f'Incomplete decoded variants at {key}')
                row = {'key':key, 'volume':cid, 'slice_index':z, 'organ':organ, 'prompt_xy':prompt,
                       'scores':{}, 'diagnostics':details, 'decode_wall_seconds':decode_seconds}
                for variant, (p, mi) in predictions.items():
                    pp = p.float().cpu().numpy()
                    mm = mi.float().cpu().numpy() if mi is not None else None
                    row['scores'][variant] = scalar_scores(pp, truth, mm)
                    if key in example_keys and variant == 'gaussian_learned':
                        path = output / 'arrays' / (key.replace(':', '_') + '.npz')
                        row['array_file'] = str(path.relative_to(output))
                        row['array_sha256'] = save_arrays(path, sample.image[0, 0].numpy(), truth, pp, mm,
                            {'key':key, 'prompt_xy':prompt, 'identity':identity, 'variant':variant})
                rows.append(row)
                for module in model.modules():
                    if hasattr(module, 'last_adapter_output'):
                        module.last_adapter_output = None
            path = output / 'records' / arm / (slice_key.replace(':', '_') + '.json')
            atomic_json(path, {'slice':slice_key, 'identity':identity, 'rows':rows, 'session_id':ledger.sid,
                               'encoder_seconds':encoder_seconds, 'decoder_fallback':policy})
            written += 1
            if written % 25 == 0:
                print(f'{arm}: {len(saved) + written}/{len(selected)} slices', flush=True)
            del embedding, img, labels
    if fingerprint(model) != before:
        raise RuntimeError('Adapter tensors changed during inference')
    del model
    gc.collect()
    torch.cuda.empty_cache()


def auc(labels, scores):
    from scipy.stats import rankdata

    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    finite = np.isfinite(scores)
    labels, scores = labels[finite], scores[finite]
    pos, neg = int(labels.sum()), int((~labels).sum())
    if not pos or not neg:
        return None
    return float((rankdata(scores)[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def mean_defined(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def by_volume(rows, metric, labels=None):
    """Give volumes equal weight and keep each variant's outcome explicit."""
    by_vol = defaultdict(list)
    for row in rows:
        by_vol[row['volume']].append(row)
    return {vol: auc(
        [(r['dice'] if labels is None else labels[r['key']]) < .5 for r in entries],
        [r[metric] for r in entries])
        for vol, entries in sorted(by_vol.items())}


def variant_summary(rows):
    by_pair = defaultdict(list)
    for r in rows:
        by_pair[(r['volume'], r['organ'])].append(r['dice'])
    volumes = defaultdict(list)
    for (volume, _), values in by_pair.items():
        volumes[volume].append(float(np.mean(values)))
    metrics = ('mean_entropy_full_nats', 'mean_mi_full_nats', 'mean_mi_predicted_region_nats')
    scores = {metric: by_volume(rows, metric) for metric in metrics}
    return {
        'targets':len(rows), 'volumes':len(volumes), 'volume_organ_pairs':len(by_pair),
        'pair_weighted_dice':mean_defined([float(np.mean(v)) for v in by_pair.values()]),
        'prompt_weighted_dice':mean_defined([r['dice'] for r in rows]),
        'dice_by_volume':{v:mean_defined(values) for v, values in sorted(volumes.items())},
        'failures_dice_lt_0_5':sum(r['dice'] < .5 for r in rows),
        'prompt_weighted_means_nats':{m:mean_defined([r[m] for r in rows]) for m in metrics},
        'scored_prompts_by_measure':{m:sum(r[m] is not None for r in rows) for m in metrics},
        'own_failure_auroc_by_volume':scores,
        'own_failure_mean_auroc':{m:mean_defined(list(v.values())) for m, v in scores.items()},
        'defined_auroc_volumes':{m:sum(x is not None for x in v.values()) for m, v in scores.items()},
    }


def paired(first, second):
    from scipy.stats import t

    values = {v: first[v] - second[v] for v in sorted(first.keys() & second.keys())
              if first[v] is not None and second[v] is not None}
    differences = list(values.values())
    mean = mean_defined(differences)
    interval = None
    if len(differences) >= 2:
        width = float(t.ppf(.975, len(differences)-1) * np.std(differences, ddof=1) / np.sqrt(len(differences)))
        interval = [mean - width, mean + width]
    return {'differences_by_volume':values, 'mean':mean, 'descriptive_95pct_t_interval':interval}


def report(output, reference, selected, identity, scope):
    raw_rows = defaultdict(list)
    coverage, fallback = {}, {}
    for arm, (_, variants) in ARMS.items():
        saved = arm_records(output, arm, selected, reference['slices'], identity)
        rows = [r for s in sorted(saved) for r in saved[s]['rows']]
        coverage[arm] = {'slices':len(saved), 'targets':len(rows), 'scope_complete':set(saved) == selected}
        fallback[arm] = summarize_fallback(rows)
        for row in rows:
            for variant in variants:
                raw_rows[variant].append({
                    'variant':variant, **{k:row[k] for k in ('key','volume','slice_index','organ')},
                    'x':row['prompt_xy'][0], 'y':row['prompt_xy'][1], **row['scores'][variant],
                })
    complete = scope == 'full' and all(a['scope_complete'] for a in coverage.values())
    rows = [row for values in raw_rows.values() for row in values]
    if rows:
        tmp = output / 'per_prompt.tmp'
        with tmp.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, output / 'per_prompt.csv')
    summary = {'full_cohort_complete':complete, 'scope':scope, 'identity':identity,
               'coverage':coverage, 'test_cases':reference['test_cases'],
               'single_output_fallback_by_phase':fallback,
               'failure_cutoff_dice_strictly_below':.5, 'test_prompt_seed':SEED,
               'binary_mask_threshold':.5, 'mc_draws':DRAWS, 'fixed_decoder_sd':.05,
               'refinement':'none, retain official SAM2 single-output fallback'}
    if complete:
        variants = {v:variant_summary(r) for v, r in raw_rows.items()}
        assert all(s['targets'] == 2466 and s['volume_organ_pairs'] == 78 and s['volumes'] == 6
                   for s in variants.values())
        summary['variant_results'] = variants
        summary['regional_failure_models'] = {
            v:crossfit_regions(raw_rows[v]) for v in ('gaussian_learned','gaussian_fixed_sd')
        }
        summary['paired_dice_effects'] = {
            f'{a}_minus_{b}':paired(variants[a]['dice_by_volume'], variants[b]['dice_by_volume'])
            for a,b in (
                ('frozen_multimask_iou','frozen_single'),
                ('frozen_multimask_oracle','frozen_multimask_iou'),
                ('frozen_single_or_multimask_oracle','frozen_single'),
                ('gaussian_fixed_sd','gaussian_learned'),
                ('decoder_mc_dropout','deterministic_single'),
            )
        }
        replay_errors = {}
        for variant, original in [('frozen_single','frozen'), ('deterministic_single','deterministic'),
                                  ('gaussian_learned','balora')]:
            differences = np.abs([row['dice'] - reference['archived_dice'][original][row['key']]
                                  for row in raw_rows[variant]])
            replay_errors[variant] = {'mean_abs_dice_delta':float(differences.mean()),
                                      'max_abs_dice_delta':float(differences.max()),
                                      'n_dice_deltas_above_0_01':int((differences > .01).sum())}
        summary['archived_replay_drift'] = replay_errors
        for reference_variant, alternative in [('deterministic_single','decoder_mc_dropout'),
                                                ('gaussian_learned','gaussian_fixed_sd')]:
            fixed = {r['key']:r['dice'] for r in raw_rows[reference_variant]}
            scores = by_volume(raw_rows[alternative], 'mean_mi_full_nats', labels=fixed)
            comparator = ('mean_entropy_full_nats' if reference_variant == 'deterministic_single'
                          else 'mean_mi_full_nats')
            other = by_volume(raw_rows[reference_variant], comparator, labels=fixed)
            summary[f'{alternative}_fixed_outcome_comparison'] = {
                'outcomes_from':reference_variant, 'alternative_mi_auroc_by_volume':scores,
                'reference_score':comparator, 'reference_auroc_by_volume':other,
                'paired_auroc_increment':paired(scores, other)}
    return summary


def example_figure(output, reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    entries = reference['failure_examples']
    paths = [output / 'arrays' / (e['key'].replace(':', '_') + '.npz') for e in entries]
    if not all(p.is_file() for p in paths):
        return
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.3), layout='constrained')
    for i, (entry, path) in enumerate(zip(entries, paths)):
        with np.load(path, allow_pickle=False) as arrays:
            image = arrays['windowed_CT'].astype(np.float32)
            gt = arrays['ground_truth']
            pred = arrays['prediction_FP32_threshold']
            mi = arrays['epistemic_MI_FP16'].astype(float)
            ident = json.loads(arrays['identity_json'].item())
            scores = scalar_scores(arrays['mean_probability_FP16'].astype(float), gt.astype(bool), mi)
        x,y = ident['prompt_xy']
        yy, xx = np.nonzero(gt | pred)
        radius = max(96, (max(np.ptp(xx), np.ptp(yy)) + 48) / 2)
        center_x, center_y = (xx.min() + xx.max()) / 2, (yy.min() + yy.max()) / 2
        for ax in axes[i]:
            ax.imshow(image, cmap='gray', vmin=0, vmax=1)
            ax.contour(gt, levels=[.5], colors=['#60e2aa'], linewidths=.8)
            ax.set(xlim=(max(0, center_x-radius), min(1024, center_x+radius)),
                   ylim=(min(1024, center_y+radius), max(0, center_y-radius)))
            ax.set_axis_off()
        axes[i,0].plot(x,y,'+',color='#ffdb62',ms=7)
        axes[i,1].contour(pred, levels=[.5], colors=['#ffa24c'], linewidths=.9)
        im = axes[i,2].imshow(np.ma.masked_where(mi < .005, mi), cmap='magma', vmin=0, vmax=.15, alpha=.9)
        axes[i,0].set_title(f'{entry["key"]} | archived MI percentile {entry["within_volume_mi_percentile"]:.1%}', size=8)
        axes[i,1].set_title(f'Gaussian mask | replay Dice {dice_value(pred, gt.astype(bool)):.3f}', size=8)
        axes[i,2].set_title(f'Replay full-image mean MI {scores["mean_mi_full_nats"]:.5f} nats', size=8)
    fig.colorbar(im, ax=axes[:,2].tolist(), shrink=.60, label='Pixel MI (nats)')
    fig.savefig(output / 'replay_pancreas_failures.png', dpi=220)
    plt.close(fig)


def export(output, destination, ledger, status):
    """Finish progress ZIP after recording the bulk of export in the ledger."""
    temporary = destination.with_suffix('.tmp')
    try:
        with ZipFile(temporary, 'w', compression=ZIP_DEFLATED, compresslevel=3) as archive:
            for path in sorted(output.rglob('*')):
                if path.is_file() and path != ledger.path:
                    archive.write(path, path.relative_to(output))
            ledger.close(status)
            archive.write(ledger.path, ledger.path.relative_to(output))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def run(cfg):
    import fcntl
    from importlib.metadata import version
    from auditlib.setup_inputs import setup_sam2

    started = time.time()
    reference = read_json(ROOT / 'reference.json')
    output, work = Path(cfg['output_root']), Path(cfg['work_root'])
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    destination = output.parent / (output.name + '_Results.zip')
    scope = cfg['scope']
    if scope not in ('full','smoke'):
        raise ValueError("SCOPE must be 'full' or 'smoke'")
    selected = set(reference['slices'] if scope == 'full' else list(reference['slices'])[:2])
    input_root = Path(cfg['input_root'])
    if not input_root.is_dir():
        raise FileNotFoundError(f'Kaggle input directory is missing: {input_root}')
    with (work / (output.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if cfg['resume_path'] and not (output / 'provenance').is_dir():
            safe_extract(cfg['resume_path'], output)
        base_ledger = output / 'provenance' / 'full_run_ledger_carried_round3.json'
        atomic_json(base_ledger, reference['carried_ledger'])
        ledger = Ledger(output, input_root, '', cfg, started=started)
        state = {'full_cohort_complete':False,'scope':scope, 'status':'STARTED'}
        identity = None
        error = None
        try:
            print(f'Carried {ledger.prior_hours:.4f} pipeline hours; cap {ledger.limit:g} h', flush=True)
            ledger.check()
            import replay_risk
            replay_risk.pd.DataFrame({'group':[0], 'value':[1]}).groupby('group').apply(
                lambda g: g.value.mean(), include_groups=False)
            weights = checkpoint_paths(cfg, work / 'weights', reference)
            environment = setup_sam2(work / 'sam2', input_root, cfg['backbone_path'], deadline=ledger.deadline)
            environment['versions'].update({name:version(name) for name in ('pandas','scikit-learn')})
            atomic_json(output / 'provenance' / f'environment_{ledger.sid}.json', environment)
            sys.path.insert(0, str(ROOT / 'source'))
            from data_root import resolve_btcv_root
            from full_worker import loaders
            import torch
            torch.set_num_threads(2)

            root, _ = resolve_btcv_root(input_root, cfg['btcv_root'])
            configs = {m:get_model_cfg(reference, m, root, work, environment) for m, _ in ARMS.values()}
            dataset = loaders(configs['frozen'], 'test').dataset
            verify_data(dataset, reference, selected, ledger)
            run_identity = {'scope':scope, 'source_sha256':cfg['payload_sha256'],
                            'archive_sha256':reference['archive_sha256'],
                            'weight_sha256':reference['selected_weights'], 'backbone_sha256':environment['backbone_sha256'],
                            'test_data_sha256':reference['test_file_sha256'], 'sam2_commit':environment['official_commit'],
                            'gpu':environment['gpu'], 'cuda':environment['cuda'], 'versions':environment['versions']}
            identity = digest_json(run_identity)
            ident_path = output / 'provenance' / 'identity.json'
            if ident_path.is_file() and read_json(ident_path) != run_identity:
                raise ValueError('This output contains a different replay scope, source, or GPU runtime. Use its original runtime or a fresh output directory.')
            atomic_json(ident_path, run_identity)
            for arm, (method, _) in ARMS.items():
                saved = arm_records(output, arm, selected, reference['slices'], identity)
                if set(saved) != selected:
                    print(f'Running {arm} with {len(saved)}/{len(selected)} slices saved', flush=True)
                    infer_arm(arm, configs[method], dataset, selected, saved, weights, reference, output, identity, ledger)
            timing_path = output / 'encoder_timing.json'
            if not timing_path.is_file():
                timings = encoder_timing(configs, weights, dataset, selected, ledger)
                atomic_json(timing_path, {'identity':identity, **timings})
            if read_json(timing_path)['identity'] != identity:
                raise ValueError('Encoder timing belongs to a different replay')
            state.update(status='COMPLETE' if scope == 'full' else 'SMOKE_COMPLETE')
        except BudgetStop as exc:
            state.update(status='PARTIAL_BUDGET_PAUSED', message=str(exc))
        except KeyboardInterrupt:
            state.update(status='PARTIAL_INTERRUPTED')
        except Exception as exc:
            error = exc
            state.update(status='FAILED', message=str(exc))
            (output / 'last_error.txt').write_text(traceback.format_exc())
        finally:
            if identity is not None:
                try:
                    state.update(report(output, reference, selected, identity, scope))
                    example_figure(output, reference)
                except Exception as exc:
                    state.update(status='FAILED', message=f'Could not summarize saved records: {exc}')
                    if error is None:
                        error = exc
            if state['status'] == 'COMPLETE' and not state['full_cohort_complete']:
                state.update(status='PARTIAL_INCOMPLETE_COVERAGE')
            if state['status'] != 'COMPLETE':
                state['full_cohort_complete'] = False
            state['accounted_hours_before_audit_session'] = ledger.prior_hours
            atomic_json(output / 'status.json', state)
            export(output, destination, ledger, state['status'])
            print(f'{state["status"]}: {destination}', flush=True)
        if error is not None:
            raise error


if __name__ == '__main__':
    run(read_json(sys.argv[1]))
