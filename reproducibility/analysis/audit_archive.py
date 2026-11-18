"""Verify the completed archive and summarize retained posterior tensors.

Reads the original ZIP and selected checkpoint without loading executable model
code. Only the three pickle globals in this known adapter checkpoint are
accepted. All manifest entries are checked before using the records.
"""
from collections import Counter, OrderedDict
from pathlib import Path
from zipfile import ZipFile
import argparse
import hashlib
import io
import json
import pickle

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation


ARCHIVE_HASH = 'ccc8581ea2b5dfac4909056585448a9121f6e1d2a122352f6039d5680c185340'
WEIGHT_HASHES = {
    'deterministic': '40ff778fe706fda8ebac4ad9b5bd5215a2c1e6b89674f654016b22a54dff70d8',
    'balora': 'cb0d82ca0b01cb4328ef6afa1c54e246c42093c3320efd519b931063398da0ec',
}


class FloatStorage:
    pass


def tensor_descriptor(storage, offset, shape, stride, *unused):
    return storage, offset, shape, stride


class CheckpointMetadataReader(pickle.Unpickler):
    allowed = {
        ('collections','OrderedDict'): OrderedDict,
        ('torch','FloatStorage'): FloatStorage,
        ('torch._utils','_rebuild_tensor_v2'): tensor_descriptor,
    }

    def find_class(self, module, name):
        if (module,name) not in self.allowed:
            raise pickle.UnpicklingError(f'Unexpected global {module}.{name}')
        return self.allowed[(module,name)]

    def persistent_load(self, identifier):
        kind, storage_type, key, location, size = identifier
        if (kind, storage_type, location) != ('storage', FloatStorage, 'cpu') or not str(key).isdigit():
            raise pickle.UnpicklingError(f'Unexpected tensor storage {identifier}')
        return str(key), size


def read_tensor(archive, descriptor):
    (storage_id, size), offset, shape, stride = descriptor
    values = np.frombuffer(archive.read(f'best/data/{storage_id}'), dtype='<f4')
    assert len(values) == size
    end = offset + int(np.prod(shape))
    expected = tuple(int(np.prod(shape[i+1:])) for i in range(len(shape)))
    assert end <= len(values) and tuple(stride) == expected
    return values[offset:end].astype(float)


def checkpoint_stats(raw):
    with ZipFile(io.BytesIO(raw)) as z:
        assert z.read('best/byteorder') == b'little'
        state = CheckpointMetadataReader(io.BytesIO(z.read('best/data.pkl'))).load()
        tensors=state['model_state_dict']
        rows=[]
        sigma_arrays=[]
        for k, descriptor in tensors.items():
            if not k.endswith(('log_sigma_A','log_sigma_B')):
                continue
            layer, parameter = k.rsplit('.',1)
            factor = parameter[-1]
            mu_descriptor = tensors[f'{layer}.mu_{factor}']
            assert descriptor[2] == mu_descriptor[2]
            mu = read_tensor(z, mu_descriptor)
            sigma = np.exp(np.clip(read_tensor(z, descriptor), -12, 6))
            sigma_arrays.append(sigma)
            q = np.quantile(sigma,[0,.05,.5,.95,1])
            mu_rms = np.sqrt(np.mean(mu**2))
            sigma_rms = np.sqrt(np.mean(sigma**2))
            rows.append({
                'layer':layer, 'factor':factor, 'count':len(sigma), 'initial_sigma':.05,
                'mean_sigma':float(sigma.mean()), 'min':q[0], 'p05':q[1],
                'median':q[2], 'p95':q[3], 'max':q[4],
                'mean_abs_mu':np.abs(mu).mean(), 'rms_mu':mu_rms, 'rms_sigma':sigma_rms,
                'rms_sigma_over_mu':sigma_rms/mu_rms,
                'kl_to_unit_gaussian_nats':float((.5*(mu**2+sigma**2-1)-np.log(sigma)).sum()),
            })
        assert len(rows)==64
        factors = pd.DataFrame(rows)
        norms = factors.assign(mu2=factors['count'] * factors.rms_mu**2,
                               sigma2=factors['count'] * factors.rms_sigma**2)
        norms = norms.groupby('layer', as_index=False)[['count','mu2','sigma2']].sum()
        assert len(norms)==32
        norms['rms_mu'] = np.sqrt(norms.mu2 / norms['count'])
        norms['rms_sigma'] = np.sqrt(norms.sigma2 / norms['count'])
        norms['rms_sigma_over_mu'] = norms.rms_sigma / norms.rms_mu
        norms = norms.drop(columns=['mu2','sigma2'])
        sig=np.concatenate(sigma_arrays)
        s={
            'stored_epoch':state['epoch'],
            'learned_scale_count':int(len(sig)),
            'global_sigma_p05_median_p95':np.quantile(sig,[.05,.5,.95]).tolist(),
            'factor_mean_sigma_range':[min(r['mean_sigma'] for r in rows),max(r['mean_sigma'] for r in rows)],
            'sigma_median_abs_deviation_from_initial':float(np.median(np.abs(sig-.05))),
            'factor_rms_ratio_by_factor':{
                factor: np.quantile(group.rms_sigma_over_mu,[0,.5,1]).tolist()
                for factor,group in factors.groupby('factor')
            },
            'pooled_layer_rms_ratio_min_median_max':np.quantile(norms.rms_sigma_over_mu,[0,.5,1]).tolist(),
            'raw_checkpoint_kl_nats':float(factors.kl_to_unit_gaussian_nats.sum()),
            'weighted_checkpoint_kl':float(factors.kl_to_unit_gaussian_nats.sum()*1e-4/2391),
        }
        return factors,norms,s


def ece(prob, y, keep, bins=15):
    p = prob[keep].astype(float)
    y = y[keep]
    k = np.minimum((bins*p).astype(int),bins-1)
    gaps = np.bincount(k,weights=p-y,minlength=bins)
    return float(np.sum(np.abs(gaps))/len(p))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('archive',type=Path)
    p.add_argument('records',type=Path)
    p.add_argument('out',type=Path)
    args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    digest=hashlib.file_digest(args.archive.open('rb'),'sha256').hexdigest()
    assert digest==ARCHIVE_HASH
    df=pd.read_csv(args.records).set_index(['method','key'])
    methods=['frozen','deterministic','balora']
    with ZipFile(args.archive) as z:
        root=z.namelist()[0].split('/')[0]+'/'
        read=lambda p: json.loads(z.read(root+p))
        manifest=read('SHA256_manifest.json')
        for name, expected in manifest.items():
            with z.open(root+name) as f:
                assert hashlib.file_digest(f,'sha256').hexdigest()==expected, name
        state=read('run_status.json')
        assert state['full_new_protocol_complete']
        data=read('data_audit.json')
        split={key: read(f'splits/{key}.json') for key in ['train','validation','test']}
        (args.out/'verified_split_manifests.json').write_text(json.dumps(split,indent=2)+'\n')
        observed={}
        val_history={}
        weight_digests={}
        for m in methods:
            records={}
            for n in z.namelist():
                if not n.startswith(root+f'{m}/test/records/') or not n.endswith('.json'):
                    continue
                rec=json.loads(z.read(n))
                k=rec['key']
                assert k not in records
                assert abs(rec['dice']-df.at[(m,k),'dice'])<1e-7
                assert abs(rec['decode_seconds']-df.at[(m,k),'dec_s'])<1e-9
                assert abs(rec['encoder_seconds_for_slice']-df.at[(m,k),'enc_s'])<1e-9
                assert rec['empty_prediction']==bool(df.at[(m,k),'empty'])
                records[k]=rec['prompt_xy']
            assert len(records)==2466 and set(records)==set(data['expected_test_keys'])
            observed[m]=records
            if m!='frozen':
                raw=z.read(root+f'{m}/research/best.pt')
                weight_digests[m]=hashlib.sha256(raw).hexdigest()
                assert weight_digests[m]==WEIGHT_HASHES[m]
                h=read(f'{m}/research/training_history.json')
                if isinstance(h,list):
                    h={k:[i[k] for i in h] for k in ('epoch','val_dice')}
                val_history[m]={'epoch':[i+1 for i in h['epoch']],'val_dice':h['val_dice']}
                assert len(h['epoch'])==13 and np.argmax(h['val_dice'])==6
                if m=='balora':
                    factors,layers,sigma=checkpoint_stats(raw)
                    factors.to_csv(args.out/'posterior_scales_by_layer.csv',index=False)
                    layers.to_csv(args.out/'posterior_scale_to_mean_by_layer.csv',index=False)
        assert observed['frozen']==observed['deterministic']==observed['balora']
        (args.out/'validation_histories.json').write_text(json.dumps(val_history,indent=2)+'\n')
        audit=read('validation_refinement_audit/validation_comparison.json')
        (args.out/'validation_refinement_comparison.json').write_text(json.dumps(audit,indent=2)+'\n')

        # Optional descriptive region check on the 18 pre-selected image targets.
        # Use a common dilated union across all methods for each target, so the
        # methods are evaluated on the same pixels. Scores use saved FP16 p.
        roi_rows=[]
        available_panels=[]
        for key in data['qualitative_keys']:
            file=key.replace(':','_')+'.npz'
            values={}
            for m in methods:
                path=root+f'{m}/test/arrays/{file}'
                assert path in z.namelist()
                with np.load(io.BytesIO(z.read(path)),allow_pickle=False) as a:
                    values[m]=(a['ground_truth'].astype(bool),a['prediction_FP32_threshold'].astype(bool),
                               a['mean_probability_FP16'])
            gt=values['balora'][0]
            union=gt.copy()
            for m,(y,pred,prob) in values.items():
                assert np.array_equal(y,gt)
                union |= pred
                row=df.loc[(m,key)]
                assert pred.sum()==round(2*round(row.rec*row.gt_area)+row.pix_err-row.gt_area)
            region=binary_dilation(union,iterations=2)
            for m,(_,pred,prob) in values.items():
                roi_rows.append({'key':key,'volume':key.split(':')[0],'method':m,
                                 'union_pixels':int(region.sum()),'ece_15':ece(prob,gt,region),
                                 'brier':float(np.mean((prob[region].astype(float)-gt[region])**2))})
            row=df.loc[('balora',key)]
            available_panels.append({'key':key,'dice':float(row.dice),'mi':float(row.mi)})
        rois=pd.DataFrame(roi_rows)
        rois.to_csv(args.out/'selected18_union_region_metrics.csv',index=False)
        study={
            'verified_completed_archive_sha256':digest,
            'manifest_entries_verified':len(manifest),
            'record_counts':{m:len(observed[m]) for m in methods},
            'matched_prompt_keys_and_coordinates':True,
            'selected_checkpoint_sha256':weight_digests,
            'qualitative_array_records':len(roi_rows),
            'posterior_scale_summary':sigma,
            'selected18_roi_descriptive_means':rois.groupby(['method','volume'])[['ece_15','brier']].mean().groupby('method').mean().to_dict('index'),
            'qualitative_targets':available_panels,
            'training_export_fields':{m:list(read(f'{m}/research/training_history.json')[0]) if isinstance(read(f'{m}/research/training_history.json'),list) else list(read(f'{m}/research/training_history.json')) for m in methods[1:]},
        }
        (args.out/'archive_audit.json').write_text(json.dumps(study,indent=2)+'\n')


if __name__=='__main__':
    main()
