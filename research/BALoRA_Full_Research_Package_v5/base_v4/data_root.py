"""Find the complete labeled 30-volume cohort without including challenge Test."""
from pathlib import Path
import importlib.util,os
from protocol import ROOT,split_ids

def resolve_btcv_root(input_root,explicit=''):
    base=Path(explicit or input_root).expanduser().resolve()
    if not base.is_dir():raise FileNotFoundError('BTCV search directory is unavailable: '+str(base))
    spec=importlib.util.spec_from_file_location('balora_v42_discovery',ROOT/'source/datasets/splits.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    expected=set(sum(split_ids().values(),[]));candidates={base}
    names={'training','img','image','images','imagestr','label','labels','labelstr'}
    for p in base.rglob('*'):
        if p.is_dir() and p.name.lower() in names:
            candidates.add(p if p.name.lower()=='training' else p.parent)
    groups={};errors=[]
    for candidate in sorted(candidates):
        try:
            pairs=module.discover_btcv_pairs(candidate)
            if set(pairs)!=expected:
                raise ValueError('Case identities differ from the declared 30-volume cohort')
            files=tuple(str(pairs[cid][kind].resolve()) for cid in sorted(pairs) for kind in ['image','label'])
            canonical=Path(os.path.commonpath(files))
            # Recheck the selected scope, preserving strict pairing rules.
            if set(module.discover_btcv_pairs(canonical))!=expected:raise ValueError('Selected scope contains additional cases')
            groups[files]=canonical
        except (ValueError,FileNotFoundError) as exc:
            errors.append({'candidate':str(candidate),'error':str(exc)})
    if len(groups)>1:
        raise ValueError('Multiple complete BTCV copies found. Set CONFIG["btcv_root"] to one labeled Training folder: '+str(sorted(str(p) for p in groups.values())))
    if not groups:
        detail='\n'.join(e['candidate']+': '+e['error'] for e in errors[:5])
        raise ValueError('No complete labeled 30-case BTCV training cohort found. Missing labels inside that cohort remain errors.\n'+detail)
    chosen=next(iter(groups.values()))
    return str(chosen),{'requested_search_root':str(base),'selected_labeled_root':str(chosen),
        'labeled_cases':30,'expected_case_ids':sorted(expected),'candidate_errors':errors,
        'scope':'All 30 labeled volumes, partitioned 19/5/6; official unlabeled challenge Test images outside this root are not study targets',
        'strict_pairing_retained':True}
