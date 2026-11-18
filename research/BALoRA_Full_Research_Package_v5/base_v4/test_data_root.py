from pathlib import Path
import pytest
from data_root import resolve_btcv_root
from protocol import split_ids

def layout(base):
    root=base/'RawData/Training'
    (root/'img').mkdir(parents=True);(root/'label').mkdir()
    for cid in sum(split_ids().values(),[]):
        (root/'img'/(cid+'.nii')).touch()
        (root/'label'/('label'+cid[3:]+'.nii')).touch()
    challenge=base/'RawData/Test/img';challenge.mkdir(parents=True)
    for number in range(61,81):(challenge/f'img{number:04d}.nii').touch()
    return root

def test_labeled_training_is_selected_with_unlabeled_challenge_cases(tmp_path):
    root=layout(tmp_path/'btcv')
    selected,report=resolve_btcv_root(tmp_path)
    assert Path(selected)==root and report['labeled_cases']==30
    assert report['strict_pairing_retained'] and any('img0061' in r['error'] for r in report['candidate_errors'])

def test_missing_mask_inside_labeled_training_still_fails(tmp_path):
    root=layout(tmp_path/'btcv');(root/'label/label0001.nii').unlink()
    with pytest.raises(ValueError,match='No complete labeled'):resolve_btcv_root(tmp_path)

def test_duplicate_complete_copies_require_explicit_choice(tmp_path):
    a=layout(tmp_path/'a');layout(tmp_path/'b')
    with pytest.raises(ValueError,match='Multiple complete'):resolve_btcv_root(tmp_path)
    selected,_=resolve_btcv_root(tmp_path,str(a));assert Path(selected)==a

def test_no_extra_case_silently_removed_from_training(tmp_path):
    root=layout(tmp_path/'btcv');(root/'img/img0001_duplicate.nii').touch()
    with pytest.raises(ValueError,match='No complete labeled'):resolve_btcv_root(tmp_path)
