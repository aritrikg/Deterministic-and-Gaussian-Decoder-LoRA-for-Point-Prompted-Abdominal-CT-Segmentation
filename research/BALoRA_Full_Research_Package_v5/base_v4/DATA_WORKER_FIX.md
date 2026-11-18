# Labeled BTCV root selection repair v4.2

The uploaded log confirms that scanning the whole input tree included img0061 through img0080, which have no labels. These extra images are not members of the declared 30-volume labeled cohort.

This revision automatically locates one complete labeled Training folder before installing/loading SAM2. It verifies all 30 expected case IDs and every image/label pair, then uses that same root for data auditing, profiling, training and evaluation. Missing masks within the labeled cohort still fail. Multiple complete dataset copies require an explicit btcv_root. The 19/5/6 split, all foreground slices and image resolution are unchanged.

To verify the chosen folder after running, inspect:

```python
from pathlib import Path
root = Path('/kaggle/working/balora_full_run_fold1_seed42_v4')
for rel in ['resolved_btcv_root.json', 'data_status.json']:
    p = root / rel
    print('\n' + str(p))
    print(p.read_text(errors='replace')[-16000:] if p.exists() else 'File not found')
```

Keep results ZIPs from all failed attempts attached so their ledgers merge without losing spent time. Source migration is allowed only before fitted weights, test records or completed profiling exist. The corrected selection and data worker passed a synthetic 30-labeled-plus-20-unlabeled NIfTI subprocess test, preserving 1347/371/460 foreground slices and all 2466 test identities. This is engineering validation, not a completed GPU experiment.
