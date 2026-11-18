# BALoRA result verification

Use Python 3.12. Run these commands from the directory containing this README.

```bash
python -m pip install -r requirements.txt
python analysis/reanalyze.py analysis/data/per_prompt_fold1_seed42.csv reproduced
python analysis/dropout_increment.py archives/BALoRA_Round3_Inference_Results.zip reproduced/dropout_increment_by_volume.csv
```

`analysis/data/` holds the original seed-42 per-prompt CSV, split manifest, validation histories, and derived tables. `archives/BALoRA_Round3_Inference_Results.zip` contains the completed replay with 19,728 variant-prompt rows. The low-Dice failure threshold is 0.5. The dropout increment uses the original deterministic failure outcomes and compares rankings on those fixed outcomes.

To verify archive and checkpoint hashes, exact prompt matching, and the replay against the training archive, download the [completed Fold 1 results archive](https://chatgpt.com/library?file_id=libfile_dea0608e7b888191a510ac2f8cccc85e&account_id=personal). Put `BALoRA_Full_Run_Results(1).zip` in `archives/`, then run:

```bash
python analysis/audit_archive.py "archives/BALoRA_Full_Run_Results(1).zip" analysis/data/per_prompt_fold1_seed42.csv reproduced/archive
python analysis/verify_round3.py "archives/BALoRA_Full_Run_Results(1).zip" archives/BALoRA_Round3_Inference_Results.zip reproduced/replay
```

For the boundary analysis, copy the bundled `analysis/data` directory to `reproduced/spatial`. This script reads the saved CSVs from that copy and writes the regenerated boundary tables there:

```bash
python analysis/spatial_sensitivity.py "archives/BALoRA_Full_Run_Results(1).zip" reproduced/spatial
```

For the paper figure generator, copy `analysis/data` into a new workspace at `<workspace>/analysis/data`, then pass that workspace as the output root:

```bash
python analysis/make_figures.py "archives/BALoRA_Full_Run_Results(1).zip" "<workspace>"
```

To rebuild the replay notebook, place the [research source archive](https://chatgpt.com/library?file_id=libfile_ffeb1976fad881919ff7a79e9339eab0&account_id=personal) in `archives/` alongside the completed Fold 1 results archive. The bundled ledger is the exact record embedded in the verified replay notebook.

```bash
python inference/build_notebook.py --completed "archives/BALoRA_Full_Run_Results(1).zip" --ledger archives/round3_carried_ledger.json --original-source archives/BALoRA_Full_Research_Package.zip --output BALoRA_Round3_Rebuilt.ipynb
```

`visualizations/` contains the organ-wise Dice chart, failure ROC and precision–recall curves, regional calibration summary, and CSVs. The pixel-level confusion and reliability plotting script requires full matched masks and foreground probabilities. See [the plot input format](visualizations/README.txt).
