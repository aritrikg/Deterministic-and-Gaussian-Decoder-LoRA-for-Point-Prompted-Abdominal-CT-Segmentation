BALoRA — DATA-BACKED ADDITIONAL EVALUATION

FILES
  figures/01_organ_wise_dice.png        13-organ Dice (3 original models)
  figures/02_failure_roc.png            ROC using all-pixel mean uncertainty
  figures/03_failure_pr.png             PR using all-pixel mean uncertainty
  figures/04_calibration_ece_by_region.png  Actual calibration metrics (NOT a reliability diagram)
  data/                              Archived CSV evidence and derived curve CSVs
  generate_missing_pixel_diagnostics.py  Exact pixel confusion/reliability plots, when full test artifacts exist
  DATA_PROVENANCE_AND_LIMITATIONS.txt   Sources and methodological warnings

ORIGINAL EVIDENCE
  BALoRA_Strengthening_v5_Results (3).zip → analysis/per_organ.csv,
  analysis/uncertainty_per_target.csv, analysis/regional_calibration.csv.
  Dataset: BTCV, Fold 1, original model comparison, 6 test volumes, 13 organs,
  2,466 organ-target prompts per model. Confusion and calibration reliability cannot
  be exactly recovered from archived mean scores.

TO FINISH MISSING PIXEL PLOTS
  Export exact test-time output from the SAME trained versions and matched prompts:
  inputs/frozen/*.npz, inputs/deterministic/*.npz, inputs/balora/*.npz
  Each file must contain same-shaped arrays:
    pred_mask       — final binary predicted mask (0/1), including postprocessing
    gt_mask         — binary ground-truth mask (0/1)
    probability_map — actual foreground probabilities (float 0–1)
  Ideally use matched names (volume_slice_organ.npz) for all three models.
  Run:
    python generate_missing_pixel_diagnostics.py inputs output_pixel_figures
  The resulting matrices count all image pixels. Reliability bins use 15 equal-width
  intervals and must be reported as all-pixel; note background class imbalance.

INTERPRETATION
  Per-model low-Dice failure uses Dice < 0.5; labels differ across methods.
  AP varies strongly with failure prevalence. Do not present AUC values as a
  controlled comparison on fixed labels, or imply that all-pixel ECE establishes
  clinically reliable organ-boundary probabilities.
