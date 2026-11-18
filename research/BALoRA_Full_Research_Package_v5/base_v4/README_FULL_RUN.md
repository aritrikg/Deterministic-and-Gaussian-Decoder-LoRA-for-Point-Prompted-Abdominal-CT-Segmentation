# Full BALoRA research run on Kaggle

This notebook executes the complete **new Fold 1 single-seed pipeline** from raw BTCV. It does not start with saved scores or require previous adapted checkpoints.

## Run

1. Import `BALoRA_Full_Research_Kaggle.ipynb` into Kaggle.
2. Attach the original BTCV raw NIfTI image and label dataset, with all 30 cases. No old results ZIP is required for fresh training.
3. Enable **GPU T4 x2** and **Internet**. One CUDA GPU also works sequentially, with a different time forecast.
4. Keep `seed=42` for the original single-fold recipe. Set `btcv_root` only if more than one dataset copy is attached. The root should contain only the intended 30 paired cases.
5. Review the cumulative allowance in the first configuration cell, then select **Save Version → Save & Run All**.

The stages execute automatically: pinned SAM2 setup, full data/split audit, disposable real-backbone profiles, fresh deterministic and BALoRA training, validation checkpoint selection, a separate validation refinement audit, full paired tests of frozen SAM2 and both newly fitted models, uncertainty diagnostics, all tables and figures, and checkpoint export.

Training and disposable profiles may run on two distinct GPUs in separate processes. Test inference runs sequentially on one GPU so measured inference boundaries match. No distributed-training or doubled training-batch claim is made.

## Budget means cumulative

The original study already accounts for **20.884743572863048 hours**. The default `total_cumulative_hours=30.0` leaves approximately 9.1153 hard hours, or 8.6153 working hours before new setup, profiles and export. The notebook carries all those hours forward. It does not reset the original budget or assume the previous 13/17-epoch stopping points will recur.

The code **actually starts the training stages** while time is available. An insufficient full-run forecast is recorded; it does not silently shrink epochs, slices, resolution or posterior count. If a session or cumulative deadline arrives, the run saves partial state and remains incomplete. A complete fresh rerun cannot be guaranteed inside the original remaining allowance.

To allocate a separately expanded reproduction programme, explicitly increase `total_cumulative_hours` in the configuration before running or resuming. Keep all old ledgers attached. The ledger retains the original 30-hour cap and records the expanded cap separately; it never relabels an over-30-hour cumulative study as under 30 hours. The measured profile writes a forecast before fresh fitting. Any expansion is a user decision, not an automatic default or an assurance of completion.

Kaggle documents a **12-hour CPU/GPU session limit**. This notebook uses at most 10.5 requested work hours per session, additionally bounded by the remaining cumulative allowance, with a 180-second bulk-export margin and a separate 0.5-hour cumulative safety reserve. A longer complete run needs more than one Kaggle session. Runtime forecasts, dependency checks and passing tests are not an error-free GPU execution guarantee.

## Resume

The output `/kaggle/working/BALoRA_Full_Run_Results.zip` includes **best.pt and last.pt for both new fits**, optimizer, scheduler, AMP scaler, Python/NumPy/Torch RNG, the partial-epoch cursor, validation state, individual test records, hashes and the ledger. Raw decompression caches and the pretrained backbone are excluded.

Attach the previous v4 results ZIP as a new Kaggle input and run the same notebook again. It auto-discovers one matching `study_id`. If several matches exist, set `resume_root` to the intended latest ZIP or extracted output folder. Keep previous follow-up ledgers attached so none of the experimental time is lost. A completed training or test stage is skipped; unfinished work resumes. Changed raw data, source, seed or incompatible evaluation identity is rejected.

If a host timeout killed the session before its ledger closed, use `reconciliations` to supply the exact session ID, **full elapsed host duration** and supporting log description. A heartbeat is only a lower bound. Do not guess the missing tail, decrease usage or reset accounting.

## Completion

Only `FULL_NEW_FOLD1_PROTOCOL_COMPLETE` with `full_new_protocol_complete=true` means this new pipeline completed. A completed notebook cell, saved ZIP or partial training history is insufficient. Historical completion of the original three-method comparison is recorded separately and never makes a new fit complete.

The full run retains 19 training, five validation and six test volumes, 1024 × 1024 inputs, all organ-positive slices, Ktrain=2/Kinfer=4 for BALoRA and one prediction for point-estimate methods. Training uses at most 20 epochs, at least eight epochs and patience six. Every method uses the same saved positive point at each test identity. There are 460 foreground test slices, 2,466 predictions per method and 78 dependent volume-organ means. The six volumes are the assumed independent units.

The main comparison uses the previously retained **no-refinement policy fixed before the new fit**. The new validation audit reports none, legacy and guarded policies at the newly selected checkpoint. Its outcomes are measured afresh and may differ from the historical negative finding. The audit's rule recommendation is not silently substituted into the main test policy. All methods are reported regardless of which performs best.

## Figures and results

After full completion, the notebook generates **13 primary figure sets** as PNG, SVG and PDF, plus genuine qualitative panels for the predeclared maximum-area liver, pancreas and right-adrenal slice in each test volume. The panels show actual CT, target, three masks, the same oracle point and Bayesian MI. No synthetic patient image or deterministic epistemic MI is supplied.

Figures cover the implemented schematic, actual training and validation histories, per-organ and paired-volume overlap, primary probability scores, resized 2D surface distances, the new validation refinement audit, rank use, all paired differences, deletion sensitivity, risk–coverage, conditional reliability and matched inference measurements. Qualitative selection uses anatomy before fitting, rather than picking favorable model scores. Partial runs generate only explicitly labeled partial training curves when available.

CSV tables retain undefined denominators. Calibration and uncertainty diagnostics are distinct: all-pixel probability scores are background-sensitive; conditional regions are ground-truth-defined; histogram AUROC/AP are approximations; failure associations are descriptive within volumes. Bootstrap intervals jointly resample whole volumes and are exploratory. Six volumes and one fitted seed cannot establish multi-seed, external or clinical reliability. HD95 and ASSD remain resized **2D pixels**, not physical-space 3D millimetres.

Use the output dataset name **balora-full-run-fold1-seed42-v4**. Keep the generated reports and completion status when updating the manuscript. Old mean scores, epoch counts and plots are not copied into the new result report.

Official references for execution:

- https://www.kaggle.com/docs/notebooks
- https://github.com/facebookresearch/sam2
- https://github.com/facebookresearch/sam2/blob/main/INSTALL.md

SAM2 is pinned to commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`; the pretrained SAM2.1 Hiera Base+ file and all runtime/source identities are checked. Only missing non-core helpers are installed. The environment's CUDA-enabled PyTorch is retained. The official builder is resolved before model construction, addressing the earlier missing-package import failure path.
