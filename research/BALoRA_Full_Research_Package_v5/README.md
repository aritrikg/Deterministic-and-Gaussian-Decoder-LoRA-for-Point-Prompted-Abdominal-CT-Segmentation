# BALoRA submission-strengthening notebook v5

This is an executable, resumable **new internal experiment programme**, not evidence that the new experiments have already run. No GPU training was performed while authoring this package. There is no defensible way to guarantee a 10/10 study or journal acceptance by changing a notebook.

## Start in Kaggle

1. Import the current `BALoRA_Full_Research_Kaggle.ipynb`. It contains a losslessly compressed, checksum-verified runtime and is built below900,000 bytes, retaining margin under the reported1 MB source cap. **No source ZIP upload or `SOURCE_PACKAGE_PATH` is needed.** The full package is optional for docs, historical notebooks and reference previews.

2. Attach the **latest completed `BALoRA_Full_Run_Results.zip`**, containing the selected fitted weights with hashes below, and the complete labeled BTCV Training cohort. Old original-comparison ZIPs and a manuscript supplement alone are insufficient. Keep Internet on for the pinned official SAM2 checkout/dependency setup. Select **GPU T4 x2**.
3. Read the visible configuration cell. Default `PROGRAMME_MODE='EXPANDED_INTERNAL'` prepares the complete 17-fit programme under a **proposed 240-hour cumulative ceiling**. This explicitly exceeds both the original 30-hour declaration and the previously expanded 45-hour ceiling. It is not a forecast or a promise that 240 hours suffice. Change the cap to your actual allocated research allowance. For profiling only under the existing allowance choose `REMAINING_BUDGET` and cap45. No quota is purchased or changed by the notebook.
4. Run All. Disposable profiles measure real training, validation and diagnostic inference before fitting. Read `measured_forecast.json`. Every scientific fit keeps 1024 resolution, all assigned volumes/foreground targets, max20/min8/patience6, validation selection and no refinement. A deadline produces PAUSED, not an invented complete result.
5. Run the bundled-source cell directly. It restores every required source file locally and verifies its bytes. It does not search attached datasets for source packages. Scientific files, original v4 source, checkpoints, budget and experiment register are unchanged by this packaging fix. Inline images default off; all PNG/PDF/SVG figures still save normally.

6. Kaggle documentation currently states a 12-hour GPU session limit. This notebook stops GPU work cooperatively at most10.5 hours from notebook start, with an additional export margin. Hardware/account limits may be lower: set `SESSION_WORK_HOURS` accordingly and inspect the account quota UI. Session limits, weekly accelerator quota and cumulative research wall-time accounting are different quantities.
7. Save/download `BALoRA_Strengthening_v5_Results.zip`. Attach the **latest** v5 results ZIP as a Kaggle input on the next session, retain the completed v4 reference results and BTCV inputs, and Run All again. Auto-resume finds one v5 archive; with multiple copies set `RESUME_PATH` explicitly. Do not start a fresh output identity to evade a ledger.

## Verified starting comparison — complete, not pending

| Current completed fresh rerun | Test Dice | Completed / selected epoch | Selected weight SHA-256 |
|---|---:|---|---|
| Frozen SAM2 |0.5123444878082426|No fitting|a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5|
| Deterministic rank16 LoRA |0.8048708566147759|13 /7|40ff778fe706fda8ebac4ad9b5bd5215a2c1e6b89674f654016b22a54dff70d8|
| BALoRA v1, no refinement |0.7981970940632518|13 /7|cb0d82ca0b01cb4328ef6afa1c54e246c42093c3320efd519b931063398da0ec|

Each method: six test volumes,460 foreground slices,2466 slice-organ targets,78 dependent volume-organ pairs. Dice averages slices within each volume-organ pair, then averages pairs equally. A different target-weighting scheme can change the ordering; preserve the primary aggregation. Deterministic adaptation currently exceeds BALoRA by0.6674 percentage points. Neither method proves clinical utility. The older, separate comparison (D0.8111821194785646, B0.7920152291014438) is also complete and must not be merged into the fresh rerun or counted as another seed.

The completed-comparison ledger recorded **35.726407254264196 hours**. The subsequently uploaded failed attempts raise the current carried total to **35.9096370237325 hours**, as documented in `UPLOADED_FAILURE_AUDIT.json`. It comprises the original20.884743572863048-hour baseline and two closed subsequent sessions. The existing45-hour allowance leaves9.273592745735804 hours before safety/export costs. All those hours are carried forward. Concurrent GPUs share a pipeline wall clock: this ledger is not GPU-hours. Any more recent uploaded sessions are merged by session ID using the largest recorded duration; duplicate ledger copies are not added twice. Unclosed sessions require the actual host-log duration, never an invented estimate from a heartbeat.

## New fits, in scientific priority order

| Block | New fits | Question | Independent sampling |
|---|---:|---|---|
| Fold1 seeds123,2026: deterministic v1 and BALoRA v1 |4|Does the direction persist across training seeds?|The same six test volumes; seed variation reported separately|
| Fresh Gaussian Bayesian fixed-rank16 |1|What changes without the v1 allocator/rank penalty?|Six volumes, one seed; Bayesian package also includes KL|
| Fresh v2 deterministic fixed16, deterministic adaptive, Bayesian adaptive |3|Does a task-trained controller alter the Bayesian×rank comparison?|Six volumes, one seed; share the fresh fixed Bayesian arm above|
| Folds2–5, seed42, deterministic v1 and BALoRA v1 |8|Internal outer-fold replication|30 disjoint outer test assignments overall; overlapping training sets|
| Custom point-conditioned U-Net from scratch |1|A conventional medical segmentation architecture under the oracle-point task|Fold1 six volumes; baseline is not a published MedSAM reproduction|

Total **17 new fits**, not17 seeds and not a full three-seed×five-fold factorial. The reference seed42 fits are reused as completed results, not retrained by default. Each new variant profiles first. The full programme requires several sessions; the actual cost cannot be certified on a CPU authoring host. Max20-epoch forecasts do not assume early stopping. Both sequential and ideal two-GPU forecasts include25% retries/export margin, setup/analysis reserve, full tests and sensitivity inference. Hardware contention, U-Net memory and new controller cost are measured on Kaggle.

Fold1 preserves the exact recorded19/5/6 partition. The legacy generic fold generator does not reproduce that test partition. Therefore the other24 cases are explicitly shuffled with Python seed9042 and partitioned into four disjoint test groups of6; inner validation uses seed10042+fold and five volumes. Saved split JSONs are authoritative. This is a new internal replication design, not a claim that the old unexecuted five-fold splits were completed or prospectively specified. Original test results were inspected before this programme.

## Preserved v1 and separate v2

`base_v4/` is byte-verified against its original manifest (source identity7a1c30bb594633075838cfb886cd77d494dd7e605ed6ab076905a690eb2584cb). `historical/v4_notebook.ipynb` preserves the previous executable notebook. No historical score is relabeled as v2.

V1 hard integer rank selection and detached attribution do not transmit segmentation gradients through rank choice to gate preferences. Its scalar gate learns the rank-sparsity prior and receives optimizer weight decay; attribution still conditions discrete choices. V1 is replicated unchanged in its objective. New runs use common per-block encoder checkpointing; the historical deterministic rerun used it off, so timing boundaries and numerical/runtime differences must be disclosed.

V2 uses the same maximum-rank16 prefix factors and rank bank4,8,16. For adapter layerl and rankr, the update is `(alpha/r) * dropout(x) @ A[:r].T @ B[:,:r].T`, with alpha16. Bayesian factors use the preserved factorized Gaussian posterior, log-sigma clamping and isotropic Gaussian prior; fixed Bayesian and adaptive Bayesian keep that same posterior/initialization. V2 attribution uses detached absolute entropy-gradient×adapter-output averaged over batch/features, normalized by the largest layer score. This is predictive entropy for **both** deterministic and Bayesian arms, not artificial deterministic MI. Deterministic probing uses one output; Bayesian probing uses the mean of two genuine posterior samples.

For each layer: `pi = softmax((eta + 4*(a_norm-0.5)*[-1,0,1])/T)`, T1. eta has three zero-initialized trainable preferences per layer. During training `pi_ST = one_hot(argmax(pi)) - pi.detach() + pi`. All rank updates share a single dropout application and, for a Bayesian layer, one sampled maximum-rank factor pair per forward call. The forward value is a hard prefix choice. The backward pass uses a **surrogate** softmax gradient, not the exact gradient of a discrete decision. Training evaluates all candidate rank updates for that derivative; evaluation computes only the chosen rank. No rank-efficiency claim follows without the exported timings.

V2 rank penalty is the mean expected rank/16 across layers and across organ prompts. Original v1 instead penalizes its learned sigmoid usage prior. All variants average Dice+BCE and the implemented soft calibration loss across organs. Gaussian KL sums stored decoder factor terms and divides by the unfiltered training clip count. Coefficients: Dice1,BCE1,calibration0.1,KL0.0001,rank0.01 where applicable. Deterministic variants have KL0; fixed variants have rank penalty0. The fresh fixed Bayesian arm is shared between the v1 component question and the v2 factorial; it is not counted as a second independent experiment. These comparisons isolate **method packages**, including their regularizers; separate loss ablations are still needed to attribute an effect specifically to sampling versus KL or to allocation versus its penalty.

All SAM2 experiments use frozen official SAM2.1 Hiera Base+ commit2b90b9f5ceec907a1c18123530e92e794ad901a4, the verified backbone hash above, deterministic encoder LoRA and deterministic/Gaussian decoder LoRA according to the arm. CT is clipped to[-175,250]HU, scaled to[0,1], resized as in preserved code, replicated to RGB and ImageNet normalized. U-Net directly consumes the windowed CT plus a sigma10-pixel positive-point Gaussian. It has four downsampling levels,16/32/64/128/256 channels, GroupNorm, ReLU and bilinear upsampling, with activation checkpointing. It has no organ-ID/mask input. It shares AdamW lr0.0001, weight decay0.00001, cosine20-epoch schedule and selection rules. This common schedule is not evidence that each architecture has been optimally tuned.

## Uncertainty, statistics and figures

Selected-checkpoint tests export every organ-positive target, deterministic point coordinates, ranks, probabilities/uncertainty summaries, surface/calibration metrics and synchronized inference timings. Dense arrays are limited to an anatomy-defined qualitative subset: maximum GT-area liver,pancreas,right-adrenal slice per volume. All target records remain available. No masks or uncertainty maps are generated by an image model.

Additional validation-only analyses on the four new seed-replication fits use K2,4,8,16, prompt draws0,1,2, probe counts2,4,8 and fixed rank probes4,8,16 where meaningful. The primary policy remains no refinement. Probe count changes also consume a different posterior RNG stream, so those effects are not a pure isolated rank intervention. Rank agreement and matched points are checked. A three-member deterministic confidence ensemble uses **different selected weights trained at seeds42,123,2026**, not repeated deterministic predictions. Between-seed disagreement is labeled separately from Gaussian posterior MI.

The analysis exports actual curves, paired volume effects, per-organ comparisons, completed factorial contrasts, all/foreground/boundary calibration, pixel uncertainty–error association, histogram risk–coverage curves, within-volume mask-failure detection (Dice<0.5), sensitivity/rank stability, matched inference timing, bounded training-memory profiles and real qualitative panels. Figures are PNG/PDF/SVG with captions; incomplete experiments are excluded from comparisons. A schematic alone is not an experimental result. Undefined metrics and denominators are retained. HD95/ASSD are **resized 2D pixels**, not3D millimetres. A mean over defined surface values excludes empty-prediction infinities and must be read with its denominator.

All exploratory paired intervals resample whole volumes. Seed estimates are not pooled as extra patients, organ pairs are not independent observations, and six-volume intervals are not confirmatory significance tests. Five-fold point estimates remain internal validation, not untouched external confirmation. Ground-truth foreground/boundary conditioning answers different questions from all-pixel calibration. Probability-score improvement alone does not validate epistemic uncertainty or clinical reliability.

## Output and honest completion

`run_status.json`, `analysis/experiment_register.csv`, hashes, per-fit configs, logs, `last.pt`/`best.pt`, optimizer/scheduler/scaler/Python/NumPy/torch/CUDA RNG states and incremental validation/test records are saved. Validation does not advance the training RNG. Mid-epoch resume preserves shuffled order/cursor and actual update counts. Resume rejects changed data, scientific coefficients, source or runtime package identities. Training resumes from last, inference uses validation-selected best. Parent errors show the worker traceback tail rather than only a generic process failure.

`FULL_INTERNAL_PROGRAMME_COMPLETE` requires all17 research fits, their full selected tests, four replication sensitivity suites and the independent-seed ensemble. It does **not** mean external validation or all submission requirements are complete. An interrupted run remains PAUSED/FAILED with saved state. Passing CPU tests, profiling, generating a ZIP or exporting a figure does not establish research completion.

## What still prevents a universal “10/10” claim

1. Execute the new experiments and report adverse/equivocal outcomes, including any continued deterministic advantage. Do not select a winner from the inspected test data.
2. `[PENDING EXPERIMENT]` Independently untouched external data with dataset permissions, organ mapping, point policy and locked preprocessing; this package does not fabricate an external dataset. Further targeted encoder/decoder and loss-term ablations should follow a stated explanatory hypothesis, not a search for favorable test scores. The finite17-fit programme deliberately does not claim these additional studies completed.
3. `[AUTHOR INPUT]` Authors/contributions, ethics or exemption, dataset licenses/permissions, funding/conflicts, clinical expertise, patient linkage and intended use. Public release requires a real repository and persistent archive/DOI; this file cannot invent them.
4. Address the existing **BaLoRA** naming conflict. The code keeps the historical identifier for reproducibility, without inventing an acronym expansion or asserting that Bayesian LoRA is novel. Prefer a descriptive manuscript title such as “Deterministic and Bayesian low-rank SAM2 adaptation for oracle-point abdominal CT segmentation: an exploratory comparison.”
5. `[NOT REPORTED]` Untouched-domain performance,3D physical-space surface distances, clinical deployment benefit and optimally tuned published medical SAM baseline results. Do not fill these with inference from current aggregate Dice.

Primary references and platform documentation are in `REFERENCES.md`. Sixteen meaningful CPU integrity/regression checks were executed during authoring, including exact mid-epoch state resume, v1 gate-gradient limitation, v2 task gradients and incremental inference identities. They use explicitly synthetic fixtures and do not certify real-GPU integration; representative real-SAM2 smoke profiles are mandatory on Kaggle.

## Repeated execution cell

The old `_V5_RUN_STARTED` latch is no longer used. RUN_PENDING is the default: repeating the execution cell resumes unfinished stages, including a previously failed worker stage when its ledger is closed and time remains. Completed checkpoints and records are reused. Only a completed full programme or explicit RESULTS_ONLY mode displays cached output. The original deadline and all recorded hours are retained. Concurrency is blocked with a file lock, and unclosed accounting still needs recovery evidence. See `RERUN_FIX.md`. Scientific source and checkpoints remain unchanged.

## Uploaded failed follow-up audit

The uploaded follow-up failed before research fitting: SAM2 training.train shadowed the project module. The corrected worker entrypoint pins and prints both BALoRA training and official SAM2 provenance. No submitit installation is needed for this repair. Two closed failed-attempt ledgers are carried from the supplied archive; the cumulative floor is now35.9096370237325h (9.0903629762675h before reserves under the former45h cap). No new research checkpoint was present. Original v5 scientific Python files and method identities remain unchanged.

## Full-run continuation

Use EXECUTION_ACTION=RUN_PENDING and PROGRAMME_MODE=EXPANDED_INTERNAL for the full scheduled programme. Failed or partial status is no longer returned as a cached substitute for execution. Every actual attempt has a unique execution ID; a newly exported archive contains execution_receipt.json with that ID and the number of worker commands launched. An older ZIP is never presented as the export of an attempt that failed before exporting. Identical duplicate resume ZIPs are deduplicated by content. Session and cumulative limits remain enforced; this is resumable execution, not a promise that all17 new fits finish in one session.
