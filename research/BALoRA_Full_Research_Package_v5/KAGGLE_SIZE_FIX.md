# Self-contained Kaggle source fix

The first notebook exceeded the source limit by embedding the entire distribution, including old notebook copies. The next small loader required an attached ZIP with an exact source manifest; an absent or different package caused the reported FileNotFoundError before any training began.

The current notebook fixes both packaging problems. It embeds only the runtime files, required original assets, carried ledger evidence and concise documentation, compressed together losslessly. The builder enforces a900,000-byte notebook ceiling, below the reported1,000,000-byte source limit. Old notebook copies and figure previews remain in this optional companion ZIP.

## What to do

1. Import the newly updated `BALoRA_Full_Research_Kaggle.ipynb`.
2. Keep BTCV, the latest completed reference results and any latest v5 resume results attached.
3. Run the bundled-source cell or Run All. **No source ZIP input, manifest path or SOURCE_PACKAGE_PATH is required.** Existing source-package inputs may remain attached; the bootstrap does not depend on them.

The bootstrap checks the compressed archive, validates every contained file and writes only the declared regular files to its working source folder. It never resets training state or cumulative accounting. Every original v4 file, v5 scientific Python file and carried evidence file remains byte-identical to the preceding package. No retraining is needed to fix this source-loading error; new experiments still follow the existing study plan.

The full programme's budget, session limits, dataset requirements and scientific limitations are unchanged. Resolving the bootstrap error does not establish that later GPU experiments have completed. Original CPU regression checks and explicit packaging checks use synthetic fixtures, not research results.
