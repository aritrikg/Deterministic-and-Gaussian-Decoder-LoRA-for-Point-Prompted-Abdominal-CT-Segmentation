# RUN_PENDING full-run continuation

The previous launcher could show a cached FAILED/PAUSED result after any worker activity. The current default RUN_PENDING actually calls the scientific runner for unfinished work when the existing ledger is closed and time remains. It reuses completed fits and saved evaluation records; it does not restart completed research. Explicit RESULTS_ONLY and an already complete full programme retain display-only behavior.

The original host deadline and cumulative accounting remain intact. Active locks and unresolved unclosed ledgers still block a duplicate launch. Every actual attempt has a new execution ID. execution_receipt.json and run_status.json in its output archive identify the attempt and worker count. An old archive is preserved but is not returned as a fresh export if startup fails before creating one.

The wrong SAM2/BALoRA training import remains fixed by worker_entry.py. Scientific v5 Python files and checkpoint identities are unchanged. The uploaded failed sessions remain counted at35.9096370237325h. Matching duplicate resume ZIP copies are accepted once; different resume histories still require an explicit RESUME_PATH.

Run All with EXECUTION_ACTION=RUN_PENDING and PROGRAMME_MODE=EXPANDED_INTERNAL. If the host deadline is reached, save the current ZIP and resume in a new Kaggle execution session. A full programme needs actual GPU execution and may span several sessions. CPU/fixture tests are not completed training evidence.
