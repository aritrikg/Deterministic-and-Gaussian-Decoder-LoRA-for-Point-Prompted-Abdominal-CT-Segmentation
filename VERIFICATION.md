# Verification

Checks performed on 18 November 2026 using Python 3.12 and PyTorch 2.11.0 CPU.

- Verified the original training source manifest against the completed Fold 1 study identity. The archived training notebook embeds the same source manifest.
- Ran the original training pipeline and model component test suite. 181 passed and one CUDA AMP test was skipped. The later research programme passed its 16 CPU integrity tests.
- Opened all five Streamlit views in local app tests. Confirmed the CT case explorer loaded saved mask overlays and displayed the Gaussian mutual-information map.
- Recomputed the three primary Dice scores from all 7,398 prompt records. Verified 7,640 manifest entries in the completed results archive and matched the replay's three original prediction arms to the archived Dice values with zero difference across all 7,398 rows.
- Regenerated the dropout comparison, spatial sensitivity tables, and paper figure outputs from their archived inputs. Rebuilt the Round 3 notebook using the bundled ledger.
- Checked the upload service's health route, token handling, request validation and PNG response routing with a simulated inference function. Extracted both trained adapters from the completed archive and verified their checksums.

Fresh BTCV GPU training and live mask generation were outside these CPU checks. For GPU setup, use the guides in the root README and in `demo/BALoRA_GPU_Backend/`.

The package adds the setup guide and CPU dependency list, lets the replay builder read its exact accounting JSON directly, creates the output directory for regenerated figures, and corrects the GPU setup script's startup message. The original training code and the verified inference notebook are preserved.
