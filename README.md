# BALoRA M.Tech Minor Project

**Deterministic and Gaussian Decoder LoRA for Point-Prompted Abdominal CT Segmentation**

Aritrik Ghosh · Supervisor Sourav Saha  
Department of Computer Science and Engineering, Swami Vivekananda University

## Open the demonstration

Extract the ZIP. Open a terminal in the extracted `BALoRA_Final_Project` folder. With Python 3.12, run:

```bash
cd demo/BALoRA_Streamlit
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

Open [localhost:8501](http://localhost:8501). In **Benchmark**, select **Equal volume–organ pairs** to match the primary analysis. Then open **CT cases** and choose a liver, pancreas, or right-adrenal example. The viewer shows the same positive click, reference mask, and predictions from all three methods. Enable **Show Gaussian decoder mutual information** to display the archived uncertainty map. The 18 saved cases and benchmark tables work without a GPU or model weights.

The **Upload CT** section previews a supplied image. To generate masks for a new image, configure the optional GPU inference service using [its setup guide](demo/BALoRA_GPU_Backend/README.md). It uses the two validation-selected adapters from the completed Fold 1 archive and the pinned SAM 2.1 Hiera Base+ checkpoint. The guide covers the HTTPS endpoint and matching token.

The dashboard source is from the [BALoRA3 repository](https://github.com/aritrikg/balora3/tree/dd47011f8b6b71a11aeca6e54e6f41ce777c66ff), commit `dd47011f8b6b71a11aeca6e54e6f41ce777c66ff`.

## Where the code is

| To show | File |
| --- | --- |
| Gaussian weight sampling and deterministic LoRA | [models/bayesian_lora.py](research/BALoRA_Full_Research_Package_v5/base_v4/source/models/bayesian_lora.py) |
| Two uncertainty probes and four final draws | [models/sam2_wrapper.py](research/BALoRA_Full_Research_Package_v5/base_v4/source/models/sam2_wrapper.py), `decode_with_adaptive_rank` |
| Active ranks 4, 8, and 16 | [models/rank_controller.py](research/BALoRA_Full_Research_Package_v5/base_v4/source/models/rank_controller.py) |
| Predictive entropy and mutual information | [models/confidence.py](research/BALoRA_Full_Research_Package_v5/base_v4/source/models/confidence.py) |
| Frozen SAM2 and deterministic LoRA | [models/frozen_sam2.py](research/BALoRA_Full_Research_Package_v5/base_v4/source/models/frozen_sam2.py), [models/deterministic_sam2.py](research/BALoRA_Full_Research_Package_v5/base_v4/source/models/deterministic_sam2.py) |
| BTCV dataset, preprocessing and split | [datasets/btcv_dataset.py](research/BALoRA_Full_Research_Package_v5/base_v4/source/datasets/btcv_dataset.py), [datasets/splits.py](research/BALoRA_Full_Research_Package_v5/base_v4/source/datasets/splits.py) |
| Training orchestration and validation selection | [full_pipeline.py](research/BALoRA_Full_Research_Package_v5/base_v4/full_pipeline.py), [training/train.py](research/BALoRA_Full_Research_Package_v5/base_v4/source/training/train.py), [training/deterministic_control.py](research/BALoRA_Full_Research_Package_v5/base_v4/source/training/deterministic_control.py) |
| Held-out evaluation and reporting | [evaluation/evaluate.py](research/BALoRA_Full_Research_Package_v5/base_v4/source/evaluation/evaluate.py), [evaluation/point_estimate.py](research/BALoRA_Full_Research_Package_v5/base_v4/source/evaluation/point_estimate.py), [analysis/reanalyze.py](reproducibility/analysis/reanalyze.py) |
| Inference replay, including decoder dropout and fixed Gaussian standard deviation | [inference/inference_audit.py](reproducibility/inference/inference_audit.py), [Round 3 notebook](reproducibility/BALoRA_Round3_Inference_Audit.ipynb) |
| Streamlit UI and upload backend | [app.py](demo/BALoRA_Streamlit/app.py), [server.py](demo/BALoRA_GPU_Backend/server.py) |

## Reproduce the presentation scores on CPU

From the extracted project root, run:

```bash
cd reproducibility
python -m pip install -r requirements.txt
python analysis/reanalyze.py analysis/data/per_prompt_fold1_seed42.csv reproduced
```

Open `reproduced/analysis_summary.json`. The completed seed-42 Fold 1 study used 19 training, 5 validation and 6 held-out BTCV volumes, with 2,466 matched test prompts per model. The primary Dice metric averages over slices within each volume–organ pair and then weights the 78 pairs equally.

| Method | Pair-weighted Dice |
| --- | ---: |
| Frozen SAM2 | 0.5123 |
| Deterministic LoRA | 0.8049 |
| Gaussian decoder LoRA (BALoRA) | 0.7982 |

The verification scripts, saved per-prompt results, replay outputs, and additional plot code are in [reproducibility/](reproducibility/README.md).

## Run the original training or replay in Kaggle

1. **Original training.** Import [the Fold 1 training notebook](research/BALoRA_Full_Research_Package_v5/historical/v4_notebook.ipynb). Attach the 30 labeled BTCV image/label pairs, select a CUDA GPU, and enable Internet for the pinned SAM2 setup. Its embedded source matches the code identity in the completed study. The archived run used a 45-hour cumulative allowance across sessions, with a 10.5-hour work limit per session. Review the allowance in the notebook and carry the latest saved output forward when resuming. See the [original run notes](research/BALoRA_Full_Research_Package_v5/base_v4/README_FULL_RUN.md) for the resume procedure.
2. **Inference replay.** Import [the Round 3 notebook](reproducibility/BALoRA_Round3_Inference_Audit.ipynb) and attach the same BTCV data and the completed Fold 1 results ZIP. The default `SCOPE = 'full'` runs the complete test set. Set `SCOPE = 'smoke'` to try two slices first. The replay reads existing model weights.
3. **Follow-on experiments.** [BALoRA_Full_Research_Kaggle.ipynb](research/BALoRA_Full_Research_Kaggle.ipynb) is the later multi-seed and multi-fold experiment programme. The presentation numbers above are from the completed Fold 1 comparison.

External inputs for training or live inference are the [BTCV challenge data](https://www.synapse.org/Synapse:syn3193805/wiki/89480), the SAM2 backbone fetched by setup, and the approximately 255 MiB [completed Fold 1 results archive](https://chatgpt.com/library?file_id=libfile_dea0608e7b888191a510ac2f8cccc85e&account_id=personal), which supplies the trained adapters for replay and live inference. The expected archive SHA-256 is `ccc8581ea2b5dfac4909056585448a9121f6e1d2a122352f6039d5680c185340`. Original training starts from the pretrained backbone and fresh adapters.

## Package verification

See [verification notes](VERIFICATION.md) for the checks performed on this package.
