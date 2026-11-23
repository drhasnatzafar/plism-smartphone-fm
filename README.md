# Do Pathology Foundation Models Survive the Smartphone?

Code and results for the paper **"Do Pathology Foundation Models Survive the Smartphone? A Paired Evaluation of Nine Encoders on Microscope-Eyepiece Photographs"** (Hasnat Zafar, Ali Zafar; 2026).

Whole-slide scanners are unaffordable for many low-resource laboratories, where a smartphone photograph taken through the microscope eyepiece is a low-cost alternative. This repository evaluates whether pathology foundation models still work on such photographs, using paired scanner and smartphone images of the same tissue from **PLISM-sm**.

## Data
PLISM-sm (Ochi et al., *Scientific Data* 11:330, 2024; CC BY 4.0), available on Figshare: https://doi.org/10.25452/figshare.plus.23590791.
4,454 registered tissue locations × 13 devices (7 whole-slide scanners, 6 smartphones) = 57,902 tiles under 13 staining conditions. The data is not redistributed here.

## Encoders
UNI, UNI2-h, Prov-GigaPath, H-optimus-0, Virchow2, Phikon-v2, CONCH, DINOv2-L and an ImageNet ViT-B/16, all frozen. Gated models require accepting their licences on Hugging Face and an access token (`HF_TOKEN`).

## Analyses
(a) paired embedding consistency; (b) phone-to-scanner retrieval; (c) tissue-type probes trained on scanner tiles and tested on phone tiles (5-fold location-grouped CV, 1,000 location-level bootstrap CIs); (d) adaptations: per-device centring, CORAL, few-shot adaptation with 1, 5 or 10 labelled phone tiles per class; (e) per-phone degradation versus image sharpness and saturation.

## Main result
| Encoder | Scanner BA | Phone BA | Drop |
|---|---|---|---|
| UNI2-h | 0.985 | 0.969 | 0.017 |
| Prov-GigaPath | 0.979 | 0.935 | 0.044 |
| H-optimus-0 | 0.988 | 0.936 | 0.052 |
| UNI | 0.982 | 0.927 | 0.055 |
| Virchow2 | 0.985 | 0.929 | 0.056 |
| Phikon-v2 | 0.976 | 0.890 | 0.086 |
| DINOv2-L | 0.893 | 0.746 | 0.147 |
| CONCH | 0.922 | 0.741 | 0.182 |
| ImageNet ViT-B/16 | 0.895 | 0.670 | 0.225 |

Linear probe, balanced accuracy (BA). Full results with confidence intervals are in `results/`.

## Repository contents
| Path | Contents |
|---|---|
| `extract.py`, `encoders.py`, `plism_data.py`, `imageproc.py` | Stream PLISM-sm and extract embeddings for each encoder |
| `analysis.py` | Analyses (a)–(e); writes CSVs, LaTeX tables and figures |
| `make_figs.py` | Paper figures from the result CSVs |
| `kaggle_plism.ipynb` | Notebook used to run the study on Kaggle |
| `results/` | All result CSVs, LaTeX tables and figures reported in the paper |
| `tests/` | Tests on synthetic data |
| `DEVELOPER_NOTES.md` | Detailed usage, options and design decisions |

## Reproduce
```bash
pip install -r requirements.txt
export HF_TOKEN=...   # after accepting the licences of the gated models on Hugging Face

# 1. Extract embeddings for the nine encoders used in the paper (one .npy per encoder + a shared index.csv)
python extract.py --tar <PLISM-sm tar> --csv <PLISM-sm csv> --out emb --batch 64 \
    --model uni,uni2h,gigapath,hoptimus0,virchow2,phikon2,conch,dinov2_l,vitb16_in

# 2. Main analyses (a)-(e): CSVs, LaTeX tables and figures in results/
#    Use a fresh --results directory for each configuration; cached classification results are
#    recomputed automatically if --seed, --boot, --folds, --max-train or --lr-iter change.
python analysis.py --emb emb --results results --boot 1000 --seed 0

# 3. Sensitivity analyses of the paper (within-stain ordinary vs spatially disjoint folds,
#    paired adaptation comparison over three seeds) -> results/review/
python review_analyses.py --emb emb --out results/review --seeds 3 --boot 1000
#    outputs: within_stain.csv, paired_adaptation.csv, grouping_counts.csv, spatial_components.csv

# 4. Figures as printed in the paper
python make_figs.py results && python make_supp_figs.py   # -> results/figures/

# tests (synthetic data, no GPU or model weights needed)
pytest -q
```
Encoder keys and checkpoints: `python encoders.py`. See `DEVELOPER_NOTES.md` for all options.
The study was run on Kaggle (2x T4 GPU for extraction, CPU for analysis); `kaggle_plism.ipynb` is the notebook used.

## Citation
If you use this code, please cite the paper (arXiv link to be added) and the PLISM dataset (Ochi et al., 2024).

## Licence
Code: MIT (see `LICENSE`). PLISM-sm data and the encoder weights are subject to their own licences.
