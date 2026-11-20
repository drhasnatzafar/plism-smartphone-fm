# Do Pathology Foundation Models Survive the Smartphone?

Code and results for the paper **"Do Pathology Foundation Models Survive the Smartphone? A Paired Evaluation of Nine Encoders on Microscope-Eyepiece Photographs"** (Hasnat Zafar, Ali Zafar; 2026).

Whole-slide scanners are unaffordable for many low-resource laboratories, where a smartphone photograph taken through the microscope eyepiece is a low-cost alternative. This repository evaluates whether pathology foundation models still work on such photographs, using paired scanner and smartphone images of the same tissue from **PLISM-sm**.

## Data
PLISM-sm (Ochi et al., *Scientific Data* 11:330, 2024; CC BY 4.0), available on Figshare: https://doi.org/10.25452/figshare.plus.23590791.
4,454 registered tissue locations × 13 devices (7 whole-slide scanners, 6 smartphones) = 57,902 tiles under 13 staining conditions. The data is not redistributed here.

## Encoders
UNI, UNI2-h, Prov-GigaPath, H-optimus-0, Virchow2, Phikon-v2, CONCH, DINOv2-L and an ImageNet ViT-B/16, all frozen. Gated models require accepting their licences on Hugging Face and an access token (`HF_TOKEN`).

## Analyses
(a) paired embedding consistency; (b) phone-to-scanner retrieval; (c) tissue-type probes trained on scanner tiles and tested on phone tiles (5-fold location-grouped CV, 1,000 location-level bootstrap CIs); (d) adaptations: per-device centring, CORAL, few-shot calibration; (e) per-phone degradation versus image sharpness and saturation.

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
python extract.py --tar <PLISM-sm tar> --csv <PLISM-sm csv> --model uni --out emb --batch 64
python analysis.py --emb emb --results results --boot 1000
pytest -q
```
See `DEVELOPER_NOTES.md` for all options. The study was run on Kaggle (2× T4 GPU for extraction, CPU for analysis).

## Citation
If you use this code, please cite the paper (arXiv link to be added) and the PLISM dataset (Ochi et al., 2024).

## Licence
Code: MIT (see `LICENSE`). PLISM-sm data and the encoder weights are subject to their own licences.
