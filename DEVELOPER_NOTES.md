# Do pathology foundation models survive the smartphone? (PLISM-sm)

Research code for probing patch encoders (nine were used in the paper) on PLISM-sm (Ochi et al., Sci Data 2024; CC BY 4.0): same tissue
tile, same stain, imaged by 6 smartphones and 7 WSI scanners. Built for Kaggle (2x T4, 12 h, ~20 GB disk).
These notes were written during development (November 2026), when everything except real-weight model loading had been
tested on synthetic data only (see *Status*). The study itself was then run on Kaggle with the real weights of all nine
encoders (2x T4); its outputs are in `results/` and `results/review/`.

## Files
| file | purpose |
|---|---|
| `encoders.py` | model registry (`python encoders.py` lists it), per-model preprocessing, fp16 inference, `debug` fake encoder |
| `extract.py` | stream the tar once per model -> `<model>.npy` (float16) + shared `index.csv` |
| `plism_data.py`, `imageproc.py` | CSV mapping, tar streaming, index building; quality proxies, Reinhard (numpy only) |
| `analysis.py` | analyses a-e, CSV + LaTeX + figures (PNG/PDF) |
| `tests/` | pytest on a synthetic tar.gz (3 stains x 13 devices x 6 locations) and synthetic embeddings |
| `kaggle_plism.ipynb` | install, HF secret, smoke test, resumable loop, analysis |

## Run
```bash
pip install -r requirements.txt
python extract.py --tar /kaggle/input/<ds>/41397708 --csv /kaggle/input/<ds>/41397675 --model uni --out /kaggle/working/emb --batch 64
python extract.py ... --model uni --out /kaggle/working/emb --stain-norm reinhard --stain-ref S360   # -> uni__reinhard.npy
python extract.py ... --model uni --out /tmp/smoke --limit 256                                         # smoke test: SEPARATE --out
python analysis.py --emb /kaggle/working/emb --results /kaggle/working/results [--models uni,virchow2] [--analyses abcde] [--boot 1000]
pytest -q
```
`--model` also accepts a comma list. If the CSV column names differ from the guesses, the loader prints the CSV
columns and the mapping it used; fix with `--colmap '{"tissue": "<column>", "path": "<column>"}'`. Device
phone/scanner assignment is keyword based; override with `--device-map '{"<label>": "phone"}'`.

## Outputs (`--results`)
`paired_consistency.csv`, `retrieval.csv`, `retrieval_summary.csv`, `classification.csv` (tidy: model, classifier, variant,
kind in {metric, drop, gain_vs_baseline}, eval_set in {scanner_heldout, phone_all, phone:<device>}, metric, value, ci_lo, ci_hi),
`classification_main.csv`, `per_phone_drop.csv`, `quality_correlation.csv`, `tile_quality_correlation.csv`; `tables/*.tex`
(booktabs); `figures/fig_*.png|pdf`; `cache/` (per model and analysis; delete a file to recompute).

## Design decisions
* **Streaming, no extraction.** `tarfile` mode `r|gz`; PNG decode + resize/normalise in a thread pool; the tar is read
  sequentially in the main thread. `index.csv` (row order = tar order) is built in one extra pass that also computes
  image-quality proxies (Laplacian variance, mean HSV saturation) and per-device Lab statistics; every model asserts
  that member names match the index row by row.
* **Resumable.** Finished models are skipped; interrupted ones resume from 4096-row chunk files in `.parts/<model>/`
  (the tar is re-streamed, but the finished prefix is neither decoded nor encoded). `--limit` runs write a limit-stamped index and
  refuse to share an output dir with a full run.
* **Resize 512 -> model resolution** (224; CONCH/CONCH v1.5 448; GigaPath 256 -> centre-crop 224) with PIL bicubic. plism-benchmark
  instead feeds 224-px tiles; results are therefore not directly comparable to its leaderboard.
* **fp16 autocast** (T4 has no bf16); `DataParallel` over both GPUs when available. Embeddings stored float16, analysed after float32 L2 normalisation.
* **Groups.** `group_id` = stain + coordinates (same location on all 13 devices) is used for pairing/retrieval. For *splitting and
  bootstrapping* we use the coarser `loc_id` = tissue + coordinates, which also merges the same location across stains, so no
  tissue location can be in both train and test even if stains share coordinates. `GroupKFold(5, shuffle, seed 0)` on `loc_id`.
* **Retrieval.** Rank of the true tile with ties counted half (a collapsed embedding gets chance, not 100 %). `same_stain`: gallery =
  that scanner's tiles of the query's stain; `cross_stain`: gallery = that scanner's tiles from all stains (only the same group is correct).
  All 42 phone->scanner and 42 ordered scanner->scanner pairs are computed; summaries average them.
* **Probes.** Train on scanner tiles only (all 7 scanners pooled) of train-fold locations; features L2-normalised, then standardised on train;
  LR = multinomial lbfgs, C=1, max 200 its, tol 1e-3 (not tuned); kNN = cosine, k=20, uniform votes. Test on held-out scanner tiles and on phone tiles of
  the same test locations. Out-of-fold predictions are pooled; balanced accuracy / macro-F1 equal sklearn's (tested).
  CIs: 1000 bootstrap resamples of *locations*; the drop (scanner - phone) uses the same resamples for both terms.
* **Fixes (transductive where stated).** *center*: per-device mean subtraction; scanner means from train-fold tiles, phone means from the
  **unlabelled test-fold phone tiles (transductive)**. *CORAL*: source = train scanner features, target = **unlabelled test-fold phone tiles
  (transductive)**, covariance shrinkage 0.1, source (and held-out scanner) features mapped, phones untouched. *few-shot k in {1,5,10}*: k random labelled phone
  tiles per class from train-fold locations added to the training set (one draw per fold; not repeated). Stain normalisation (image level) is the
  separate `--stain-norm reinhard` flag: Lab mean/std matching of every tile (phones and scanners) to the mean statistics of a reference scanner
  (`--stain-ref`, default S360), foreground pixels only. Macenko is not implemented.
* **Quality correlation.** Spearman across the 6 phones (n = 6, descriptive only) between LR baseline BA drop and median-sharpness ratio / saturation, per model and pooled
  (model-centred); plus tile-level Spearman between cosine-to-scanners and sharpness/saturation differences (`tile_quality_correlation.csv`).

## Models, licences, gating
Wrappers adapted from owkin/plism-benchmark (repo README: **CC BY 4.0**; no separate LICENSE file found; `plismbench/models/*.py`; attribution kept in comments `(plism-benchmark)`).
Licence/gating column is **from memory, UNVERIFIED** (Hugging Face was blocked): verify on each model card; most are non-commercial/gated.

| key | model | dim | input | licence (UNVERIFIED) | gated |
|---|---|---|---|---|---|
| uni | UNI (ViT-L/16) | 1024 | 224 | CC-BY-NC-ND 4.0 | yes |
| uni2h | UNI2-h | 1536 | 224 | CC-BY-NC-ND 4.0 | yes |
| conch | CONCH v1 (image enc., pre-projection) | 512 | 448 | CC-BY-NC-ND 4.0 | yes (+ `pip install git+https://github.com/mahmoodlab/CONCH.git`) |
| conchv15 | CONCH v1.5 via TITAN `return_conch()` | 768 | 448 | CC-BY-NC-ND 4.0 | yes (`trust_remote_code`) |
| virchow2 | Virchow2 (CLS + mean patch) | 2560 | 224 | CC-BY-NC-ND 4.0 | yes |
| hoptimus0 / hoptimus1 | H-optimus-0 / -1 | 1536 | 224 | Apache-2.0 / CC-BY-NC-ND 4.0 | no / yes |
| h0mini (h0mini_cm) | H0-mini, CLS (optional CLS+mean) | 768 (1536) | 224 | CC-BY-NC-ND 4.0 | yes |
| phikon2 | Phikon-v2 (transformers, CLS) | 1024 | 224 | non-commercial | yes |
| gigapath | Prov-GigaPath tile encoder | 1536 | 256->224 | Apache-2.0 w/ gating | yes |
| vitb16_in | ViT-B/16 IN-21k->1k (timm augreg) baseline | 768 | 224 | Apache-2.0 | no |
| dinov2_l | DINOv2 ViT-L/14 (timm) baseline | 1024 | 224 | see card | no |

Set `HF_TOKEN`; request access on each gated model page beforehand.

## UNVERIFIED (could not be checked: Hugging Face blocked, no weights, no GPU)
1. **Nothing was run with real weights or on a GPU.** All timm/transformers/conch loading code is untested; `torch`/`timm` were not even installed here.
2. Hub model cards could not be read: preprocessing for UNI2-h, Virchow2, H-optimus-0/-1, H0-mini, UNI is taken from plism-benchmark wrappers (ImageNet or H-optimus
   mean/std, 224). `_timm_cfg_check` prints a warning at load time if timm's `pretrained_cfg` mean/std disagree.
3. **CONCH v1.5**: 448 px + ImageNet mean/std assumed; the TITAN `eval_transform` is printed at load, compare it. Loading via `TITAN.return_conch()` follows plism-benchmark.
4. **Phikon-v2** mean/std/size: defaults assumed, overwritten from `AutoImageProcessor` at load when reachable (image size is NOT taken from it).
5. **H0-mini**: CLS only follows plism-benchmark and the task brief; the model card may recommend CLS + mean patch tokens (`h0mini_cm`, register-token count / patch start index 5 assumed).
6. H-optimus-1 timm kwargs identical to H-optimus-0 assumed; Virchow2 patch tokens start at index 5 (4 registers) per plism-benchmark.
7. DINOv2 ViT-L/14 via timm at `img_size=224` (pos-embed interpolation) is my choice; the baseline ViT-B/16 normalisation is re-read from timm at load.
8. All licence / gating strings above.
9. PLISM-sm specifics: that tar paths are `<stain>/<device>/<x0>_<y0>_<x1>_<y1>.png`, that the same filename (coordinates) denotes the same location on all devices
   within a stain, CSV column names (candidate lists in `plism_data.DEFAULT_COLMAP`), device label strings (keyword rules in `plism_data.device_type`), and
   that the 13 stains x 13 devices layout yields 4,454 complete groups. The loader prints unmatched/unknown items; check the first lines of the log.
10. Throughput/ETA on T4 is unmeasured (PNG decode + gzip may dominate for small models; `--workers`).

## Status
`pytest -q`: 23 tests pass (synthetic tar.gz streaming, index/alignment, crash-resume, limit guard, classification cache invalidated when settings change, stain-norm flag, colmap override, no-CSV fallback,
leakage-free folds, memorisation test, sklearn-equivalence of the bootstrap metrics, retrieval/cosine sanity, CORAL, end-to-end outputs).

## Known risks
LR probes on ~25k x 2560 features x 6 variants x 5 folds x 12 models is CPU heavy (hours); use `--max-train` or `--analyses abc` first.
Cross-stain retrieval/negatives assume positions repeat only within stain. Phone-vs-scanner probe sets differ in size (7 vs 6 devices) but rates are balanced-accuracy normalised.
