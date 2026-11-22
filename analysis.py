#!/usr/bin/env python
"""Analyses of PLISM-sm embeddings: do pathology foundation models survive the smartphone?

All analyses read ``<emb>/index.csv`` + ``<emb>/<model>.npy`` (written by extract.py); CPU only, numpy/sklearn.

    python analysis.py --emb /kaggle/working/emb --results /kaggle/working/results [--models uni,virchow2] \
                       [--analyses abcde] [--boot 1000] [--folds 5]

a  paired consistency        cosine(phone, scanner) of the same group vs scanner-scanner and phone-phone
b  cross-device retrieval    phone->scanner (same stain / cross stain), scanner->scanner reference
c  tissue classification     LR + kNN probes trained on scanners; tested on held-out scanner tiles and phones,
                             group-disjoint 5-fold CV, balanced acc / macro-F1, group-bootstrap CIs of the drop
d  cheap fixes               per-device mean-centering, CORAL, k-shot phone calibration (same protocol)
e  per-phone + quality       per-phone drop; correlation with sharpness / saturation proxies

Definitions
-----------
group   : same stain + same tissue location, imaged by all 13 devices (``group_id`` in index.csv)
loc     : tissue + location (``loc_id``), merges the same location across stains.  ALL splits and bootstrap
          resampling are done on ``loc`` which is a (conservative) superset of ``group``: the same tissue
          location never appears in train and test.
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

KS = (1, 5, 10)
VARIANTS = ["baseline", "center", "coral", "fewshot1", "fewshot5", "fewshot10"]
CLASSIFIERS = ["lr", "knn"]


# ======================================================================================
# loading / grid
# ======================================================================================
def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def load_index(emb_dir) -> pd.DataFrame:
    return pd.read_csv(Path(emb_dir) / "index.csv")


def load_embeddings(emb_dir, model: str, n_expected: int | None = None) -> np.ndarray:
    """float16 file -> float32, L2-normalised (cosine geometry for all analyses)."""
    E = np.load(Path(emb_dir) / f"{model}.npy").astype(np.float32)
    if n_expected is not None and len(E) != n_expected:
        raise ValueError(f"{model}: {len(E)} rows != index {n_expected}")
    bad = ~np.isfinite(E).all(1)
    if bad.any():
        warnings.warn(f"{model}: {bad.sum()} rows with non-finite values set to 0")
        E[bad] = 0.0
    return l2norm(E)


@dataclass
class Grid:
    devices: list          # device names (column order)
    is_phone: np.ndarray   # bool (n_dev,)
    rows: np.ndarray       # (G, n_dev) row ids into the embedding matrix, complete groups only
    stain: np.ndarray      # (G,)
    group_ids: np.ndarray  # (G,)

    @property
    def P(self): return np.where(self.is_phone)[0]
    @property
    def S(self): return np.where(~self.is_phone)[0]


def build_grid(index: pd.DataFrame) -> Grid:
    """Arrange rows as (group x device); keeps only groups imaged by every device exactly once."""
    index = index[index.device_type.isin(["phone", "scanner"])]
    devices = list(pd.unique(index.device))
    gcode, gids = pd.factorize(index.group_id)
    dcode = index.device.map({d: i for i, d in enumerate(devices)}).values
    grid = -np.ones((len(gids), len(devices)), dtype=np.int64)
    dup = 0
    for r, g, d in zip(index["row"].values, gcode, dcode):
        dup += grid[g, d] >= 0
        grid[g, d] = r
    if dup:
        warnings.warn(f"{dup} duplicate (group, device) entries; last one kept")
    keep = (grid >= 0).all(1)
    stain = index.groupby("group_id").stain.first().reindex(gids).values
    is_phone = np.array([index.loc[index.device == d, "device_type"].iloc[0] == "phone" for d in devices])
    if keep.sum() < len(keep):
        print(f"[grid] keeping {keep.sum()}/{len(keep)} complete groups")
    return Grid(devices, is_phone, grid[keep], stain[keep], gids[keep])


def _by_stain(grid: Grid):
    for s in pd.unique(grid.stain):
        yield np.where(grid.stain == s)[0]


# ======================================================================================
# a. paired consistency
# ======================================================================================
def paired_similarities(E: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """S[g, i, j] = cos(device i, device j) within group g; neg[g, p, s] = cos(phone p of g, scanner s of g+1)
    (g+1 = next group of the SAME stain: a different-tissue-location negative)."""
    G, D = grid.rows.shape
    S = np.empty((G, D, D), np.float32)
    neg = np.empty((G, len(grid.P), len(grid.S)), np.float32)
    for gi in _by_stain(grid):
        X = E[grid.rows[gi]]  # (g, 13, d)
        S[gi] = np.einsum("gid,gjd->gij", X, X, optimize=True)
        Xr = np.roll(X, -1, axis=0)
        neg[gi] = np.einsum("gpd,gsd->gps", X[:, grid.P], Xr[:, grid.S], optimize=True)
    return S, neg


def _q(x: np.ndarray) -> dict:
    x = np.asarray(x).ravel()
    return {"median": float(np.median(x)), "q25": float(np.percentile(x, 25)), "q75": float(np.percentile(x, 75)),
            "mean": float(x.mean()), "n": int(x.size)}


def summarise_paired(model: str, S: np.ndarray, neg: np.ndarray, grid: Grid) -> pd.DataFrame:
    P, Sc = grid.P, grid.S
    rows = []
    iu = np.triu_indices(len(Sc), 1)
    ip = np.triu_indices(len(P), 1)
    pools = {
        "phone-scanner": S[:, P][:, :, Sc],
        "scanner-scanner": S[:, Sc][:, :, Sc][:, iu[0], iu[1]],
        "phone-phone": S[:, P][:, :, P][:, ip[0], ip[1]],
        "phone-scanner (different location)": neg,
    }
    for name, v in pools.items():
        rows.append({"model": model, "level": "overall", "comparison": name, "device": "all", **_q(v)})
    for a, p in enumerate(P):  # per phone: vs scanners, vs other phones
        rows.append({"model": model, "level": "device", "comparison": "phone-scanner", "device": grid.devices[p], **_q(S[:, p][:, Sc])})
        others = [q for q in P if q != p]
        rows.append({"model": model, "level": "device", "comparison": "phone-phone", "device": grid.devices[p], **_q(S[:, p][:, others])})
    for s in Sc:  # per scanner: vs other scanners, vs phones
        others = [q for q in Sc if q != s]
        rows.append({"model": model, "level": "device", "comparison": "scanner-scanner", "device": grid.devices[s], **_q(S[:, s][:, others])})
        rows.append({"model": model, "level": "device", "comparison": "phone-scanner", "device": grid.devices[s], **_q(S[:, s][:, P])})
    return pd.DataFrame(rows)


# ======================================================================================
# b. cross-device retrieval
# ======================================================================================
def rank_of_truth(Q: np.ndarray, Gm: np.ndarray, true_idx: np.ndarray, eps: float = 1e-6, chunk: int = 1024) -> np.ndarray:
    """Rank (0 = best) of the true gallery item for each query by cosine; ties count half (no free wins for
    degenerate/collapsed embeddings)."""
    out = np.empty(len(Q))
    for i in range(0, len(Q), chunk):
        s = Q[i:i + chunk] @ Gm.T
        t = s[np.arange(len(s)), true_idx[i:i + chunk]][:, None]
        greater = (s > t + eps).sum(1)
        equal = (np.abs(s - t) <= eps).sum(1) - 1
        out[i:i + chunk] = greater + 0.5 * equal
    return out


def retrieval(E: np.ndarray, grid: Grid, model: str, ks=KS) -> pd.DataFrame:
    """Query = tile of device a; gallery = tiles of device b.  ``same_stain``: gallery restricted to the query's
    stain; ``cross_stain``: gallery = that device's tiles from ALL stains (positive = same group only)."""
    pairs = [(a, b, "phone->scanner") for a in grid.P for b in grid.S] + \
            [(a, b, "scanner->scanner") for a in grid.S for b in grid.S if a != b]
    stain_idx = list(_by_stain(grid))
    res = []
    for a, b, kind in pairs:
        r_same = np.concatenate([rank_of_truth(E[grid.rows[gi, a]], E[grid.rows[gi, b]], np.arange(len(gi))) for gi in stain_idx])
        r_cross = rank_of_truth(E[grid.rows[:, a]], E[grid.rows[:, b]], np.arange(len(grid.rows)))
        for proto, r in (("same_stain", r_same), ("cross_stain", r_cross)):
            res.append({"model": model, "kind": kind, "query_device": grid.devices[a], "gallery_device": grid.devices[b],
                        "protocol": proto, **{f"top{k}": float((r < k).mean()) for k in ks}, "n_queries": len(r)})
    return pd.DataFrame(res)


# ======================================================================================
# c/d. classification protocol
# ======================================================================================
def make_folds(loc: np.ndarray, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """fold id per row with GroupKFold on ``loc`` (shuffled when the installed sklearn supports it)."""
    n_splits = min(n_splits, len(np.unique(loc)))
    try:
        gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    except TypeError:  # older sklearn: deterministic, size-balanced
        gkf = GroupKFold(n_splits=n_splits)
    fold = np.empty(len(loc), np.int64)
    for f, (_, te) in enumerate(gkf.split(np.zeros(len(loc)), groups=loc)):
        fold[te] = f
    return fold


def _mat_pow(C: np.ndarray, p: float) -> np.ndarray:
    w, V = np.linalg.eigh(C)
    return (V * np.maximum(w, 1e-12) ** p) @ V.T


def coral_matrix(Xs: np.ndarray, Xt: np.ndarray, shrink: float = 0.1) -> np.ndarray:
    """CORAL (Sun et al. 2016): A = Cs^{-1/2} Ct^{1/2}; source features are mapped with X @ A.
    Covariances are shrunk toward a scaled identity ((1-a) C + a tr(C)/D I) for stability in D up to 2560."""
    def cov(X):
        C = np.cov(X, rowvar=False, dtype=np.float64)
        return (1 - shrink) * C + shrink * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
    return (_mat_pow(cov(Xs), -0.5) @ _mat_pow(cov(Xt), 0.5)).astype(np.float32)


def _center_by_device(X: np.ndarray, dev: np.ndarray, means: dict) -> np.ndarray:
    X = X.copy()
    for d, m in means.items():
        X[dev == d] -= m
    return X


def knn_predict(Xtr, ytr, Xte, k=20, n_cls=None, chunk=2048) -> np.ndarray:
    """Cosine kNN with uniform voting (ties -> lowest class id)."""
    n_cls = n_cls or int(ytr.max()) + 1
    Xtr, Xte = l2norm(Xtr), l2norm(Xte)
    kk = min(k, len(Xtr))
    out = np.empty(len(Xte), np.int64)
    for i in range(0, len(Xte), chunk):
        s = Xte[i:i + chunk] @ Xtr.T
        idx = np.argpartition(-s, kk - 1, axis=1)[:, :kk]
        flat = (np.arange(len(s))[:, None] * n_cls + ytr[idx]).ravel()
        out[i:i + chunk] = np.bincount(flat, minlength=len(s) * n_cls).reshape(len(s), n_cls).argmax(1)
    return out


def lr_predict(Xtr, ytr, Xtes, C=1.0, max_iter=200, seed=0):
    """Standardise on train, multinomial logistic regression (lbfgs), predict each matrix in ``Xtes``."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence warnings: fixed iteration budget by design
        clf = LogisticRegression(C=C, max_iter=max_iter, tol=1e-3, random_state=seed).fit((Xtr - mu) / sd, ytr)
    return [clf.predict((X - mu) / sd) for X in Xtes]  # predict() already returns class labels


def class_counts(y, p, loc, n_loc, C):
    """Per-loc, per-class true-positive / support / predicted counts (n_loc x C) for fast bootstrapping."""
    ok = y == p
    tp = np.bincount(loc[ok] * C + y[ok], minlength=n_loc * C).reshape(n_loc, C)
    sup = np.bincount(loc * C + y, minlength=n_loc * C).reshape(n_loc, C)
    prd = np.bincount(loc * C + p, minlength=n_loc * C).reshape(n_loc, C)
    return tp.astype(np.float64), sup.astype(np.float64), prd.astype(np.float64)


def metrics_from_counts(tp, sup, prd) -> tuple[np.ndarray, np.ndarray]:
    """Balanced accuracy (mean recall over classes present) and macro-F1 (over classes in truth or prediction);
    identical to sklearn's balanced_accuracy_score / f1_score(average='macro'). Works on (..., C) arrays."""
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(sup > 0, tp / sup, np.nan)
        f1 = np.where(sup + prd > 0, 2 * tp / (sup + prd), np.nan)
    return np.nanmean(rec, -1), np.nanmean(f1, -1)


def bootstrap_weights(n_loc: int, B: int, seed: int) -> np.ndarray:
    """(B+1, n_loc) multiplicities; row 0 = original sample, rows 1.. = resampling locations with replacement."""
    rng = np.random.default_rng(seed)
    M = np.stack([np.bincount(rng.integers(0, n_loc, n_loc), minlength=n_loc) for _ in range(B)]).astype(np.float64)
    return np.vstack([np.ones((1, n_loc)), M])


def _ci(x: np.ndarray) -> tuple[float, float, float]:
    return float(x[0]), float(np.nanpercentile(x[1:], 2.5)), float(np.nanpercentile(x[1:], 97.5))


def run_probe_protocol(E: np.ndarray, index: pd.DataFrame, variants=VARIANTS, classifiers=CLASSIFIERS, n_splits=5,
                       seed=0, max_train: int | None = None, lr_iter=200, knn_k=20, coral_shrink=0.1,
                       n_fewshot_seeds=1, verbose=True) -> dict:
    """Out-of-fold predictions for every (variant, classifier).

    Train: scanner tiles of train-fold locs.  Test: (i) scanner tiles of test-fold locs, (ii) phone tiles of
    test-fold locs.  Returns dict with per-row predictions and the bookkeeping needed for bootstrapping.
    """
    idx = index[index.tissue.notna() & index.device_type.isin(["phone", "scanner"])]
    if idx.empty:
        raise ValueError("no rows with tissue labels")
    rows = idx["row"].values
    y_all, classes = pd.factorize(idx.tissue)
    loc_all, locs = pd.factorize(idx.loc_id)
    is_ph = (idx.device_type == "phone").values
    dev = idx.device.values
    fold = make_folds(loc_all, n_splits, seed)
    C = len(classes)
    X = E[rows]
    s_i, p_i = np.where(~is_ph)[0], np.where(is_ph)[0]
    out = {"classes": classes, "locs": locs, "loc_s": loc_all[s_i], "loc_p": loc_all[p_i], "y_s": y_all[s_i], "y_p": y_all[p_i],
           "dev_p": dev[p_i], "fold": fold, "fold_s": fold[s_i], "fold_p": fold[p_i], "pred_s": {}, "pred_p": {}}
    for v in variants:
        for c in classifiers:
            out["pred_s"][(v, c)] = np.full(len(s_i), -1, np.int64)
            out["pred_p"][(v, c)] = np.full(len(p_i), -1, np.int64)
    for f in range(fold.max() + 1):
        tr = s_i[fold[s_i] != f]
        te_s = s_i[fold[s_i] == f]
        te_p = p_i[fold[p_i] == f]
        ph_tr = p_i[fold[p_i] != f]  # labelled phone tiles of TRAIN locations (few-shot pool)
        if len(tr) == 0 or len(te_p) == 0:
            continue
        if max_train and len(tr) > max_train:
            tr = np.sort(np.random.default_rng(seed + f).choice(tr, max_train, replace=False))
        Xtr, ytr, Xs, Xp = X[tr], y_all[tr], X[te_s], X[te_p]
        for v in variants:
            Xtr_v, ytr_v, Xs_v, Xp_v = Xtr, ytr, Xs, Xp
            if v == "center":
                # scanners: per-device mean from TRAIN-fold tiles, applied to train and test scanner tiles.
                # phones: per-device mean of the UNLABELLED TEST-fold phone tiles (transductive!).
                ms = {d: Xtr[dev[tr] == d].mean(0) for d in np.unique(dev[tr])}
                mp = {d: Xp[dev[te_p] == d].mean(0) for d in np.unique(dev[te_p])}
                Xtr_v = _center_by_device(Xtr, dev[tr], ms)
                Xs_v = _center_by_device(Xs, dev[te_s], ms) if len(te_s) else Xs
                Xp_v = _center_by_device(Xp, dev[te_p], mp)
            elif v == "coral":
                # align SOURCE (train scanner) 2nd-order stats to the unlabelled test-fold phone tiles (transductive)
                # CORAL with explicit mean handling: centre source, re-colour to target covariance, add target mean
                mu_s, mu_t = Xtr.mean(0), Xp.mean(0)
                A = coral_matrix(Xtr, Xp, coral_shrink)
                Xtr_v, Xs_v = (Xtr - mu_s) @ A + mu_t, (Xs - mu_s) @ A + mu_t
            elif v.startswith("fewshot"):
                k = int(v[len("fewshot"):])
                rng = np.random.default_rng(seed * 1000 + f)
                pick = []
                for cl in range(C):
                    pool = ph_tr[y_all[ph_tr] == cl]
                    if len(pool):
                        pick.append(rng.choice(pool, min(k, len(pool)), replace=False))
                pick = np.concatenate(pick) if pick else np.array([], int)
                Xtr_v, ytr_v = np.vstack([Xtr, X[pick]]), np.concatenate([ytr, y_all[pick]])
            for c in classifiers:
                if c == "lr":
                    ps, pp = lr_predict(Xtr_v, ytr_v, [Xs_v, Xp_v], max_iter=lr_iter, seed=seed)
                else:
                    ps, pp = knn_predict(Xtr_v, ytr_v, Xs_v, knn_k, C), knn_predict(Xtr_v, ytr_v, Xp_v, knn_k, C)
                out["pred_s"][(v, c)][np.searchsorted(s_i, te_s)] = ps
                out["pred_p"][(v, c)][np.searchsorted(p_i, te_p)] = pp
        if verbose:
            print(f"  fold {f}: train {len(tr)}, test scanner {len(te_s)}, test phone {len(te_p)}", flush=True)
    return out


def summarise_probe(model: str, res: dict, B: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Tidy table: metrics (+95% group-bootstrap CI), scanner->phone drops, and gains over the baseline."""
    C, n_loc = len(res["classes"]), len(res["locs"])
    M = bootstrap_weights(n_loc, B, seed)
    phones = list(pd.unique(res["dev_p"]))
    subsets = {"scanner_heldout": None, "phone_all": slice(None)}
    for p in phones:
        subsets[f"phone:{p}"] = res["dev_p"] == p

    def boot(v, c, name):
        if name == "scanner_heldout":
            y, p, l = res["y_s"], res["pred_s"][(v, c)], res["loc_s"]
        else:
            m = subsets[name]
            y, p, l = res["y_p"][m], res["pred_p"][(v, c)][m], res["loc_p"][m]
        tp, sup, prd = class_counts(y, p, l, n_loc, C)
        ba, f1 = metrics_from_counts(M @ tp, M @ sup, M @ prd)
        return {"balanced_accuracy": ba, "macro_f1": f1}

    rows, cache = [], {}
    for (v, c) in res["pred_s"]:
        for name in subsets:
            cache[(v, c, name)] = boot(v, c, name)
    for (v, c, name), mets in cache.items():
        for mn, arr in mets.items():
            val, lo, hi = _ci(arr)
            rows.append({"model": model, "classifier": c, "variant": v, "kind": "metric", "eval_set": name, "metric": mn,
                         "value": val, "ci_lo": lo, "ci_hi": hi})
            if name != "scanner_heldout":  # drop = scanner - phone, same resamples (paired)
                val, lo, hi = _ci(cache[(v, c, "scanner_heldout")][mn] - arr)
                rows.append({"model": model, "classifier": c, "variant": v, "kind": "drop", "eval_set": name, "metric": mn,
                             "value": val, "ci_lo": lo, "ci_hi": hi})
                if v != "baseline" and ("baseline", c, name) in cache:  # gain on phones vs. the same probe w/o fix
                    val, lo, hi = _ci(arr - cache[("baseline", c, name)][mn])
                    rows.append({"model": model, "classifier": c, "variant": v, "kind": "gain_vs_baseline", "eval_set": name,
                                 "metric": mn, "value": val, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


# ======================================================================================
# e. quality proxies
# ======================================================================================
def device_quality(index: pd.DataFrame) -> pd.DataFrame:
    """Per device: median sharpness (Laplacian variance), mean saturation; sharpness relative to scanner median."""
    q = index.groupby(["device", "device_type"]).agg(sharpness=("sharpness", "median"), saturation=("saturation", "mean")).reset_index()
    ref = q.loc[q.device_type == "scanner", "sharpness"].median()
    q["sharpness_rel"] = q.sharpness / ref
    ref_s = q.loc[q.device_type == "scanner", "saturation"].mean()
    q["saturation_rel"] = q.saturation - ref_s
    return q


def quality_correlation(cls: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation across phones between the scanner->phone drop (LR, baseline, balanced accuracy)
    and quality proxies; per model (n = #phones, descriptive) and pooled after removing each model's mean."""
    q = device_quality(index).set_index("device")
    d = cls[(cls.kind == "drop") & (cls.classifier == "lr") & (cls.variant == "baseline") & (cls.metric == "balanced_accuracy")
            & cls.eval_set.str.startswith("phone:")].copy()
    d["device"] = d.eval_set.str[len("phone:"):]
    d = d.join(q[["sharpness_rel", "saturation", "saturation_rel"]], on="device")
    d["drop_c"] = d.value - d.groupby("model").value.transform("mean")
    rows = []
    for proxy in ("sharpness_rel", "saturation"):
        # pooled summary: average the model-centred drop per phone -> one point per phone (n = number of phones);
        # pooling all model x phone pairs would be pseudo-replication (models share the same phones).
        pooled = d.groupby("device", as_index=False).agg(drop_c=("drop_c", "mean"), **{proxy: (proxy, "first")})
        pooled["value"] = pooled["drop_c"]
        for name, g in list(d.groupby("model")) + [("MEAN OVER MODELS (per phone)", pooled)]:
            if len(g) >= 3:
                col = "value"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # constant input -> nan
                    r, p = sps.spearmanr(g[col], g[proxy])
                rows.append({"model": name, "proxy": proxy, "spearman_rho": float(r), "p_value": float(p), "n_phones": int(g.device.nunique()), "n": len(g)})
    return pd.DataFrame(rows)


def tile_level_quality(model: str, S: np.ndarray, grid: Grid, index: pd.DataFrame) -> pd.DataFrame:
    """Per phone: Spearman between a tile's mean cosine to its group's scanner tiles and (log sharpness ratio vs.
    the scanners of the group, saturation difference).  Supplementary / descriptive."""
    ix = index.set_index("row")
    sharp = ix.sharpness.reindex(grid.rows.ravel()).values.reshape(grid.rows.shape)
    sat = ix.saturation.reindex(grid.rows.ravel()).values.reshape(grid.rows.shape)
    rows = []
    for p in grid.P:
        cs = S[:, p][:, grid.S].mean(1)
        for name, v in (("log_sharpness_ratio", np.log(np.maximum(sharp[:, p], 1e-6) / np.maximum(sharp[:, grid.S].mean(1), 1e-6))),
                        ("saturation_diff", sat[:, p] - sat[:, grid.S].mean(1))):
            r, pv = sps.spearmanr(cs, v)
            rows.append({"model": model, "device": grid.devices[p], "proxy": name, "spearman_rho": float(r), "p_value": float(pv), "n": len(cs)})
    return pd.DataFrame(rows)


# ======================================================================================
# output helpers: LaTeX + figures
# ======================================================================================
def latex_table(df: pd.DataFrame, caption: str, label: str, path: Path, fmt: str = "{:.3f}") -> None:
    """Minimal booktabs writer (no jinja2 needed)."""
    def esc(s): return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("->", r"$\to$")
    def cell(v): return esc(fmt.format(v)) if isinstance(v, (float, np.floating)) and np.isfinite(v) else esc(v)
    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\begin{{tabular}}{{l{'r' * (df.shape[1] - 1)}}}", r"\toprule",
             " & ".join(esc(c) for c in df.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(cell(v) for v in r) + r" \\" for r in df.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{esc(caption)}}}", rf"\label{{{label}}}", r"\end{table}"]
    Path(path).write_text("\n".join(lines) + "\n")


def _fmt_ci(v, lo, hi, d=3): return f"{v:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def savefig(fig, base: Path) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}", dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def make_outputs(res_dir: Path, models: list[str], index: pd.DataFrame) -> None:
    """Aggregate per-model cache files into final CSVs, LaTeX tables and figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cache = res_dir / "cache"

    def gather(prefix):
        fs = [pd.read_csv(cache / f"{prefix}__{m}.csv") for m in models if (cache / f"{prefix}__{m}.csv").exists()]
        return pd.concat(fs, ignore_index=True) if fs else None

    fig_dir = res_dir / "figures"
    tab_dir = res_dir / "tables"
    fig_dir.mkdir(exist_ok=True), tab_dir.mkdir(exist_ok=True)
    mlist = lambda df: [m for m in models if m in set(df.model)]

    pc = gather("paired")
    if pc is not None:
        pc.to_csv(res_dir / "paired_consistency.csv", index=False)
        ov = pc[pc.level == "overall"]
        t = ov.assign(v=[f"{r.median:.3f} [{r.q25:.3f}, {r.q75:.3f}]" for r in ov.itertuples()]).pivot(index="model", columns="comparison", values="v").reindex(mlist(ov)).reset_index()
        latex_table(t, "Paired cosine similarity, median [IQR], across groups and device pairs.", "tab:paired", tab_dir / "paired.tex")
        fig, ax = plt.subplots(figsize=(1 + 0.9 * len(mlist(ov)), 3.6))
        w = 0.2
        for j, comp in enumerate(["scanner-scanner", "phone-scanner", "phone-phone", "phone-scanner (different location)"]):
            d = ov[ov.comparison == comp].set_index("model").reindex(mlist(ov))
            ax.bar(np.arange(len(d)) + (j - 1.5) * w, d["median"], w, yerr=[d["median"] - d.q25, d.q75 - d["median"]], label=comp, capsize=2)
        ax.set_xticks(np.arange(len(mlist(ov))), mlist(ov), rotation=45, ha="right")
        ax.set_ylabel("cosine similarity (median, IQR)")
        ax.legend(fontsize=7)
        savefig(fig, fig_dir / "fig_paired")

    rt = gather("retrieval")
    if rt is not None:
        rt.to_csv(res_dir / "retrieval.csv", index=False)
        g = rt.groupby(["model", "kind", "protocol"])[["top1", "top5", "top10"]].mean().reset_index()
        g.to_csv(res_dir / "retrieval_summary.csv", index=False)
        t = g.pivot(index="model", columns=["kind", "protocol"], values="top1").reindex(mlist(g))
        t.columns = [f"{k} {p} top1" for k, p in t.columns]
        latex_table(t.reset_index(), "Top-1 retrieval of the same group (mean over device pairs).", "tab:retrieval", tab_dir / "retrieval.tex")
        fig, ax = plt.subplots(figsize=(1 + 0.9 * len(mlist(g)), 3.6))
        for j, (kind, proto) in enumerate([("scanner->scanner", "same_stain"), ("phone->scanner", "same_stain"), ("phone->scanner", "cross_stain")]):
            d = g[(g.kind == kind) & (g.protocol == proto)].set_index("model").reindex(mlist(g))
            ax.bar(np.arange(len(d)) + (j - 1) * 0.27, d.top1, 0.27, label=f"{kind}, {proto}")
        ax.set_xticks(np.arange(len(mlist(g))), mlist(g), rotation=45, ha="right")
        ax.set_ylabel("top-1 accuracy")
        ax.legend(fontsize=7)
        savefig(fig, fig_dir / "fig_retrieval")

    cl = gather("classification")
    if cl is not None:
        cl.to_csv(res_dir / "classification.csv", index=False)
        ba = cl[(cl.metric == "balanced_accuracy") & (cl.variant == "baseline")]
        rows = []
        for m in mlist(cl):
            for c in CLASSIFIERS:
                s = ba[(ba.model == m) & (ba.classifier == c)]
                get = lambda kind, es: s[(s.kind == kind) & (s.eval_set == es)].iloc[0]
                a, b, d = get("metric", "scanner_heldout"), get("metric", "phone_all"), get("drop", "phone_all")
                rows.append({"model": m, "probe": c, "scanner BA": a.value, "phone BA": b.value, "drop [95% CI]": _fmt_ci(d.value, d.ci_lo, d.ci_hi)})
        t = pd.DataFrame(rows)
        t.to_csv(res_dir / "classification_main.csv", index=False)
        latex_table(t, "Tissue probe (46 classes, group-disjoint 5-fold): balanced accuracy and scanner-to-phone drop.", "tab:probe", tab_dir / "probe.tex")
        # drop figure
        fig, axs = plt.subplots(1, 2, figsize=(8, 0.5 + 0.35 * len(mlist(cl))), sharey=True)
        for ax, c in zip(axs, CLASSIFIERS):
            d = ba[(ba.classifier == c) & (ba.kind == "drop") & (ba.eval_set == "phone_all")].set_index("model").reindex(mlist(cl))
            y = np.arange(len(d))
            ax.errorbar(d.value, y, xerr=[d.value - d.ci_lo, d.ci_hi - d.value], fmt="o", capsize=3)
            ax.axvline(0, color="k", lw=0.5)
            ax.set_yticks(y, d.index)
            ax.set_title(f"{c.upper()} probe")
            ax.set_xlabel("balanced-accuracy drop (scanner - phone)")
        savefig(fig, fig_dir / "fig_drop")
        # fixes
        fx = cl[(cl.metric == "balanced_accuracy") & (cl.kind == "metric") & (cl.eval_set == "phone_all")]
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
        for ax, c in zip(axs, CLASSIFIERS):
            d = fx[fx.classifier == c].pivot(index="model", columns="variant", values="value").reindex(mlist(cl))
            d = d[[v for v in VARIANTS if v in d]]
            d.plot.bar(ax=ax, width=0.85, legend=(c == "lr"))
            ax.set_title(f"{c.upper()} probe: phone balanced accuracy")
            ax.set_xlabel("")
        savefig(fig, fig_dir / "fig_fixes")
        g = fx[fx.classifier == "lr"].pivot(index="model", columns="variant", values="value").reindex(mlist(cl))
        latex_table(g[[v for v in VARIANTS if v in g]].reset_index(), "Phone balanced accuracy (LR probe) with cheap fixes.", "tab:fixes", tab_dir / "fixes.tex")
        # per phone heatmap
        pp = cl[(cl.kind == "drop") & (cl.classifier == "lr") & (cl.variant == "baseline") & (cl.metric == "balanced_accuracy") & cl.eval_set.str.startswith("phone:")]
        H = pp.assign(device=pp.eval_set.str[6:]).pivot(index="model", columns="device", values="value").reindex(mlist(cl))
        H.to_csv(res_dir / "per_phone_drop.csv")
        latex_table(H.reset_index(), "Balanced-accuracy drop per phone (LR probe).", "tab:perphone", tab_dir / "per_phone.tex")
        fig, ax = plt.subplots(figsize=(1.5 + 0.9 * H.shape[1], 0.8 + 0.35 * len(H)))
        im = ax.imshow(H.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(H.shape[1]), [c[:16] for c in H.columns], rotation=45, ha="right")
        ax.set_yticks(range(H.shape[0]), H.index)
        fig.colorbar(im, ax=ax, label="BA drop")
        savefig(fig, fig_dir / "fig_per_phone")
        qc = quality_correlation(cl, index)
        qc.to_csv(res_dir / "quality_correlation.csv", index=False)
        latex_table(qc, "Spearman correlation between per-phone drop and image-quality proxies.", "tab:quality", tab_dir / "quality.tex")
        q = device_quality(index).set_index("device")
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        for m in mlist(cl):
            if m in H.index:
                ax.scatter(q.loc[H.columns, "sharpness_rel"], H.loc[m], s=14, label=m)
        ax.set_xscale("log")
        ax.set_xlabel("median sharpness / scanner median")
        ax.set_ylabel("BA drop")
        if len(H) <= 12:
            ax.legend(fontsize=6)
        savefig(fig, fig_dir / "fig_quality")
    tq = gather("tilequality")
    if tq is not None:
        tq.to_csv(res_dir / "tile_quality_correlation.csv", index=False)
    print(f"[outputs] wrote CSV/LaTeX/figures to {res_dir}")


# ======================================================================================
# driver
# ======================================================================================
def _cached(res_dir: Path, prefix: str, model: str, fn) -> pd.DataFrame:
    p = res_dir / "cache" / f"{prefix}__{model}.csv"
    if p.exists():
        print(f"[{model}] {prefix}: cached")
        return pd.read_csv(p)
    df = fn()
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return df


def analyse_model(emb_dir, res_dir, model: str, index: pd.DataFrame, analyses="abcde", boot=1000, folds=5, seed=0,
                  max_train=None, lr_iter=200, variants=VARIANTS) -> None:
    res_dir = Path(res_dir)
    E = load_embeddings(emb_dir, model, len(index))
    grid = build_grid(index)
    need_pairs = any(a in analyses for a in "ae")
    if need_pairs:
        S, neg = paired_similarities(E, grid)
    if "a" in analyses:
        _cached(res_dir, "paired", model, lambda: summarise_paired(model, S, neg, grid))
    if "e" in analyses:
        _cached(res_dir, "tilequality", model, lambda: tile_level_quality(model, S, grid, index))
    if "b" in analyses:
        _cached(res_dir, "retrieval", model, lambda: retrieval(E, grid, model))
    if any(a in analyses for a in "cde"):
        vs = list(variants) if "d" in analyses else ["baseline"]
        cp = res_dir / "cache" / f"classification__{model}.csv"
        if cp.exists() and not set(vs) <= set(pd.read_csv(cp, usecols=["variant"]).variant):
            print(f"[{model}] cached classification lacks requested variants -> recomputing")
            cp.unlink()

        def run():
            print(f"[{model}] probes ({', '.join(vs)}) ...", flush=True)
            r = run_probe_protocol(E, index, variants=vs, n_splits=folds, seed=seed, max_train=max_train, lr_iter=lr_iter)
            return summarise_probe(model, r, boot, seed)
        _cached(res_dir, "classification", model, run)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--models", default=None, help="comma list of embedding file stems (default: every *.npy in --emb)")
    ap.add_argument("--analyses", default="abcde")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=None, help="cap on scanner training tiles per fold (speed)")
    ap.add_argument("--lr-iter", type=int, default=200)
    a = ap.parse_args(argv)
    emb, res = Path(a.emb), Path(a.results)
    res.mkdir(parents=True, exist_ok=True)
    index = load_index(emb)
    models = a.models.split(",") if a.models else sorted(p.stem for p in emb.glob("*.npy") if not p.stem.endswith(".tmp"))
    print(f"[analysis] models: {models}")
    for m in models:
        analyse_model(emb, res, m, index, a.analyses, a.boot, a.folds, a.seed, a.max_train, a.lr_iter)
    make_outputs(res, models, index)


if __name__ == "__main__":
    main()
