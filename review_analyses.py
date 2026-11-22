"""Additional analyses requested in review (run on Kaggle CPU, ~1.5-2 h).

  python review_analyses.py --out /kaggle/working/review [--seeds 3] [--quick]

Finds index.csv and <model>.npy under /kaggle/input automatically (or pass --emb DIR [DIR ...]).
Writes CSVs to --out and prints a summary:
  1. grouping_counts.csv   - groups vs loc_ids, how many loc_ids span several stains
  2. within_stain.csv      - sensitivity analysis: probes trained and tested within one stain, folds grouped by
                             spatially overlapping fields (connected components of overlapping boxes) -> no field or
                             overlapping field can be on both sides; pooled over the 13 stains, all 9 encoders
  3. paired_adaptation.csv - UNI2-h (no adaptation) vs H-optimus-0 (10-shot) on identical folds, paired location
                             bootstrap of the phone-BA difference, repeated over seeds (new fold split and new
                             few-shot draw per seed)
"""
import argparse, glob, sys, time
from pathlib import Path
import numpy as np, pandas as pd

_here = Path(__file__).resolve().parent
_cands = [_here] + [Path(p).parent for p in glob.glob("/kaggle/input/**/analysis.py", recursive=True) + glob.glob("/kaggle/working/**/analysis.py", recursive=True)]
sys.path.insert(0, str(next((c for c in _cands if (c / "analysis.py").exists()), _here)))
from analysis import run_probe_protocol  # noqa: E402
import warnings
from sklearn.linear_model import LogisticRegression


def lr_predict(Xtr, ytr, Xtes, C=1.0, max_iter=200, seed=0):
    """Same probe as analysis.lr_predict, but returns labels directly (safe when a class is absent from training)."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(C=C, max_iter=max_iter, tol=1e-3, random_state=seed).fit((Xtr - mu) / sd, ytr)
    return [clf.predict((X - mu) / sd) for X in Xtes]

MODELS = ["uni2h", "gigapath", "hoptimus0", "uni", "virchow2", "phikon2", "dinov2_l", "conch", "vitb16_in"]


def find_inputs(emb_dirs):
    files = {}
    roots = emb_dirs or ["/kaggle/input", "/kaggle/working"]
    for m in MODELS:
        hits = [h for r in roots for h in glob.glob(f"{r}/**/{m}.npy", recursive=True)]
        if hits:
            files[m] = Path(sorted(hits)[0])
    if "uni2h" not in files:
        raise SystemExit(f"uni2h.npy not found under {roots}")
    ref = pd.read_csv(files["uni2h"].parent / "index.csv")
    for m, f in files.items():  # every embedding file must share the reference row order
        ix = pd.read_csv(f.parent / "index.csv")
        assert len(ix) == len(ref) and (ix.path.values == ref.path.values).all(), f"index order differs for {m}"
    return ref, files


def labelled(index):
    return index[index.tissue.notna() & index.device_type.isin(["phone", "scanner"])].reset_index(drop=True)


def bal_acc(y, p, w=None):
    w = np.ones(len(y)) if w is None else w
    acc = []
    for c in np.unique(y):
        m = y == c
        if w[m].sum() > 0:
            acc.append((w[m] * (p[m] == c)).sum() / w[m].sum())
    return float(np.mean(acc))


def boot_ci(y, p, grp, B, rng, y2=None, p2=None):
    """Percentile CI of BA (or of BA(p) - BA(p2) on the same tiles) resampling groups with replacement."""
    g, gi = np.unique(grp, return_inverse=True)
    vals = []
    for _ in range(B):
        mult = np.bincount(rng.integers(0, len(g), len(g)), minlength=len(g))
        w = mult[gi].astype(float)
        v = bal_acc(y, p, w) if p2 is None else bal_acc(y, p, w) - bal_acc(y2, p2, w)
        vals.append(v)
    return np.percentile(vals, [2.5, 97.5])


# ---------------------------------------------------------------- 1. grouping counts
def grouping_counts(ix):
    per_loc = ix.groupby("loc_id").stain.nunique()
    out = {"tiles": len(ix), "groups": ix.group_id.nunique(), "loc_ids": ix.loc_id.nunique(),
           "loc_ids_spanning_>1_stain": int((per_loc > 1).sum()),
           "groups_in_multistain_loc_ids": int(ix[ix.loc_id.isin(per_loc[per_loc > 1].index)].group_id.nunique()),
           "stains": ix.stain.nunique(), "mean_groups_per_stain": ix.groupby("stain").group_id.nunique().mean()}
    return pd.DataFrame([out])


# ---------------------------------------------------------------- 2. within-stain sensitivity
def spatial_components(ix):
    """Connected components of groups whose field boxes overlap (share area), computed within each stain."""
    g = ix.groupby("group_id").agg(stain=("stain", "first"), x0=("x0", "first"), y0=("y0", "first"),
                                   x1=("x1", "first"), y1=("y1", "first")).reset_index()
    comp = {}
    for st, d in g.groupby("stain"):
        ids, b = d.group_id.values, d[["x0", "y0", "x1", "y1"]].values.astype(float)
        parent = list(range(len(ids)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        ov = (b[:, None, 0] < b[None, :, 2]) & (b[None, :, 0] < b[:, None, 2]) & \
             (b[:, None, 1] < b[None, :, 3]) & (b[None, :, 1] < b[:, None, 3])  # shared area > 0
        for i, j in zip(*np.where(np.triu(ov, 1))):
            parent[find(i)] = find(j)
        for i, gid in enumerate(ids):
            comp[gid] = f"{st}#{find(i)}"
    return ix.group_id.map(comp).values


def within_stain(ix, E, comp, seed=0, n_splits=5):
    from sklearn.model_selection import GroupKFold
    y = pd.factorize(ix.tissue)[0]
    is_ph = (ix.device_type == "phone").values
    pred = np.full(len(ix), -1)
    for st in ix.stain.unique():
        rows = np.where(ix.stain.values == st)[0]
        k = min(n_splits, len(np.unique(comp[rows])))
        if k < 2:
            print(f"  [skip] stain {st}: <2 spatial components", flush=True)
            continue
        try:
            gkf = GroupKFold(n_splits=k, shuffle=True, random_state=seed)
        except TypeError:
            gkf = GroupKFold(n_splits=k)
        for tr_g, te_g in gkf.split(rows, groups=comp[rows]):
            tr = rows[tr_g][~is_ph[rows[tr_g]]]  # train on scanner tiles only
            te = rows[te_g]
            pred[te] = lr_predict(E[ix.row.values[tr]], y[tr], [E[ix.row.values[te]]], seed=seed)[0]
    return y, pred, is_ph


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", nargs="*", default=None)
    ap.add_argument("--out", default="/kaggle/working/review")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--quick", action="store_true", help="2 encoders, 1 seed, 100 bootstraps (smoke test)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    index, files = find_inputs(a.emb)
    ix = labelled(index)
    models = [m for m in MODELS if m in files]
    if a.quick:
        models, a.seeds, a.boot = models[:2], 1, 100
    rng = np.random.default_rng(0)

    gc = grouping_counts(ix); gc.to_csv(out / "grouping_counts.csv", index=False)
    print("== 1. grouping\n", gc.T.to_string(), flush=True)

    comp = spatial_components(ix)
    sz = pd.Series(comp).groupby(comp).size() / ix.device.nunique()
    print(f"== 2. within-stain: {len(sz)} spatial components from {ix.group_id.nunique()} groups; "
          f"groups per component: median {sz.median():.0f}, max {sz.max():.0f}", flush=True)
    pd.DataFrame({"component": sz.index, "groups": sz.values}).to_csv(out / "spatial_components.csv", index=False)
    rows = []
    for m in models:
        t0 = time.time()
        E = np.load(files[m]).astype(np.float32)
        y, p, is_ph = within_stain(ix, E, comp)
        ok = p >= 0
        s, ph = ok & ~is_ph, ok & is_ph
        ba_s, ba_p = bal_acc(y[s], p[s]), bal_acc(y[ph], p[ph])
        ci = boot_ci(y[s], p[s], comp[s], a.boot, rng)  # scanner BA CI
        # drop CI: resample components, recompute both BAs
        g, gi = np.unique(comp[ok], return_inverse=True)
        ys, ps, phm = y[ok], p[ok], is_ph[ok]
        drops = []
        for _ in range(a.boot):
            w = np.bincount(rng.integers(0, len(g), len(g)), minlength=len(g))[gi].astype(float)
            drops.append(bal_acc(ys[~phm], ps[~phm], w[~phm]) - bal_acc(ys[phm], ps[phm], w[phm]))
        lo, hi = np.percentile(drops, [2.5, 97.5])
        rows.append({"model": m, "scanner_BA": ba_s, "phone_BA": ba_p, "drop": ba_s - ba_p, "drop_lo": lo, "drop_hi": hi,
                     "scanner_BA_lo": ci[0], "scanner_BA_hi": ci[1]})
        print(f"  {m}: scanner {ba_s:.3f}  phone {ba_p:.3f}  drop {ba_s-ba_p:.3f} [{lo:.3f}, {hi:.3f}]  ({time.time()-t0:.0f}s)",
              flush=True)
        del E
    ws = pd.DataFrame(rows); ws.to_csv(out / "within_stain.csv", index=False)
    print(ws.round(3).to_string(index=False), flush=True)

    print("== 3. paired adaptation comparison: UNI2-h (none) vs H-optimus-0 (10-shot)", flush=True)
    if "uni2h" in files and "hoptimus0" in files:
        Eu, Eh = np.load(files["uni2h"]).astype(np.float32), np.load(files["hoptimus0"]).astype(np.float32)
        prow = []
        for sd in range(a.seeds):
            t0 = time.time()
            ru = run_probe_protocol(Eu, index, variants=("baseline",), classifiers=("lr",), seed=sd, verbose=False)
            rh = run_probe_protocol(Eh, index, variants=("baseline", "fewshot10"), classifiers=("lr",), seed=sd, verbose=False)
            assert (ru["loc_p"] == rh["loc_p"]).all() and (ru["fold"] == rh["fold"]).all()
            yp, lp = ru["y_p"], ru["loc_p"]
            pu, ph10, ph0 = ru["pred_p"][("baseline", "lr")], rh["pred_p"][("fewshot10", "lr")], rh["pred_p"][("baseline", "lr")]
            k = (pu >= 0) & (ph10 >= 0)
            d = bal_acc(yp[k], pu[k]) - bal_acc(yp[k], ph10[k])
            lo, hi = boot_ci(yp[k], pu[k], lp[k], a.boot, rng, y2=yp[k], p2=ph10[k])
            prow.append({"seed": sd, "uni2h_none": bal_acc(yp[k], pu[k]), "hoptimus0_none": bal_acc(yp[k], ph0[k]),
                         "hoptimus0_10shot": bal_acc(yp[k], ph10[k]), "diff_uni2h_minus_h0_10shot": d,
                         "diff_lo": lo, "diff_hi": hi})
            print(f"  seed {sd}: UNI2-h {prow[-1]['uni2h_none']:.3f}  H0 10-shot {prow[-1]['hoptimus0_10shot']:.3f}  "
                  f"diff {d:.3f} [{lo:.3f}, {hi:.3f}]  ({time.time()-t0:.0f}s)", flush=True)
        pa = pd.DataFrame(prow); pa.to_csv(out / "paired_adaptation.csv", index=False)
        print(pa.round(4).to_string(index=False))
        print("  10-shot H-optimus-0 across seeds: mean %.3f, range %.3f-%.3f" %
              (pa.hoptimus0_10shot.mean(), pa.hoptimus0_10shot.min(), pa.hoptimus0_10shot.max()))
    print("done ->", out)


if __name__ == "__main__":
    main()
