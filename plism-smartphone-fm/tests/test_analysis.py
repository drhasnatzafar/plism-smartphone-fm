"""Metrics sanity, leakage-free splits and probe protocol on synthetic embeddings."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score

import analysis as A
import synth


@pytest.fixture(scope="module")
def small():
    df, E = synth.make_embeddings(n_stains=3, n_locs=40, dim=32)
    return df, A.l2norm(E)


def identical_across_devices(n_stains=3, n_locs=30, dim=32, seed=0):
    df, _ = synth.make_embeddings(n_stains, n_locs, dim)
    rng = np.random.default_rng(seed)
    per_group = {g: rng.standard_normal(dim) for g in df.group_id.unique()}
    return df, A.l2norm(np.stack([per_group[g] for g in df.group_id]).astype(np.float32))


def test_grid_complete(small):
    df, _ = small
    g = A.build_grid(df)
    assert g.rows.shape == (3 * 40, 13) and g.is_phone.sum() == 6 and len(g.S) == 7
    assert len(np.unique(g.rows)) == g.rows.size  # every tile appears once


def test_identical_embeddings_give_cos_one_and_perfect_retrieval():
    df, E = identical_across_devices()
    g = A.build_grid(df)
    S, neg = A.paired_similarities(E, g)
    np.testing.assert_allclose(S, 1.0, atol=1e-5)
    assert neg.max() < 0.9  # other-location negatives are clearly lower
    summ = A.summarise_paired("m", S, neg, g)
    ov = summ[summ.level == "overall"].set_index("comparison")
    assert ov.loc["phone-scanner", "median"] == pytest.approx(1.0, abs=1e-5)
    assert ov.loc["phone-scanner", "q25"] == pytest.approx(ov.loc["phone-scanner", "q75"], abs=1e-5)
    r = A.retrieval(E, g, "m")
    assert (r[["top1", "top5", "top10"]] == 1.0).all().all()
    assert set(r.kind) == {"phone->scanner", "scanner->scanner"} and set(r.protocol) == {"same_stain", "cross_stain"}
    assert len(r) == (6 * 7 + 7 * 6) * 2


def test_collapsed_embeddings_do_not_get_free_retrieval(small):
    df, _ = small
    g = A.build_grid(df)
    E = np.tile(A.l2norm(np.ones((1, 8), np.float32)), (len(df), 1))
    r = A.retrieval(E, g, "m")
    assert r[r.protocol == "cross_stain"].top1.max() < 0.1  # ties count half, so chance level


def test_random_embeddings_retrieval_near_chance():
    df, _ = synth.make_embeddings(3, 40, 32)
    E = A.l2norm(np.random.default_rng(0).standard_normal((len(df), 32)).astype(np.float32))
    r = A.retrieval(E, A.build_grid(df), "m")
    assert r[r.protocol == "same_stain"].top1.mean() < 0.1  # chance = 1/40
    assert r[r.protocol == "same_stain"].top10.mean() < 0.45  # chance = 0.25


def test_retrieval_phone_worse_than_scanner(small):
    df, E = small
    r = A.retrieval(E, A.build_grid(df), "m")
    m = r.groupby(["kind", "protocol"]).top1.mean()
    assert m["scanner->scanner", "same_stain"] > m["phone->scanner", "same_stain"] > 1 / 40
    assert m["phone->scanner", "cross_stain"] <= m["phone->scanner", "same_stain"]


def test_paired_ordering(small):
    df, E = small
    g = A.build_grid(df)
    S, neg = A.paired_similarities(E, g)
    ov = A.summarise_paired("m", S, neg, g).query("level=='overall'").set_index("comparison")["median"]
    assert ov["scanner-scanner"] > ov["phone-scanner"] > ov["phone-scanner (different location)"]
    assert S.shape == (len(g.rows), 13, 13)


def test_folds_have_no_leakage(small):
    df, E = small
    loc, _ = pd.factorize(df.loc_id)
    fold = A.make_folds(loc, 5, 0)
    assert set(fold) == set(range(5))
    for f in range(5):
        te, tr = df[fold == f], df[fold != f]
        assert set(te.loc_id).isdisjoint(tr.loc_id)
        assert set(te.group_id).isdisjoint(tr.group_id)
    # same loc -> same fold regardless of device / stain
    assert (pd.Series(fold).groupby(loc).nunique() == 1).all()


def test_split_prevents_memorisation_of_locations(small):
    """Embeddings = per-location random vector + labels random per location: any leakage would give ~100%."""
    df, _ = small
    rng = np.random.default_rng(1)
    locs = df.loc_id.unique()
    vec = {l: rng.standard_normal(32) for l in locs}
    E = A.l2norm(np.stack([vec[l] for l in df.loc_id]).astype(np.float32) + 0.01 * rng.standard_normal((len(df), 32)).astype(np.float32))
    lab = {l: f"c{rng.integers(0, 6)}" for l in locs}
    d2 = df.copy()
    d2["tissue"] = d2.loc_id.map(lab)
    res = A.run_probe_protocol(E, d2, variants=["baseline"], verbose=False)
    t = A.summarise_probe("m", res, B=50)
    ba = t[(t.metric == "balanced_accuracy") & (t.kind == "metric") & (t.eval_set == "scanner_heldout")]
    assert ba.value.max() < 0.45  # chance = 1/6 (+ noise); leaky split would be ~1.0
    # every row got a prediction exactly once
    assert all((p >= 0).all() for p in res["pred_s"].values()) and all((p >= 0).all() for p in res["pred_p"].values())


def test_counts_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 400)
    p = np.where(rng.random(400) < 0.6, y, rng.integers(0, 5, 400))
    loc = rng.integers(0, 20, 400)
    tp, sup, prd = A.class_counts(y, p, loc, 20, 5)
    ba, f1 = A.metrics_from_counts(tp.sum(0), sup.sum(0), prd.sum(0))
    assert ba == pytest.approx(balanced_accuracy_score(y, p)) and f1 == pytest.approx(f1_score(y, p, average="macro"))
    M = A.bootstrap_weights(20, 100, 0)
    assert M.shape == (101, 20) and (M[1:].sum(1) == 20).all() and (M[0] == 1).all()
    bab, _ = A.metrics_from_counts(M @ tp, M @ sup, M @ prd)
    assert bab.shape == (101,) and bab[0] == pytest.approx(ba)


def test_knn_and_coral_helpers():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 16)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    assert (A.knn_predict(X, y, X, k=1, n_cls=2) == y).all()
    np.testing.assert_allclose(A.coral_matrix(X, X, 0.1), np.eye(16), atol=1e-3)
    T = X @ np.diag(np.linspace(0.5, 2, 16)).astype(np.float32)
    Xa = X @ A.coral_matrix(X, T, 0.0)
    np.testing.assert_allclose(np.cov(Xa, rowvar=False), np.cov(T, rowvar=False), atol=1e-2)


def test_probe_protocol_drop_and_fixes(small):
    df, E = small
    res = A.run_probe_protocol(E, df, n_splits=5, verbose=False)
    t = A.summarise_probe("m", res, B=100)
    g = lambda **k: t.query(" and ".join(f"{a}=='{b}'" for a, b in k.items())).iloc[0]
    d = g(classifier="lr", variant="baseline", kind="drop", eval_set="phone_all", metric="balanced_accuracy")
    assert d.value > 0.02 and d.ci_lo <= d.value <= d.ci_hi and d.ci_lo > 0  # phones are really harder here
    assert g(classifier="lr", variant="center", kind="gain_vs_baseline", eval_set="phone_all", metric="balanced_accuracy").value > 0
    assert g(classifier="lr", variant="fewshot10", kind="gain_vs_baseline", eval_set="phone_all", metric="balanced_accuracy").value > 0
    assert set(t.eval_set) >= {"scanner_heldout", "phone_all"} and sum(e.startswith("phone:") for e in set(t.eval_set)) == 6
    assert {"metric", "drop", "gain_vs_baseline"} == set(t.kind)
    assert not t.value.isna().any()


def test_end_to_end_outputs(small, tmp_path):
    df, E = small
    emb = tmp_path / "emb"
    emb.mkdir()
    df.to_csv(emb / "index.csv", index=False)
    np.save(emb / "toyA.npy", E.astype(np.float16))
    np.save(emb / "toyB.npy", A.l2norm(E + 0.5 * np.random.default_rng(3).standard_normal(E.shape).astype(np.float32)).astype(np.float16))
    res = tmp_path / "res"
    A.main(["--emb", str(emb), "--results", str(res), "--boot", "30", "--lr-iter", "50"])
    for f in ["paired_consistency.csv", "retrieval.csv", "classification.csv", "quality_correlation.csv", "per_phone_drop.csv"]:
        assert (res / f).stat().st_size > 0, f
    for f in ["paired", "retrieval", "probe", "fixes", "per_phone", "quality"]:
        assert "\\toprule" in (res / "tables" / f"{f}.tex").read_text()
    for f in ["fig_paired", "fig_retrieval", "fig_drop", "fig_fixes", "fig_per_phone", "fig_quality"]:
        assert (res / "figures" / f"{f}.png").exists() and (res / "figures" / f"{f}.pdf").exists()
    A.main(["--emb", str(emb), "--results", str(res), "--boot", "30"])  # second run: everything cached
