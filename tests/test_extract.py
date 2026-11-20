"""Streaming extraction, index correctness, resume, metadata mapping, stain normalisation."""
import io
import json
import tarfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import encoders
import extract
import imageproc
import plism_data as pdata
import synth


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    tar, csv = synth.make_tar(root / "in")
    return tar, csv


def run(data, out, *extra):
    tar, csv = data
    extract.main(["--tar", str(tar), "--csv", str(csv), "--out", str(out), "--batch", "16", "--workers", "3", *extra])


def tar_order(tar):
    with tarfile.open(tar, "r:gz") as t:
        return [m.name for m in t if m.isfile()]


def test_index_correct(data, tmp_path):
    run(data, tmp_path, "--model", "debug")
    ix = pd.read_csv(tmp_path / "index.csv")
    assert len(ix) == 3 * 13 * 6
    assert list(ix.path) == tar_order(data[0])  # row order == tar stream order
    assert (ix.device_type.value_counts().to_dict()) == {"scanner": 3 * 7 * 6, "phone": 3 * 6 * 6}
    assert ix.group_id.nunique() == 3 * 6 and (ix.groupby("group_id").size() == 13).all()
    assert (ix.groupby("group_id").device.nunique() == 13).all()
    # tissue came from the CSV (non-default column names) and is constant within a location
    assert ix.tissue.notna().all() and (ix.groupby("group_id").tissue.nunique() == 1).all()
    row = ix.iloc[7]
    assert row.path.endswith(f"{row.x0}_{row.y0}_{row.x1}_{row.y1}.png") and row.stain in row.path
    assert np.isfinite(ix[["sharpness", "saturation"]].values).all()
    assert ix.loc_id.nunique() == 6  # tissue|coords merges the 3 stains
    assert (tmp_path / "lab_stats.json").exists()


def test_embeddings_aligned_with_index(data, tmp_path):
    run(data, tmp_path, "--model", "debug")
    E = np.load(tmp_path / "debug.npy")
    assert E.dtype == np.float16 and E.shape == (234, 64)
    enc = encoders.get_encoder("debug")
    with tarfile.open(data[0], "r:gz") as t:  # independent, non-streaming recomputation
        for i, m in enumerate(x for x in t if x.isfile()):
            if i % 37:
                continue
            x = enc.preprocess_one(pdata.decode_rgb(t.extractfile(m).read()))
            np.testing.assert_allclose(E[i], enc.encode(x[None])[0], atol=2e-2, rtol=1e-2)


def test_skip_and_limit(data, tmp_path):
    run(data, tmp_path / "a", "--model", "debug")
    mtime = (tmp_path / "a" / "debug.npy").stat().st_mtime_ns
    run(data, tmp_path / "a", "--model", "debug")  # finished models are skipped
    assert (tmp_path / "a" / "debug.npy").stat().st_mtime_ns == mtime
    run(data, tmp_path / "b", "--model", "debug", "--limit", "20")
    assert np.load(tmp_path / "b" / "debug.npy").shape[0] == 20 == len(pd.read_csv(tmp_path / "b" / "index.csv"))
    with pytest.raises(RuntimeError, match="limit"):  # smoke-test index must not silently be reused for a full run
        run(data, tmp_path / "b", "--model", "debug")


def test_resume_after_crash(data, tmp_path, monkeypatch):
    run(data, tmp_path / "clean", "--model", "debug")
    tar, csv = data
    out = tmp_path / "crash"
    out.mkdir()
    index = pdata.ensure_index(tar, csv, out)
    calls = {"n": 0}
    orig = encoders.DebugEncoder.encode

    def flaky(self, b):
        calls["n"] += 1
        if calls["n"] == 8:
            raise KeyboardInterrupt("simulated session death")
        return orig(self, b)

    monkeypatch.setattr(encoders.DebugEncoder, "encode", flaky)
    with pytest.raises(KeyboardInterrupt):
        extract.extract_model(tar, index, out, "debug", batch=16, chunk_rows=32, workers=2)
    assert not (out / "debug.npy").exists()
    parts = sorted((out / ".parts" / "debug").glob("part*.npy"))
    assert parts, "partial chunks must survive the crash"
    monkeypatch.setattr(encoders.DebugEncoder, "encode", orig)
    seen = {"first": None}
    orig_it = pdata.iter_tar_images

    def spy(tar_, skip=0, limit=None):
        seen["first"] = skip
        return orig_it(tar_, skip=skip, limit=limit)

    monkeypatch.setattr(pdata, "iter_tar_images", spy)
    extract.extract_model(tar, index, out, "debug", batch=16, chunk_rows=32, workers=2)
    assert seen["first"] and seen["first"] > 0  # resumed, did not restart
    np.testing.assert_array_equal(np.load(out / "debug.npy"), np.load(tmp_path / "clean" / "debug.npy"))
    assert not (out / ".parts" / "debug").exists()


def test_stain_norm_flag(data, tmp_path):
    run(data, tmp_path, "--model", "debug")
    run(data, tmp_path, "--model", "debug", "--stain-norm", "reinhard")
    a, b = np.load(tmp_path / "debug.npy"), np.load(tmp_path / "debug__reinhard.npy")
    assert a.shape == b.shape and not np.allclose(a, b)


def test_reinhard_matches_reference_stats():
    rng = np.random.default_rng(0)
    src = np.clip(rng.normal([200, 120, 170], 20, (64, 64, 3)), 0, 255).astype(np.uint8)
    ref = np.clip(rng.normal([150, 80, 140], 35, (64, 64, 3)), 0, 255).astype(np.uint8)
    rm, rs = imageproc.lab_stats(ref, stride=1, mask=False)
    out = imageproc.reinhard_normalise(src, rm, rs)
    om, osd = imageproc.lab_stats(out, stride=1, mask=False)
    assert np.abs(om - rm).max() < 3 and np.abs(osd - rs).max() < 3
    rt = imageproc.lab_to_rgb(imageproc.rgb_to_lab(src))  # Lab round trip
    assert np.abs(rt.astype(int) - src.astype(int)).max() <= 1


def test_quality_proxies():
    flat = np.full((32, 32, 3), 128, np.uint8)
    noisy = np.random.default_rng(0).integers(0, 255, (32, 32, 3)).astype(np.uint8)
    assert imageproc.laplacian_variance(flat) == 0 < imageproc.laplacian_variance(noisy)
    grey, red = np.full((8, 8, 3), 100, np.uint8), np.zeros((8, 8, 3), np.uint8)
    red[..., 0] = 200
    assert imageproc.mean_saturation(grey) == 0 and imageproc.mean_saturation(red) == pytest.approx(1.0)


def test_csv_colmap_override_and_no_csv(data, tmp_path):
    tar, csv = data
    raw = pd.read_csv(csv).rename(columns={"Tissue type": "Organ system label"})
    p = tmp_path / "meta2.csv"
    raw.to_csv(p, index=False)
    out = tmp_path / "o1"
    extract.main(["--tar", str(tar), "--csv", str(p), "--out", str(out), "--model", "debug", "--limit", "30",
                  "--colmap", json.dumps({"tissue": "Organ system label"})])
    assert pd.read_csv(out / "index.csv").tissue.notna().all()
    out = tmp_path / "o2"  # no CSV at all: stain/device/coords from the tar path, tissue missing
    extract.main(["--tar", str(tar), "--out", str(out), "--model", "debug", "--limit", "30"])
    ix = pd.read_csv(out / "index.csv")
    assert ix.tissue.isna().all() and ix.stain.str.startswith("stain").all() and ix.x0.notna().all()
    assert set(ix.device_type) <= {"phone", "scanner", "unknown"}


def test_device_type_rules():
    for d in synth.PHONES:
        assert pdata.device_type(d) == "phone" == pdata.device_type(synth.folder(d))
    for d in synth.SCANNERS:
        assert pdata.device_type(d) == "scanner" == pdata.device_type(synth.folder(d))
    assert pdata.device_type("weird", {"weird": "phone"}) == "phone"


def test_preprocess_shapes_all_models():
    rgb = np.random.default_rng(0).integers(0, 255, (512, 512, 3)).astype(np.uint8)
    for k in encoders.list_models():
        enc = encoders.get_encoder(k)
        x = enc.preprocess_one(rgb)
        assert x.shape == (3, enc.spec.crop, enc.spec.crop) and x.dtype == np.float32 and np.isfinite(x).all()
