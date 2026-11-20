#!/usr/bin/env python
"""Stream PLISM-sm from the .tar.gz (never extracted) and write one embedding file per model.

    python extract.py --tar /kaggle/input/.../41397708 --csv /kaggle/input/.../41397675 \
                      --model uni --out /kaggle/working/emb [--batch 64] [--limit N] [--stain-norm reinhard]

Outputs in ``--out``:
    index.csv              row order of ALL embedding files (built once, reused; also image-quality proxies)
    lab_stats.json         per-device Lab mean/std (Reinhard reference statistics)
    <model>.npy            float16, N x D, rows aligned with index.csv  (``<model>__reinhard.npy`` with stain norm)
    <model>.json           small provenance record

Resumability: finished models are skipped; an interrupted model resumes from ``.parts/<name>/`` chunks
(re-streaming the tar, but without decoding/encoding the already finished prefix).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import imageproc
import plism_data as pd_
from encoders import get_encoder, list_models

CHUNK_ROWS = 4096  # rows per resumable part file


def out_name(model: str, stain_norm: str) -> str:
    return model if stain_norm == "none" else f"{model}__{stain_norm}"


def pick_reference(stats: dict, ref_device: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Reference Lab statistics = stats of the (unique) device whose name contains ``ref_device``."""
    hits = [d for d in stats if pd_.norm(ref_device) in pd_.norm(d)]
    if len(hits) != 1:
        raise SystemExit(f"--stain-ref {ref_device!r} must match exactly one device in {sorted(stats)}; got {hits}")
    s = stats[hits[0]]
    return np.array(s["mean"], np.float32), np.array(s["std"], np.float32), hits[0]


def extract_model(tar, index, out_dir: Path, model: str, batch: int = 64, stain_norm: str = "none",
                  stain_ref: str = "S360", workers: int = 8, limit: int | None = None, overwrite: bool = False,
                  device: str | None = None, chunk_rows: int = CHUNK_ROWS) -> Path | None:
    name = out_name(model, stain_norm)
    final = out_dir / f"{name}.npy"
    if final.exists() and not overwrite:
        print(f"[{name}] already done -> skip ({final})")
        return final
    n = len(index) if limit is None else min(limit, len(index))
    parts_dir = out_dir / ".parts" / name
    if overwrite and parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    parts = sorted(parts_dir.glob("part*.npy"))
    done = sum(int(np.load(p, mmap_mode="r").shape[0]) for p in parts)
    if done:
        print(f"[{name}] resuming from row {done}/{n} ({len(parts)} parts)")

    enc = get_encoder(model)
    ref = None
    if stain_norm == "reinhard":
        stats = json.loads((out_dir / "lab_stats.json").read_text())
        mu, sd, ref_dev = pick_reference(stats, stain_ref)
        ref = (mu, sd)
        print(f"[{name}] Reinhard reference device: {ref_dev}")
    elif stain_norm != "none":
        raise SystemExit(f"unknown --stain-norm {stain_norm}")

    if done < n:
        enc.load(device=device)
        paths = index["path"].values

        def work(item):  # runs in worker threads: decode (+ stain norm) + model preprocessing
            i, member, data = item
            if paths[i] != member:
                raise RuntimeError(f"tar order differs from index at row {i}: {member} vs {paths[i]}")
            rgb = pd_.decode_rgb(data)
            if ref is not None:
                rgb = imageproc.reinhard_normalise(rgb, *ref)
            return enc.preprocess_one(rgb)

        bs = min(batch, enc.spec.max_batch)
        items = ((i, m, rd()) for i, m, rd in pd_.iter_tar_images(tar, skip=done, limit=n))
        buf, rows, part_id = [], [], len(parts)
        t0, seen = time.time(), 0

        def flush_rows():
            nonlocal rows, part_id
            if rows:
                np.save(parts_dir / f"part{part_id:05d}.npy.tmp.npy", np.concatenate(rows).astype(np.float16))
                (parts_dir / f"part{part_id:05d}.npy.tmp.npy").rename(parts_dir / f"part{part_id:05d}.npy")
                part_id += 1
                rows = []

        def run_batch():
            nonlocal buf, seen
            z = enc.encode(np.stack(buf))
            rows.append(z)
            seen += len(buf)
            buf = []
            if sum(len(r) for r in rows) >= chunk_rows:
                flush_rows()
            if (seen // bs) % 20 == 0:
                rate = seen / (time.time() - t0)
                print(f"[{name}] {done + seen}/{n} rows  {rate:.1f} img/s  ETA {(n - done - seen) / rate / 60:.1f} min", flush=True)

        for x in pd_.ordered_prefetch(work, items, workers=workers, window=4 * bs):
            buf.append(x)
            if len(buf) == bs:
                run_batch()
        if buf:
            run_batch()
        flush_rows()
        print(f"[{name}] finished: {seen} images in {time.time() - t0:.0f}s ({seen / max(time.time() - t0, 1e-9):.1f} img/s)")

    arr = np.concatenate([np.load(p) for p in sorted(parts_dir.glob("part*.npy"))])
    if arr.shape[0] != n:
        raise RuntimeError(f"[{name}] got {arr.shape[0]} rows, expected {n} (tar shorter than index?)")
    tmp = out_dir / f"{name}.tmp.npy"
    np.save(tmp, arr.astype(np.float16))
    tmp.rename(final)
    (out_dir / f"{name}.json").write_text(json.dumps(
        {"model": model, "stain_norm": stain_norm, "n": int(n), "dim": int(arr.shape[1]), "spec": {
            "resize": enc.spec.resize, "crop": enc.spec.crop, "mean": list(enc.spec.mean), "std": list(enc.spec.std)}}))
    shutil.rmtree(parts_dir, ignore_errors=True)
    print(f"[{name}] wrote {final} {arr.shape} float16; non-finite values: {int((~np.isfinite(arr)).sum())}")
    return final


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tar", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--model", required=True, help=f"one key or comma list from {list_models()}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="only the first N images in tar order (smoke test; use a separate --out)")
    ap.add_argument("--stain-norm", default="none", choices=["none", "reinhard"])
    ap.add_argument("--stain-ref", default="S360", help="substring of the reference scanner name for Reinhard")
    ap.add_argument("--workers", type=int, default=8, help="decode/preprocess threads")
    ap.add_argument("--colmap", default=None, help='JSON dict (or path to a JSON file) canonical field -> CSV column')
    ap.add_argument("--device-map", default=None, help='JSON dict (or file) device label -> "phone"|"scanner"')
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    def js(v):
        if v is None:
            return None
        return json.loads(Path(v).read_text()) if Path(v).exists() else json.loads(v)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index = pd_.ensure_index(a.tar, a.csv, out, limit=a.limit, colmap=js(a.colmap), device_override=js(a.device_map), workers=a.workers)
    for m in a.model.split(","):
        extract_model(a.tar, index, out, m.strip(), batch=a.batch, stain_norm=a.stain_norm, stain_ref=a.stain_ref,
                      workers=a.workers, limit=a.limit, overwrite=a.overwrite, device=a.device)


if __name__ == "__main__":
    main()
