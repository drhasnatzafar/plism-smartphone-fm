"""PLISM-sm data access: metadata CSV loading, tar streaming, and the master ``index.csv``.

Design notes
------------
* The 19 GB tar.gz is never extracted.  It is streamed with ``tarfile`` mode ``'r|gz'``; PNG bytes are decoded
  on the fly by a small thread pool (PIL/zlib release the GIL).
* The metadata CSV column names are unknown at the time of writing.  ``load_metadata`` prints the columns it
  finds and maps them to canonical names through ``DEFAULT_COLMAP`` (candidate lists, matched after
  lower-casing and removing non-alphanumerics).  Override with ``--colmap '{"tissue": "Tissue type"}'``.
  Stain / device / coordinates can always be derived from the tar path ``<stain>/<device>/<x0>_<y0>_<x1>_<y1>.png``.
* ``index.csv`` row order == tar stream order.  Every embedding file ``<model>.npy`` has the same row order.
"""
from __future__ import annotations

import io
import json
import re
import tarfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import pandas as pd
from PIL import Image

import imageproc

IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# canonical field -> candidate column names (normalised: lower-case, alphanumerics only)
DEFAULT_COLMAP: dict[str, list[str]] = {
    "path": ["imagepath", "path", "file", "filename", "image", "imgpath", "filepath", "relativepath", "name"],
    "stain": ["stain", "staining", "stainingcondition", "staincondition", "stainid"],
    "device": ["device", "devicename", "scanner", "scannername", "camera", "model", "imagingdevice"],
    "tissue": ["tissue", "tissuetype", "tissueclass", "organ", "tissuename", "class", "label"],
    "coords": ["coordinates", "coordinate", "coords", "coord", "location", "position"],
    "x0": ["x0", "xmin", "left", "xstart", "x1coord"],
    "y0": ["y0", "ymin", "top", "ystart", "y1coord"],
    "x1": ["x1", "xmax", "right", "xend"],
    "y1": ["y1", "ymax", "bottom", "yend"],
    "group": ["group", "groupid", "locationid", "tileid", "patchid"],
}

# Substring keywords on the normalised device string.  Phones are tested first.
PHONE_KEYWORDS = ["samsung", "galaxy", "moto", "xiaomi", "redmi", "iphone", "itel", "transsion",
                  "phone", "mobile", "smartphone", "p33", "note9", "g8"]
SCANNER_KEYWORDS = ["hamamatsu", "aperio", "philips", "ultrafast", "nanozoomer", "scanner", "leica"]
SCANNER_EXACT = {"p",  # PLISM code for the Philips Ultrafast scanner
                 "s360", "s210", "sq", "s60", "at2", "gt450", "ultrafast", "hs360", "hs210", "hsq", "hs60"}


def norm(s: object) -> str:
    """Lower-case and strip everything but letters/digits."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def device_type(name: str, override: dict[str, str] | None = None) -> str:
    """Classify a device label as ``'phone'`` / ``'scanner'`` / ``'unknown'``."""
    if override:
        for k, v in override.items():
            if norm(k) == norm(name):
                return v
    n = norm(name)
    if n in SCANNER_EXACT:
        return "scanner"
    if any(k in n for k in SCANNER_KEYWORDS):
        return "scanner"
    if any(k in n for k in PHONE_KEYWORDS):
        return "phone"
    return "unknown"


# --------------------------------------------------------------------------------------
# tar paths
# --------------------------------------------------------------------------------------
_COORD_RE = re.compile(r"(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)")


def parse_tar_path(name: str) -> dict:
    """Derive stain / device / coordinates / key from ``[top/]<stain>/<device>/<x0>_<y0>_<x1>_<y1>.png``."""
    parts = [p for p in name.replace("\\", "/").split("/") if p]
    stem = parts[-1].rsplit(".", 1)[0]
    m = _COORD_RE.search(stem)
    coords = [int(v) for v in m.groups()] if m else [None] * 4
    return {
        "path": name,
        "stain": parts[-3] if len(parts) >= 3 else None,
        "device": parts[-2] if len(parts) >= 2 else None,
        "x0": coords[0], "y0": coords[1], "x1": coords[2], "y1": coords[3],
        "key": "/".join(parts[-3:]).rsplit(".", 1)[0],
    }


def iter_tar_images(tar_path: str | Path, skip: int = 0, limit: int | None = None) -> Iterator[tuple[int, str, Callable[[], bytes]]]:
    """Stream image members of a ``.tar.gz`` once, in archive order.

    Yields ``(i, member_name, read_fn)`` where ``i`` is the image counter.  ``read_fn()`` returns the raw
    bytes and MUST be called before the next iteration (stream mode).  The first ``skip`` images are
    passed over without reading their payload (they still have to be decompressed by gzip).
    ``limit`` caps the total image counter (exclusive).
    """
    if str(tar_path).startswith(("http://", "https://")):
        # stream straight from the web (e.g. Figshare); urllib follows the redirect to the storage URL
        import urllib.request
        req = urllib.request.Request(str(tar_path), headers={"User-Agent": "plism-study/1.0"})
        opener = lambda: tarfile.open(fileobj=urllib.request.urlopen(req, timeout=120), mode="r|gz")
    else:
        opener = lambda: tarfile.open(str(tar_path), mode="r|gz")
    with opener() as tar:
        i = 0
        for m in tar:
            base = m.name.rsplit("/", 1)[-1]
            if not m.isfile() or base.startswith(".") or not base.lower().endswith(IMG_EXT):
                continue
            if limit is not None and i >= limit:
                break
            if i < skip:
                i += 1
                continue
            f = tar.extractfile(m)
            yield i, m.name, (lambda f=f: f.read())
            i += 1


def ordered_prefetch(fn: Callable, items: Iterable, workers: int = 8, window: int = 64) -> Iterator:
    """``map(fn, items)`` on a thread pool, order preserved, at most ``window`` items in flight.

    ``items`` is consumed lazily from the calling thread, which matters because tar members must be read
    sequentially in the producer thread.
    """
    with ThreadPoolExecutor(max_workers=workers) as ex:
        q: deque = deque()
        for it in items:
            q.append(ex.submit(fn, it))
            if len(q) >= window:
                yield q.popleft().result()
        while q:
            yield q.popleft().result()


def decode_rgb(data: bytes) -> np.ndarray:
    """Decode image bytes to a uint8 RGB array."""
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"))


# --------------------------------------------------------------------------------------
# metadata CSV
# --------------------------------------------------------------------------------------


def resolve_columns(columns: list[str], colmap: dict | None = None, verbose: bool = True) -> dict[str, str]:
    """Map canonical field -> actual CSV column.  ``colmap`` entries (field -> exact column) win."""
    colmap = dict(colmap or {})
    byn = {norm(c): c for c in columns}
    out: dict[str, str] = {}
    for field, cands in DEFAULT_COLMAP.items():
        if field in colmap:
            if colmap[field] not in columns:
                raise KeyError(f"colmap[{field!r}]={colmap[field]!r} not in CSV columns {columns}")
            out[field] = colmap[field]
            continue
        for c in cands:
            if c in byn:
                out[field] = byn[c]
                break
    if verbose:
        print(f"[metadata] CSV columns: {columns}")
        print(f"[metadata] mapping used: {out}")
        missing = [k for k in ("path", "stain", "device", "tissue", "coords", "x0") if k not in out]
        if missing:
            print(f"[metadata] not found in CSV (will derive from tar path where possible): {missing}")
    return out


def load_metadata(csv_path: str | Path | None, colmap: dict | None = None, verbose: bool = True) -> pd.DataFrame | None:
    """Read the metadata CSV and return a frame with canonical columns (subset of path/stain/device/tissue/
    x0..y1/group) plus ``key`` (last three path components without extension) when a path column exists."""
    if csv_path is None or not Path(csv_path).exists():
        if verbose:
            print("[metadata] no CSV given/found; deriving everything from tar paths (tissue will be missing)")
        return None
    raw = pd.read_csv(csv_path)
    cm = resolve_columns(list(raw.columns), colmap, verbose)
    df = pd.DataFrame(index=raw.index)
    for f in ("path", "stain", "device", "tissue", "group", "x0", "y0", "x1", "y1"):
        if f in cm:
            df[f] = raw[cm[f]]
    if "x0" not in df and "coords" in cm:
        parsed = raw[cm["coords"]].astype(str).str.extractall(r"(-?\d+)").unstack()
        if parsed.shape[1] >= 4:
            for j, f in enumerate(("x0", "y0", "x1", "y1")):
                df[f] = pd.to_numeric(parsed.iloc[:, j], errors="coerce")
    if "path" in df:
        df["key"] = df["path"].astype(str).map(lambda p: parse_tar_path(p)["key"])
    for f in ("stain", "device", "tissue"):
        if f in df:
            df[f] = df[f].astype(str)
    if verbose:
        print(f"[metadata] {len(df)} rows; "
              + "; ".join(f"{f}: {df[f].nunique()} unique" for f in ("stain", "device", "tissue") if f in df))
    return df


def merge_metadata(tar_df: pd.DataFrame, meta: pd.DataFrame | None, verbose: bool = True) -> pd.DataFrame:
    """Left-join CSV metadata onto the tar listing.  CSV wins for stain/device/tissue when present."""
    out = tar_df.copy()
    out["tissue"] = np.nan
    if meta is None:
        return out
    m = None
    if "key" in meta:
        m = out[["key"]].merge(meta.drop_duplicates("key"), on="key", how="left")
        rate = m["tissue"].notna().mean() if "tissue" in m else m["stain"].notna().mean()
        if verbose:
            print(f"[metadata] joined on path key: match rate {rate:.1%}")
    if m is None or (("tissue" in m and m["tissue"].notna().mean() < 0.9)):
        if {"stain", "device", "x0"}.issubset(meta.columns):  # fall back on (stain, device, coords)
            for c in ("x0", "y0", "x1", "y1"):
                meta[c] = pd.to_numeric(meta[c], errors="coerce")
            meta = meta.assign(_k=meta["stain"].map(norm) + "|" + meta["device"].map(norm) + "|" + meta[["x0", "y0", "x1", "y1"]].astype("Int64").astype(str).agg("_".join, axis=1))
            o = out.assign(_k=out["stain"].map(norm) + "|" + out["device"].map(norm) + "|" + out[["x0", "y0", "x1", "y1"]].astype("Int64").astype(str).agg("_".join, axis=1))
            m2 = o[["_k"]].merge(meta.drop_duplicates("_k"), on="_k", how="left")
            if verbose:
                print(f"[metadata] fallback join on (stain, device, coords): match rate {m2['tissue'].notna().mean() if 'tissue' in m2 else 0:.1%}")
            m = m2
    if m is not None:
        for f in ("tissue", "group"):
            if f in m:
                out[f] = m[f].values
        for f in ("stain", "device"):  # prefer CSV labels (usually cleaner) when available
            if f in m:
                out[f] = np.where(pd.notna(m[f].values), m[f].values, out[f].values)
    return out


# --------------------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------------------


def add_groups(df: pd.DataFrame, device_override: dict[str, str] | None = None) -> pd.DataFrame:
    """Add ``device_type``, ``group_id`` (stain + location, shared by all devices) and ``loc_id`` (tissue +
    location, used for leakage-free splitting; merges the same location across stains)."""
    df = df.copy()
    df["device_type"] = df["device"].map(lambda d: device_type(d, device_override))
    coord = df[["x0", "y0", "x1", "y1"]].astype("Int64").astype(str).agg("_".join, axis=1)
    if "group" in df and df["group"].notna().all():
        df["group_id"] = df["group"].astype(str)
    else:
        df["group_id"] = df["stain"].astype(str) + "|" + coord
    tissue = df["tissue"].fillna("NA").astype(str)
    df["loc_id"] = tissue + "|" + coord
    return df.drop(columns=[c for c in ("group",) if c in df])


INDEX_COLS = ["row", "path", "stain", "device", "device_type", "group_id", "loc_id", "x0", "y0", "x1", "y1",
              "tissue", "sharpness", "saturation"]


def _stats_worker(item):
    i, name, data = item
    rgb = decode_rgb(data)
    mu, sd = imageproc.lab_stats(rgb)
    return i, name, imageproc.laplacian_variance(rgb), imageproc.mean_saturation(rgb), mu, sd


def build_index(tar_path, csv_path, out_dir, limit: int | None = None, colmap: dict | None = None,
                device_override: dict[str, str] | None = None, workers: int = 8) -> pd.DataFrame:
    """One streaming pass over the tar: member names, image-quality proxies, per-device Lab statistics.

    Writes ``index.csv``, ``index.meta.json`` and ``lab_stats.json`` (per-device mean of per-image Lab
    mean/std, used as Reinhard reference) into ``out_dir``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[index] streaming {tar_path} (limit={limit}) ...")
    t0 = time.time()
    items = ((i, n, rd()) for i, n, rd in iter_tar_images(tar_path, limit=limit))
    rows, mus, sds = [], [], []
    for k, (i, name, sharp, sat, mu, sd) in enumerate(ordered_prefetch(_stats_worker, items, workers=workers)):
        rows.append({**parse_tar_path(name), "row": i, "sharpness": sharp, "saturation": sat})
        mus.append(mu)
        sds.append(sd)
        if (k + 1) % 2000 == 0:
            print(f"[index] {k + 1} images, {(k + 1) / (time.time() - t0):.0f} img/s")
    if not rows:
        raise RuntimeError(f"no images found in {tar_path}")
    tar_df = pd.DataFrame(rows)
    df = merge_metadata(tar_df, load_metadata(csv_path, colmap), verbose=True)
    df = add_groups(df, device_override)
    df = df[INDEX_COLS]
    if (df.device_type == "unknown").any():
        print("[index] WARNING unknown device types (pass --device-map to fix): "
              f"{sorted(df.loc[df.device_type == 'unknown', 'device'].unique())}")
    mus, sds = np.array(mus), np.array(sds)
    stats = {d: {"mean": mus[(df.device == d).values].mean(0).tolist(), "std": sds[(df.device == d).values].mean(0).tolist()}
             for d in df.device.unique()}
    df.to_csv(out_dir / "index.csv", index=False)
    (out_dir / "lab_stats.json").write_text(json.dumps(stats, indent=1))
    (out_dir / "index.meta.json").write_text(json.dumps({"n": len(df), "limit": limit, "tar": str(tar_path)}))
    print(f"[index] wrote {len(df)} rows in {time.time() - t0:.0f}s; devices={df.device.nunique()} stains={df.stain.nunique()} "
          f"groups={df.group_id.nunique()} phones={df.loc[df.device_type=='phone','device'].nunique()} "
          f"scanners={df.loc[df.device_type=='scanner','device'].nunique()}")
    return df


def ensure_index(tar_path, csv_path, out_dir, limit: int | None = None, **kw) -> pd.DataFrame:
    """Reuse ``<out_dir>/index.csv`` if present (refuse if it was built with a different ``limit``)."""
    out_dir = Path(out_dir)
    ip, mp = out_dir / "index.csv", out_dir / "index.meta.json"
    if ip.exists() and mp.exists():
        meta = json.loads(mp.read_text())
        if meta.get("limit") != limit:
            raise RuntimeError(f"{ip} was built with limit={meta.get('limit')} but this run uses limit={limit}. "
                               "Use a separate --out directory for smoke tests.")
        print(f"[index] reusing {ip} ({meta['n']} rows)")
        return pd.read_csv(ip)
    return build_index(tar_path, csv_path, out_dir, limit=limit, **kw)
