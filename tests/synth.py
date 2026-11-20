"""Synthetic PLISM-like data generators for tests (tiny PNGs in a tar.gz + metadata CSV; synthetic embeddings)."""
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PHONES = ["Samsung Galaxy S20 5G", "Motorola moto g8", "Xiaomi Redmi Note9 Pro", "iPhone 6", "iPhone 13 mini", "Transsion iTel P33"]
SCANNERS = ["Hamamatsu S360", "Hamamatsu S210", "Hamamatsu SQ", "Hamamatsu S60", "Aperio AT2", "Aperio GT450", "Philips Ultrafast"]
DEVICES = PHONES + SCANNERS  # 13
TISSUES = ["liver", "lung", "colon", "kidney"]


def folder(name: str) -> str:
    return name.replace(" ", "")


def make_tar(root: Path, n_stains=3, n_locs=6, size=48, seed=0, top="PLISM_sm") -> tuple[Path, Path]:
    """3 stains x 13 devices x 6 locations small PNGs; CSV with deliberately non-default column names."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    tar_path, csv_path = root / "41397708", root / "41397675"  # no extensions, like Kaggle
    locs = [(i * 512, (i % 3) * 512, i * 512 + 512, (i % 3) * 512 + 512) for i in range(n_locs)]
    loc_tissue = [TISSUES[i % len(TISSUES)] for i in range(n_locs)]
    base = {t: rng.uniform(0.3, 0.9, (size, size, 3)) for t in TISSUES}
    rows = []
    with tarfile.open(tar_path, "w:gz") as tar:
        for s in range(n_stains):
            for d, dev in enumerate(DEVICES):
                for li, c in enumerate(locs):
                    tex = base[loc_tissue[li]] * (1 + 0.1 * s) + rng.normal(0, 0.05, (size, size, 3))
                    tex = tex * (0.8 + 0.4 * (d / 13)) + (0.05 if dev in PHONES else 0.0)
                    arr = np.clip(tex * 255, 0, 255).astype(np.uint8)
                    buf = io.BytesIO()
                    Image.fromarray(arr).save(buf, format="PNG")
                    name = f"{top}/stain{s:02d}/{folder(dev)}/{c[0]}_{c[1]}_{c[2]}_{c[3]}.png"
                    ti = tarfile.TarInfo(name)
                    ti.size = len(buf.getvalue())
                    tar.addfile(ti, io.BytesIO(buf.getvalue()))
                    rows.append({"File": name.split("/", 1)[1], "Staining": f"stain{s:02d}", "Device": dev,
                                 "Tissue type": loc_tissue[li], "x0": c[0], "y0": c[1], "x1": c[2], "y1": c[3]})
    pd.DataFrame(rows).sample(frac=1.0, random_state=1).to_csv(csv_path, index=False)  # shuffled, like a real CSV
    return tar_path, csv_path


def make_embeddings(n_stains=3, n_locs=40, dim=32, n_tissue=6, seed=0, phone_shift=1.5, noise=1.0, proto_scale=1.0):
    """Synthetic (index_df, E): tissue prototype + per-location offset + per-device shift (phones larger) + noise."""
    rng = np.random.default_rng(seed)
    proto = rng.standard_normal((n_tissue, dim)) * proto_scale
    dev_shift = {d: rng.standard_normal(dim) * (phone_shift if d in PHONES else 0.15) for d in DEVICES}
    rows, embs = [], []
    loc_tissue = rng.integers(0, n_tissue, n_locs)
    loc_off = rng.standard_normal((n_locs, dim)) * 0.5
    for s in range(n_stains):
        for dev in DEVICES:
            for li in range(n_locs):
                c = (li * 512, 0, li * 512 + 512, 512)
                e = proto[loc_tissue[li]] + loc_off[li] + dev_shift[dev] + rng.standard_normal(dim) * noise
                embs.append(e)
                rows.append({"path": f"stain{s}/{folder(dev)}/{c[0]}_{c[1]}_{c[2]}_{c[3]}.png", "stain": f"stain{s}", "device": dev,
                             "device_type": "phone" if dev in PHONES else "scanner", "group_id": f"stain{s}|{c[0]}_0_{c[2]}_512",
                             "loc_id": f"t{loc_tissue[li]}|{c[0]}_0_{c[2]}_512", "x0": c[0], "y0": 0, "x1": c[2], "y1": 512,
                             "tissue": f"t{loc_tissue[li]}",
                             "sharpness": float(rng.uniform(50, 200) * (0.5 if dev in PHONES else 1)),
                             "saturation": float(rng.uniform(0.2, 0.5))})
    df = pd.DataFrame(rows)
    df.insert(0, "row", np.arange(len(df)))
    return df, np.asarray(embs, dtype=np.float32)
