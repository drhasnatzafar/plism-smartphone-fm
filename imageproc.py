"""Light-weight, numpy-only image utilities: quality proxies and Reinhard stain normalisation.

No scikit-image / OpenCV dependency.  Everything operates on ``uint8`` RGB arrays of shape (H, W, 3).
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------------------
# Quality proxies
# --------------------------------------------------------------------------------------


def laplacian_variance(rgb: np.ndarray) -> float:
    """Sharpness proxy: variance of the 4-neighbour Laplacian of the grayscale image (0-255 scale)."""
    g = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    return float(lap.var())


def mean_saturation(rgb: np.ndarray) -> float:
    """Mean HSV saturation, S = (max - min) / max, in [0, 1] (pixels with max == 0 count as 0)."""
    x = rgb.astype(np.float32)
    mx, mn = x.max(-1), x.min(-1)
    s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return float(s.mean())


def tissue_mask(rgb: np.ndarray, white_thr: int = 220) -> np.ndarray:
    """Crude foreground mask: a pixel is background if all channels exceed ``white_thr``."""
    return rgb.min(-1) < white_thr


# --------------------------------------------------------------------------------------
# sRGB <-> CIE Lab (D65) in numpy
# --------------------------------------------------------------------------------------
_M_RGB2XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375],
     [0.2126729, 0.7151522, 0.0721750],
     [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ).astype(np.float32)
_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
_EPS, _KAPPA = 216.0 / 24389.0, 24389.0 / 27.0


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB (..., 3) -> float32 Lab (..., 3)."""
    c = rgb.astype(np.float32) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = (lin @ _M_RGB2XYZ.T) / _WHITE
    f = np.where(xyz > _EPS, np.cbrt(xyz), (_KAPPA * xyz + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], -1).astype(np.float32)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """float Lab (..., 3) -> uint8 RGB (..., 3), clipped to gamut."""
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f ** 3 > _EPS, f ** 3, (116.0 * f - 16.0) / _KAPPA) * _WHITE
    lin = np.clip(xyz @ _M_XYZ2RGB.T, 0.0, 1.0)
    c = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return np.clip(np.rint(c * 255.0), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------------------
# Reinhard colour transfer (Reinhard et al., 2001) in Lab space
# --------------------------------------------------------------------------------------


def lab_stats(rgb: np.ndarray, stride: int = 4, mask: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel Lab mean/std of an image over (strided) foreground pixels. Returns (mean[3], std[3])."""
    sub = rgb[::stride, ::stride]
    lab = rgb_to_lab(sub).reshape(-1, 3)
    if mask:
        m = tissue_mask(sub).reshape(-1)
        if m.mean() > 0.1:  # keep fall-back to all pixels for (almost) blank tiles
            lab = lab[m]
    return lab.mean(0), lab.std(0)


def reinhard_normalise(rgb: np.ndarray, ref_mean: np.ndarray, ref_std: np.ndarray) -> np.ndarray:
    """Match the per-channel Lab mean/std of ``rgb`` to the reference statistics.

    Statistics of the source image are computed on foreground pixels only; the affine map is applied to the
    whole image.  Returns uint8 RGB.
    """
    lab = rgb_to_lab(rgb)
    mu, sd = lab_stats(rgb, stride=2)
    out = (lab - mu) / np.maximum(sd, 1e-3) * np.asarray(ref_std, np.float32) + np.asarray(ref_mean, np.float32)
    return lab_to_rgb(out)
