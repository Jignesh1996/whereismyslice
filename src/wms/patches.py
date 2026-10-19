"""Small histology patches with exact ground truth, and patch localisation inside a full MRI slice.

Ground truth for SbR BigBrain: a histology pixel q maps to MRI pixel T_ref^-1 q (reference affine), and SbR MRI
slice N is the coronal plane y = y0 + dy N of BigBrain's MRI at 0.4007 mm/px (scripts/09), so every patch pixel
has a known 3D position. Patches are cut with an optional synthetic distortion (wms.distort), whose backward
map takes patch pixels to the original section.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .distort import Distortion, apply, random_distortion
from .metrics import gaussian_blur
from .planes import Pose, as_params, plane_points

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Patch:
    img: np.ndarray            # (h, w) tissue-bright histology patch (distorted)
    mask: np.ndarray           # (h, w) tissue mask
    to_histo: callable         # patch (x, y) -> original histology pixel (x, y)
    px_mm: float               # nominal pixel size given to the method (true size x unknown shrinkage)
    sid: int
    side_px: int = 0           # side of the original square cut (before distortion), in histology px


def sbr_geometry():
    """Fitted SbR BigBrain MRI slice geometry: (x0, z0, y0, dy, px_mm)."""
    d = pd.read_csv(ROOT / "results/sbr_bigbrain_slice_positions.csv")
    dy, y0 = np.polyfit(d.slice, d.y, 1)
    return float(d.x.median()), float(d.z.median()), float(y0), float(dy), float(d.px_mm.median())


def sbr_mri_world(sid, cols, rows, shape=(303, 348), geom=None):
    """World mm of SbR MRI slice pixels (col, row) (float arrays)."""
    x0, z0, y0, dy, px = geom or sbr_geometry()
    H, W = shape
    x = x0 + (np.asarray(cols) - (W - 1) / 2) * px
    z = z0 + ((H - 1) / 2 - np.asarray(rows)) * px
    return np.stack([x, np.full_like(x, y0 + dy * sid), z], -1)


def cut_patch(h, hm, T_ref, rng, size_mm=(20, 40), mri_px_mm=0.4007, distortion=None, min_tissue=0.6, tries=50):
    """Random square patch from histology section h (tissue >= min_tissue), optionally distorted.

    distortion: kwargs for wms.distort.random_distortion (rotation, affine, elastic, cuts) or None.
    Returns Patch with a nominal pixel size that includes a random 0.9-1.1 shrinkage the method does not know.
    """
    histo_px = mri_px_mm * np.sqrt(abs(np.linalg.det(np.linalg.inv(T_ref[:2, :2]))))      # true mm per histology px
    side = int(round(rng.uniform(*size_mm) / histo_px))
    H, W = h.shape
    for _ in range(tries):
        cy, cx = rng.integers(side // 2, H - side // 2), rng.integers(side // 2, W - side // 2)
        y0, x0 = cy - side // 2, cx - side // 2
        if hm[y0:y0 + side, x0:x0 + side].mean() >= min_tissue:
            break
    else:
        return None
    img, msk = h[y0:y0 + side, x0:x0 + side], hm[y0:y0 + side, x0:x0 + side]
    if distortion:
        dist = random_distortion(img.shape, np.ones_like(msk), rng, **distortion)
        img, msk = apply(dist, img, msk)
        to_h = lambda x, y, d=dist: tuple(v + o for v, o in zip(d.G(x, y), (x0, y0)))
    else:
        to_h = lambda x, y: (np.asarray(x) + x0, np.asarray(y) + y0)
    return Patch(img.astype(np.float32), msk, to_h, histo_px * rng.uniform(0.9, 1.1), -1, side)


def truth_mri_px(patch, T_ref):
    """MRI pixel (x, y) of the patch centre."""
    h, w = patch.img.shape
    hx, hy = patch.to_histo(np.array([(w - 1) / 2]), np.array([(h - 1) / 2]))
    q = np.linalg.inv(T_ref) @ np.array([hx[0], hy[0], 1.0])
    return q[:2]


def _rot_scale(img, angle_deg, zoom, out):
    """Rotate (deg) and zoom an (n, h, w) batch about its centre into an out x out canvas (zeros outside).
    zoom = output pixels per input pixel; output o = zoom * R(angle) * input, about the centres."""
    th = np.deg2rad(angle_deg)
    n, h, w = img.shape
    rinv = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    theta = torch.zeros(n, 2, 3, device=img.device)
    theta[:, 0, :2] = torch.tensor(rinv[0] * out / (w * zoom), dtype=torch.float32)
    theta[:, 1, :2] = torch.tensor(rinv[1] * out / (h * zoom), dtype=torch.float32)
    grid = F.affine_grid(theta, (n, 1, out, out), align_corners=False)
    return F.grid_sample(img[:, None], grid, align_corners=False)[:, 0]


def ncc_map(image, templates, tmask):
    """Masked NCC of each template (T, h, w) with weights tmask over every position of image (H, W) -> (T, H-h+1, W-w+1)."""
    Tn, h, w = templates.shape
    H, W = image.shape
    m = tmask
    n = m.sum((1, 2)).clamp_min(1)[:, None, None]
    t = (templates - (templates * m).sum((1, 2))[:, None, None] / n) * m
    tn = t.square().sum((1, 2)).sqrt()[:, None, None]
    I = image[None, None]
    s1 = F.conv2d(I, m[:, None])[0]
    s2 = F.conv2d(I.square(), m[:, None])[0]
    num = F.conv2d(I, t[:, None])[0]
    var = (s2 - s1.square() / n).clamp_min(0)
    return num / (tn * var.sqrt() + 1e-6), var


def classical_patch_search(patch, mri, mri_px_mm=0.4007, angles=np.arange(0, 360, 10), scales=(0.9, 1.0, 1.1),
                           device="cuda", top_k=5):
    """Template matching of a histology patch over an MRI slice: rotation x scale grid, masked NCC of blurred images,
    both contrast polarities (histology and MR contrast are roughly inverted). Returns [(score, x, y, angle, scale)]."""
    img = torch.tensor(mri, dtype=torch.float32, device=device)
    img = gaussian_blur(img[None], 1.0)[0]
    p = torch.tensor(patch.img, dtype=torch.float32, device=device)[None]
    pm = torch.tensor(patch.mask, dtype=torch.float32, device=device)[None]
    r = patch.px_mm / mri_px_mm                                            # MRI px per patch px (nominal)
    out = []
    for s in scales:
        zoom = r * s
        side = int(np.ceil(max(p.shape[1:]) * zoom * 1.45)) | 1
        for a in angles:
            t = gaussian_blur(_rot_scale(p, a, zoom, side), 1.0)
            m = (_rot_scale(pm, a, zoom, side) > 0.5).float()
            score, var = ncc_map(img, t, m)
            score = score.abs().masked_fill(var < 1e-4, 0)[0]
            v, i = score.flatten().max(0)
            yy, xx = divmod(int(i), score.shape[1])
            out.append((float(v), xx + (side - 1) / 2, yy + (side - 1) / 2, float(a), float(s)))
    out.sort(key=lambda o: -o[0])
    return out[:top_k]
