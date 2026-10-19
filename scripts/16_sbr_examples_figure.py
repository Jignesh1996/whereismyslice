#!/usr/bin/env python3
"""Example SbR BigBrain slices: MRI (cyan) with histology (red) placed by the reference affine (top) and by our
estimate with no initialisation (bottom), cropped to the tissue. Overlap shows as white.
Usage: 16_sbr_examples_figure.py --slices 361 16 22 106 --out figures/sbr_BigBrain_examples.png
"""
import argparse
import sys
from importlib import import_module
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from wms.inplane import warp

sbr = import_module("08_sbr_inplane")
ROOT = Path(__file__).resolve().parents[1]


def placed(img, T, shape):
    return warp(torch.tensor(img, dtype=torch.float32)[None], torch.tensor(T[None], dtype=torch.float32), shape)[0].numpy()


def overlay(m, mm, hw, hmw):
    """red = histology (tissue mask blended with stain, so pale white matter still shows), cyan = MRI; overlap = white."""
    t = hmw > 0.5
    hn = np.clip(hw / max(np.percentile(hw[t], 95), 1e-6), 0, 1) if t.any() else hw
    r = np.clip(0.65 * hmw + 0.35 * hn, 0, 1)
    c = np.clip(m / max(np.percentile(m[mm], 99), 1e-6), 0, 1)
    return np.stack([r, c, c], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", type=int, nargs="+", default=[361, 16, 22, 106])
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default="figures/sbr_BigBrain_examples.png")
    a = ap.parse_args()
    df = pd.read_csv(ROOT / "results/sbr_BigBrain_inplane.csv").set_index("slice")
    n = len(a.slices)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 7.4), facecolor="white")
    for j, sid in enumerate(a.slices):
        _, h, hm, m, mm = sbr.load_pieces("BigBrain", sid)[0]
        T_ref, _ = sbr.reference_affine("BigBrain", sid, "whole", h, hm, "cpu")
        r = df.loc[sid]
        T = np.eye(3)
        T[:2] = np.array([float(v) for v in r["T"].split()]).reshape(2, 3)
        rows, cols = np.nonzero(mm | (placed(hm.astype(np.float32), T_ref, m.shape) > 0.5))
        pad = 8
        sl = (slice(max(rows.min() - pad, 0), rows.max() + pad), slice(max(cols.min() - pad, 0), cols.max() + pad))
        name = a.labels[j] if a.labels else f"slice {sid}"
        for i, (T_, who, dice) in enumerate(((T_ref, "reference", r.dice_reference),
                                             (T, "ours, no starting guess", r.dice_ours))):
            img = overlay(m, mm, placed(h, T_, m.shape), placed(hm.astype(np.float32), T_, m.shape))
            ax = axes[i, j]
            ax.imshow(img[sl])
            ax.set_title(f"{name} · {who}\ntissue overlap {dice:.2f}", fontsize=14, color="#002448")
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / a.out, dpi=100, facecolor=fig.get_facecolor())
    print(ROOT / a.out)


if __name__ == "__main__":
    main()
