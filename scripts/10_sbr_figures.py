#!/usr/bin/env python3
"""Figures for the SbR in-plane test: distribution of distance-to-reference and Dice, and example slices
(MRI with the histology outline placed by the reference affine and by our estimate).
Output: figures/sbr_<dataset>_inplane.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

from wms.inplane import warp

sbr = import_module("08_sbr_inplane")
ROOT = Path(__file__).resolve().parents[1]


def placed_mask(hm, T, shape):
    return warp(torch.tensor(hm, dtype=torch.float32)[None], torch.tensor(T[None], dtype=torch.float32), shape)[0].numpy() > 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BigBrain")
    ap.add_argument("--unit", default="mm")
    a = ap.parse_args()
    df = pd.read_csv(ROOT / f"results/sbr_{a.dataset}_inplane.csv").sort_values("err_mean_mm").reset_index(drop=True)
    picks = [0, len(df) // 4, len(df) // 2, 3 * len(df) // 4, len(df) - 1]
    fig = plt.figure(figsize=(18, 8.5), facecolor="white")
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.25])
    ax = fig.add_subplot(gs[0, :2])
    ax.hist(df.err_mean_mm.clip(upper=10), bins=40, color="#3b6ea5")
    ax.set(xlabel=f"mean distance between reference and our placement ({a.unit}, clipped at 10)", ylabel="slices",
           title=f"{a.dataset}: n={len(df)}, median {df.err_mean_mm.median():.2f} {a.unit}")
    ax = fig.add_subplot(gs[0, 2:4])
    ax.scatter(df.dice_reference, df.dice_ours, s=8, c=np.log10(df.err_mean_mm + 1e-3), cmap="viridis")
    lo = min(df.dice_reference.min(), df.dice_ours.min())
    ax.plot([lo, 1], [lo, 1], "k--", lw=0.8)
    ax.set(xlabel="tissue Dice, reference affine", ylabel="tissue Dice, ours",
           title=f"ours higher on {np.mean(df.dice_ours > df.dice_reference):.0%} of slices")
    ax = fig.add_subplot(gs[0, 4])
    ax.axis("off")
    ax.text(0, 0.9, f"distance < 1 {a.unit}: {np.mean(df.err_mean_mm < 1):.0%}\n"
                    f"distance < 2 {a.unit}: {np.mean(df.err_mean_mm < 2):.0%}\n"
                    f"Dice ours {df.dice_ours.median():.3f}\n"
                    f"Dice reference {df.dice_reference.median():.3f}\n"
                    f"{df.seconds.median():.1f} s / slice (GPU)", fontsize=13, va="top")
    for j, k in enumerate(picks):
        r = df.iloc[k]
        piece = r.get("piece", "whole")
        items = {p[0]: p for p in sbr.load_pieces(a.dataset, int(r.slice))}
        _, h, hm, m, mm = items[piece if piece in items else next(iter(items))]
        T_gt, _ = sbr.reference_affine(a.dataset, int(r.slice), piece, h, hm, "cpu")
        T = np.eye(3)
        T[:2] = np.array([float(v) for v in r["T"].split()]).reshape(2, 3)
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(m, cmap="gray")
        ax.contour(placed_mask(hm, T_gt, m.shape), levels=[0.5], colors="#ff9f1c", linewidths=1.2)
        ax.contour(placed_mask(hm, T, m.shape), levels=[0.5], colors="#2ec4b6", linewidths=1.0, linestyles="--")
        label = ["best", "25th pct", "median", "75th pct", "worst"][j]
        ax.set_title(f"{label}: slice {int(r.slice)} {'' if piece == 'whole' else piece}\n"
                     f"distance {r.err_mean_mm:.2f} {a.unit}; Dice ref {r.dice_reference:.3f} / ours {r.dice_ours:.3f}", fontsize=9)
        ax.axis("off")
    fig.text(0.5, 0.01, "orange = histology outline placed by the reference affine, teal dashed = placed by our estimate (no initialisation)",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(ROOT / f"figures/sbr_{a.dataset}_inplane.png", dpi=80)


if __name__ == "__main__":
    main()
