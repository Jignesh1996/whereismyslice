#!/usr/bin/env python3
"""Dual encoder training figures: example training pairs exactly as the trainer renders them (scripts/06: random
oblique pose, histology plane with artefacts, matching MRI plane, canonical view) and the training curve.
Output: figures/dual_encoder_pairs.png, figures/dual_encoder_curve.png
"""
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
from wms import volumes
from wms.augment import augment_torch
from wms.planes import Pose

t06 = import_module("06_train_dual_encoder")
ROOT = Path(__file__).resolve().parents[1]
NAVY, TEAL, ORANGE = "#002448", "#1F6E7A", "#D4A54D"


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hist, mri = volumes.load("bigbrain_400", dev), volumes.load("bigbrain_mri", dev)
    bounds = hist.bounds_mm
    n_px = int(np.max(bounds[1] - bounds[0]) * 1.15 / t06.RENDER_PX)
    rng = np.random.default_rng(7)
    params = t06.draw_poses(24, bounds, rng, (0.0, 20.0)).to(dev)
    vh, vm, keep = t06.render_pair(hist, mri, params, n_px)
    torch.manual_seed(3)
    vh, vm, params = augment_torch(vh[keep]), vm[keep], params[keep].cpu().numpy()
    # pick 5 spread along the brain
    order = np.argsort(params[:, 1])
    pick = order[np.linspace(0, len(order) - 1, 5).round().astype(int)]
    fig, axes = plt.subplots(2, 5, figsize=(19, 8.2), facecolor="white", gridspec_kw=dict(hspace=0.12, wspace=0.05))
    for j, i in enumerate(pick):
        p = Pose.from_array(params[i])
        for r, (img, name) in enumerate(((vh[i], "histology + artefacts"), (vm[i], "MRI, same plane"))):
            ax = axes[r, j]
            x = img.float().cpu().numpy()
            ax.imshow(1 - x if r == 0 else x, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                tilt = np.degrees(np.arccos(min(1.0, abs(p.normal[1]))))
                ax.set_title(f"y {p.y:+.0f} mm, tilt {tilt:.0f}°", fontsize=18, color=NAVY)
            if j == 0:
                ax.set_ylabel(name, fontsize=17, color=NAVY)
    fig.savefig(ROOT / "figures/dual_encoder_pairs.png", dpi=90, bbox_inches="tight", facecolor="white")

    log = pd.read_csv(ROOT / "results/train_log.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor="white")
    ax.plot(log.minutes, log.nce, color=TEAL, lw=3)
    ax.set_xlabel("training time (minutes, laptop GPU)", fontsize=15, color=NAVY)
    ax.set_ylabel("contrastive loss", fontsize=15, color=NAVY)
    ax.tick_params(labelsize=13)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylim(0, None)
    fig.savefig(ROOT / "figures/dual_encoder_curve.png", dpi=100, bbox_inches="tight", facecolor="white")
    print("figures/dual_encoder_pairs.png, figures/dual_encoder_curve.png")


if __name__ == "__main__":
    main()
