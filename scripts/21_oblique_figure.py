#!/usr/bin/env python3
"""Oblique test examples (scripts/19): for each tilt, the resliced histology section, the MRI plane at the pose we
found, and the MRI plane at the true pose. Output: figures/oblique_examples.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAVY, TEAL = "#002448", "#1F6E7A"


def main():
    ex = np.load(ROOT / "results/oblique_examples.npz")
    tilts = sorted(int(k.split("_")[1]) for k in ex.files if k.startswith("query_"))
    fig, axes = plt.subplots(3, len(tilts), figsize=(4.6 * len(tilts), 11), facecolor="white",
                             gridspec_kw=dict(hspace=0.32, wspace=0.08))
    rows = ("histology,\ncut obliquely", "MRI where\nwe found it", "MRI at the\ntrue pose")
    for j, t in enumerate(tilts):
        tt, tf, ang, off = ex[f"info_{t}"]
        for i, key in enumerate(("query", "found", "true")):
            ax = axes[i, j]
            img = ex[f"{key}_{t}"]
            ax.imshow(1 - img if key == "query" else img, cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(TEAL if key == "found" else "white")
                sp.set_linewidth(5 if key == "found" else 0)
            if i == 0:
                ax.set_title(f"tilted {tt:.0f}°", fontsize=22, color=NAVY, fontweight="bold")
            if key == "found":
                ax.set_xlabel(f"found {tf:.1f}°\nerror {ang:.1f}° · {off:.1f} mm", fontsize=17, color=TEAL)
            if j == 0:
                ax.set_ylabel(rows[i], fontsize=18, color=NAVY)
    fig.savefig(ROOT / "figures/oblique_examples.png", dpi=90, bbox_inches="tight", facecolor="white")
    print(ROOT / "figures/oblique_examples.png")


if __name__ == "__main__":
    main()
