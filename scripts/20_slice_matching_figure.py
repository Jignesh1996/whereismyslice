#!/usr/bin/env python3
"""Figure for SbR slice matching (scripts/18): does the section land in the right neighbourhood of MRI slices?

Left: one raw section and the MRI slices around where it was placed (-4, -2, found, +2, +4 mm), true slice marked.
Right: similarity of the section to MRI planes against distance from the TRUE slice, all sections (median and
25-75 % band), with the example's curve; a peak at 0 means the true slice is the most similar one.
Output: figures/sbr_slice_neighbours.png, figures/sbr_slice_profile.png
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
from wms import volumes
from wms.classical import localise, refine, to_isotropic, _volume_level
from wms.patches import sbr_geometry
from wms.planes import rotation, sample

sbr = import_module("08_sbr_inplane")
s18 = import_module("18_sbr_slice_matching")
ROOT = Path(__file__).resolve().parents[1]
NAVY, TEAL, ORANGE, GREY = "#002448", "#1F6E7A", "#D4A54D", "#828A9C"
SHOW_MM = (-4, -2, 0, 2, 4)
SBR_OFFSET_MM = 1.0   # SbR slice N pairs histology with the released MRI, which sits 1.0 mm off (scripts/02); all
                      # three methods in scripts/18 find the same +1.0 mm, so truth = SbR plane + 1.0 mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=int, default=142)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    geom = sbr_geometry()
    y0, dy = geom[2], geom[3]
    df = pd.read_csv(ROOT / "results/sbr_slice_matching.csv").set_index("slice")
    prof = np.load(ROOT / "results/sbr_slice_profiles.npz")
    offs = prof["offsets"]

    # example: re-run the two stages for the chosen slice to get its pose
    vh, vm = volumes.load("bigbrain_400", dev), volumes.load("bigbrain_mri", dev)
    _, h, hm, m, mm = sbr.load_pieces("BigBrain", a.slice)[0]
    bh, _ = localise(h, (s18.PX, s18.PX), vh, metric="ncc", coarse_kw=s18.COARSE)
    match = refine(h, (s18.PX, s18.PX), vm, [bh[0]], metric="mi", levels=((1.0, 60, 0.2), (0.5, 40, 0.1)))[0]
    p = match.pose
    n = rotation(torch.tensor([[p.pitch, p.yaw, p.roll]], dtype=torch.float32))[0].numpy()[:, 1]
    base = np.array([p.x, p.y, p.z, p.pitch, p.yaw, p.roll, p.log_s], np.float32)
    sec = to_isotropic(h, (s18.PX, s18.PX), 0.5, dev)
    sec = (sec.flip(1) if match.flipped else sec).cpu().numpy()
    planes = []
    with torch.no_grad():
        for d in SHOW_MM:
            q = base.copy()
            q[:3] += d * n
            planes.append(sample(_volume_level(vm, 0.5), torch.tensor(q[None], device=dev), sec.shape, 0.5)[0].cpu().numpy())
    y_true = y0 + dy * a.slice + SBR_OFFSET_MM
    y_found = float(df.loc[a.slice, "y_true"] + df.loc[a.slice, "hist_to_mri_y_signed_mm"])
    true_at = y_true - y_found                                           # offset (mm) of the true slice from ours

    fig, axes = plt.subplots(1, 6, figsize=(21, 4.4), facecolor="white", gridspec_kw=dict(wspace=0.06))
    ax = axes[0]
    ax.imshow(1 - sec, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"raw section {a.slice}", fontsize=19, color=NAVY)
    ax.axis("off")
    for ax, d, pl in zip(axes[1:], SHOW_MM, planes):
        found = d == 0
        ax.imshow(pl, cmap="gray")
        ax.set_title("found slice" if found else f"MRI {d:+d} mm", fontsize=19, color=TEAL if found else NAVY,
                     fontweight="bold" if found else "normal")
        for sp in ax.spines.values():
            sp.set_edgecolor(TEAL if found else "white")
            sp.set_linewidth(6 if found else 0)
        ax.set_xticks([])
        ax.set_yticks([])
        if abs(d - true_at) < 1.0:
            ax.set_xlabel("= true slice", fontsize=18, color="#9A6F14", fontweight="bold")
    fig.savefig(ROOT / "figures/sbr_slice_neighbours.png", dpi=100, bbox_inches="tight", facecolor="white")

    fig = plt.figure(figsize=(8, 5.6), facecolor="white")
    # profiles against distance from the true slice
    ax = fig.add_subplot(1, 1, 1)
    grid = np.arange(-5.0, 5.01, 0.2)
    curves = []
    for k in prof.files:
        if not k.startswith("s"):
            continue
        sid = int(k[1:])
        if sid not in df.index or df.loc[sid, "ref_dice"] < 0.9:
            continue
        shift = df.loc[sid, "hist_to_mri_y_signed_mm"] - SBR_OFFSET_MM   # found - true
        x = offs + shift
        c = prof[k]
        c = (c - c.min()) / max(c.max() - c.min(), 1e-9)
        curves.append(np.interp(grid, x, c, left=np.nan, right=np.nan))
    C = np.array(curves)
    med, q1, q3 = np.nanmedian(C, 0), np.nanpercentile(C, 25, 0), np.nanpercentile(C, 75, 0)
    ax.fill_between(grid, q1, q3, color=TEAL, alpha=0.18, lw=0)
    ax.plot(grid, med, color=TEAL, lw=3, label=f"all sections (n={len(C)}), median")
    ax.axvline(0, color=ORANGE, lw=2.5, ls="--", label="true MRI slice")
    ax.set_xlabel("distance from the true MRI slice (mm)", fontsize=15, color=NAVY)
    ax.set_ylabel("similarity to the section (scaled)", fontsize=15, color=NAVY)
    ax.tick_params(labelsize=13)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=13, frameon=False, loc="lower center")
    fig.savefig(ROOT / "figures/sbr_slice_profile.png", dpi=100, bbox_inches="tight", facecolor="white")
    print("figures/sbr_slice_neighbours.png, figures/sbr_slice_profile.png", f"true slice at {true_at:+.2f} mm from ours")


if __name__ == "__main__":
    main()
