#!/usr/bin/env python3
"""Pose + local alignment of full-resolution (20 um) BigBrain sections.

1. Pose: classical localise on the section at 0.26 mm (thumbnail scale), against the target volume.
2. Local: render the target at that pose on the section grid (0.2 mm), then B-spline register the section
   onto it (Mattes MI). Reports NCC / NMI / tissue Dice before and after and the displacement size.
Targets: BigBrain histology 200 um and BigBrain MRI. Output: figures/local_<n>.png, results/local_alignment.csv
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from wms import sections, volumes
from wms.classical import dice, local_align, localise, render_like
from wms.metrics import mutual_information, ncc

ROOT = Path(__file__).resolve().parents[1]
WORK_PX = 0.2


def scores(a, b):
    ta, tb = torch.tensor(a)[None], torch.tensor(b)[None]
    return float(ncc(ta, tb)), float(mutual_information(ta, tb)), float(dice(a, b))


def checker(a, b, n=8):
    H, W = a.shape
    m = ((np.arange(H)[:, None] * n // H + np.arange(W)[None] * n // W) % 2).astype(bool)
    return np.where(m, a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", type=int, nargs="+", default=[2000, 4000, 6000])
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    # (volume rendered for local alignment, volume used for the pose search, metric, label)
    targets = [("bigbrain_200", "bigbrain_400", "ncc", "histology 200 um"), ("bigbrain_mri", "bigbrain_mri", "mi", "BigBrain MRI")]
    for n in a.sections:
        full, px_full = sections.load(n, "full_res", reduce=5)        # 0.1 mm: still 2x finer than WORK_PX
        fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), facecolor="white")
        for row, (tname, pname, metric, label) in enumerate(targets):
            vol = volumes.load(tname, dev)
            pvol = vol if pname == tname else volumes.load(pname, dev)
            # pose at ~thumbnail scale (fast), then everything at WORK_PX
            small = torch.nn.functional.interpolate(torch.tensor(full)[None, None], scale_factor=0.4, mode="area")[0, 0].numpy()
            px_small = (px_full[0] / 0.4, px_full[1] / 0.4)
            best, _ = localise(small, px_small, pvol, metric=metric, coarse_kw=dict(rolls=(-6, 0, 6)))
            m = best[0]
            sec, plane = render_like(full, px_full, vol, m, WORK_PX)
            before = scores(sec, plane)
            warped, field, _ = local_align(sec, plane, WORK_PX, mesh_mm=10.0, metric="mi")
            after = scores(warped, plane)
            disp = np.linalg.norm(field, axis=-1)[sec > 0.08]
            rows.append(dict(section=n, target=tname, y=m.pose.y, pitch=m.pose.pitch, yaw=m.pose.yaw,
                             ncc_before=before[0], ncc_after=after[0], nmi_before=before[1], nmi_after=after[1],
                             dice_before=before[2], dice_after=after[2],
                             disp_median_mm=float(np.median(disp)), disp_p95_mm=float(np.percentile(disp, 95))))
            print(f"pm{n:04d} -> {tname:12s} {m.pose}\n   NCC {before[0]:.3f}->{after[0]:.3f}  NMI {before[1]:.3f}->{after[1]:.3f}  "
                  f"Dice {before[2]:.3f}->{after[2]:.3f}  displacement median {np.median(disp):.2f} mm, 95% {np.percentile(disp, 95):.2f} mm", flush=True)
            norm = lambda x: x / max(np.percentile(x, 99.5), 1e-6)
            axes[row, 0].imshow(plane, cmap="gray")
            axes[row, 0].set_title(f"{label} at estimated pose\ny={m.pose.y:.2f} mm, pitch {m.pose.pitch:.1f}, yaw {m.pose.yaw:.1f} deg", fontsize=10)
            axes[row, 1].imshow(checker(norm(sec), norm(plane)), cmap="gray")
            axes[row, 1].set_title(f"pose only: checkerboard (Dice {before[2]:.3f})", fontsize=10)
            axes[row, 2].imshow(checker(norm(warped), norm(plane)), cmap="gray")
            axes[row, 2].set_title(f"+ local B-spline (Dice {after[2]:.3f}, 95% disp {np.percentile(disp, 95):.1f} mm)", fontsize=10)
            for ax in axes[row]:
                ax.axis("off")
            del vol, pvol
            torch.cuda.empty_cache()
        fig.suptitle(f"BigBrain section pm{n:04d} (20 um, shown at {WORK_PX} mm)", fontsize=13)
        fig.tight_layout()
        (ROOT / "figures").mkdir(exist_ok=True)
        fig.savefig(ROOT / f"figures/local_{n:04d}.png", dpi=80)
        plt.close(fig)
    # appends, so sections can run one process each (the 200 um volume is reloaded per section and host memory
    # fragments): for n in 2000 4000 6000; do python scripts/07_local_alignment.py --sections $n; done
    csv = ROOT / "results/local_alignment.csv"
    old = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    new = pd.DataFrame(rows)
    if len(old):
        old = old[~old.section.isin(new.section)]
    pd.concat([old, new]).sort_values(["section", "target"]).to_csv(csv, index=False)


if __name__ == "__main__":
    main()
