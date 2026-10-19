#!/usr/bin/env python3
"""Good first issue #1: check that each histology volume lines up with its MRI.

Renders coronal / axial / sagittal slices of both pairs with MRI edges on top of histology, and prints
the normalised mutual information of the pair against the same pair shifted by 5 mm (aligned should win).
Output: figures/alignment_<space>.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import canny

from wms import volumes
from wms.planes import Pose, sample

OUT = Path(__file__).resolve().parents[1] / "figures"
PAIRS = [("bigbrain_400", "bigbrain_mri"), ("bigbrain_mni_400", "icbm_t1")]


def nmi(a, b, bins=48):
    m = (a > 0.02) & (b > 0.02)
    h, _, _ = np.histogram2d(a[m], b[m], bins=bins)
    p = h / h.sum()
    px, py = p.sum(1), p.sum(0)
    ent = lambda q: -np.sum(q[q > 0] * np.log(q[q > 0]))
    return (ent(px) + ent(py)) / ent(p)


def main():
    OUT.mkdir(exist_ok=True)
    for hname, mname in PAIRS:
        h, m = volumes.load(hname), volumes.load(mname)
        lo, hi = h.bounds_mm
        c = (lo + hi) / 2
        views = {"coronal": Pose(c[0], c[1], c[2]), "axial": Pose(c[0], c[1], c[2], pitch=-90),
                 "sagittal": Pose(c[0], c[1], c[2], yaw=90)}
        fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), facecolor="white")
        for j, (title, pose) in enumerate(views.items()):
            n = int(((hi - lo).max() + 10) / 0.4)
            a = sample(h, pose, (n, n), 0.4)[0].numpy()
            b = sample(m, pose, (n, n), 0.4)[0].numpy()
            shifted = Pose(*(pose.as_array() + [0, 0, 5, 0, 0, 0, 0]))
            b5 = sample(m, shifted, (n, n), 0.4)[0].numpy()
            score, score5 = nmi(a, b), nmi(a, b5)
            print(f"{hname:16s} vs {mname:12s} {title:8s}  NMI aligned {score:.3f}   shifted 5 mm {score5:.3f}")
            axes[0, j].imshow(a, cmap="gray")
            axes[0, j].contour(canny(b, sigma=2.5), levels=[0.5], colors="#ff4d4d", linewidths=0.6)
            axes[0, j].set_title(f"{title}: histology + MRI edges (NMI {score:.3f} vs {score5:.3f} @5 mm)", fontsize=10)
            axes[1, j].imshow(b, cmap="gray")
            axes[1, j].set_title(f"{mname}", fontsize=10)
            for ax in axes[:, j]:
                ax.axis("off")
        fig.suptitle(f"{h.space}: {hname} vs {mname}", fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT / f"alignment_{h.space}.png", dpi=90)
        plt.close(fig)


if __name__ == "__main__":
    main()
