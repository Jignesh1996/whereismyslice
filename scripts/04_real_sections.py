#!/usr/bin/env python3
"""Classical localisation of REAL BigBrain sections (thumbnails, ~0.26 mm/px) in the 3D histology volume and
in BigBrain's own MRI, with no prior on where the section is.

Sections are cut at 20 um, so the true y of section n is linear in n. A robust line fit over all sampled
sections against the histology volume gives the reference; each method is scored by its residual from
that line (mm) and by its plane tilt. Output: results/real_sections.csv, figures/real_sections.png
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from wms import sections, volumes
from wms.classical import localise

ROOT = Path(__file__).resolve().parents[1]


def robust_line(x, y, iters=5, k=3.0):
    keep = np.ones_like(x, bool)
    for _ in range(iters):
        b, a = np.polyfit(x[keep], y[keep], 1)
        r = y - (a + b * x)
        mad = np.median(np.abs(r[keep])) * 1.4826 + 1e-6
        keep = np.abs(r) < k * mad
    return a, b, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--first", type=int, default=300)
    ap.add_argument("--last", type=int, default=7100)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ns = list(range(a.first, a.last + 1, a.every))
    rows = []
    coarse = dict(angles=(-10, -5, 0, 5, 10), rolls=(-6, 0, 6), try_flip=True)
    for tname in ("bigbrain_400", "bigbrain_mri"):
        vol = volumes.load(tname, dev)
        metric = "ncc" if vol.modality == "histology" else "mi"
        for n in ns:
            img, px = sections.load(n)
            t0 = time.time()
            best, _ = localise(img, px, vol, metric=metric, coarse_kw=coarse)
            m = best[0]
            p = m.pose
            rows.append(dict(target=tname, n=n, x=p.x, y=p.y, z=p.z, pitch=p.pitch, yaw=p.yaw, roll=p.roll,
                             scale=np.exp(p.log_s), flipped=m.flipped, score=m.score, seconds=time.time() - t0))
            print(f"{tname:12s} pm{n:04d}  {p}  flip={m.flipped}  score {m.score:.3f}  {time.time() - t0:.1f}s", flush=True)
        del vol
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)

    h = df[df.target == "bigbrain_400"]
    a0, b0, keep = robust_line(h.n.values.astype(float), h.y.values)
    print(f"\nreference line from histology: y = {a0:.2f} + {b0 * 1000:.2f} um x n   (inliers {keep.sum()}/{len(keep)})")
    df["y_ref"] = a0 + b0 * df.n
    df["resid_mm"] = df.y - df.y_ref
    df["tilt_deg"] = np.degrees(np.arccos(np.cos(np.radians(df.pitch)) * np.cos(np.radians(df.yaw))))
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results/real_sections.csv", index=False)
    for t, g in df.groupby("target"):
        r = g.resid_mm.abs()
        print(f"{t:12s} |residual| median {r.median():.2f} mm, 90th pct {r.quantile(0.9):.2f} mm, within 2 mm {(r < 2).mean():.0%}; "
              f"tilt median {g.tilt_deg.median():.1f} deg; scale median {g.scale.median():.3f}; flips {g.flipped.sum()}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5), facecolor="white")
    for t, g in df.groupby("target"):
        ax[0].plot(g.n, g.y, "o", ms=4, label=t)
        ax[1].plot(g.n, g.resid_mm, "o-", ms=3, lw=0.8, label=t)
        ax[2].plot(g.n, g.pitch, "o-", ms=3, lw=0.8, label=f"{t} pitch")
        ax[2].plot(g.n, g.yaw, "s--", ms=3, lw=0.8, label=f"{t} yaw")
    ax[0].plot(ns, a0 + b0 * np.array(ns), "k-", lw=1, label=f"fit: {b0 * 1000:.1f} um/section")
    ax[0].set(xlabel="section number", ylabel="estimated y (mm)", title="Where is each real section?")
    ax[1].set(xlabel="section number", ylabel="y - line (mm)", title="residual from the 20 um cutting line", ylim=(-6, 6))
    ax[2].set(xlabel="section number", ylabel="deg", title="estimated plane tilt")
    for x in ax:
        x.legend(fontsize=8)
        x.grid(alpha=0.3)
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures/real_sections.png", dpi=90)


if __name__ == "__main__":
    main()
