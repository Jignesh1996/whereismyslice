#!/usr/bin/env python3
"""Where do the SbR BigBrain MRI slices sit in BigBrain's 3D MRI? Localise each 2D slice (MRI vs MRI, NCC)
with no prior, then fit y(slice number) and read the pixel size from the fitted scale.

This links the dataset's slice numbers to 3D positions (needed later to test histology->MRI slice matching)
and checks the 0.4 mm/px assumption used to report in-plane errors in mm.
Output: results/sbr_bigbrain_slice_positions.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from wms import volumes
from wms.classical import localise

ROOT = Path(__file__).resolve().parents[1]
SBR = ROOT / "data/sbr/repo/data/BigBrain/mri"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="*", default=list(range(20, 361, 20)))
    ap.add_argument("--px-guess", type=float, default=0.4)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vol = volumes.load("bigbrain_mri", dev)
    rows = []
    for sid in a.ids:
        f = f"slice_{sid:03d}.png"
        img = np.asarray(Image.open(SBR / "images" / f).convert("L"), np.float32) / 255
        img *= np.asarray(Image.open(SBR / "labels" / f).convert("L")) > 0
        best, _ = localise(img, (a.px_guess, a.px_guess), vol, metric="ncc",
                           coarse_kw=dict(rolls=(-6, 0, 6), try_flip=True))
        p = best[0].pose
        rows.append(dict(slice=sid, x=p.x, y=p.y, z=p.z, pitch=p.pitch, yaw=p.yaw, roll=p.roll, px_mm=a.px_guess * np.exp(p.log_s),
                         flipped=best[0].flipped, ncc=best[0].score))
        print(f"slice {sid:03d}: {p}  flip={best[0].flipped}  NCC {best[0].score:.3f}", flush=True)
    df = pd.DataFrame(rows)
    b, c = np.polyfit(df.slice, df.y, 1)
    df["resid_mm"] = df.y - (c + b * df.slice)
    df.to_csv(ROOT / "results/sbr_bigbrain_slice_positions.csv", index=False)
    print(f"\ny = {c:.2f} + {b:.4f} mm x slice  (|resid| median {df.resid_mm.abs().median():.2f} mm, max {df.resid_mm.abs().max():.2f})"
          f"\npixel size {df.px_mm.median():.4f} mm (range {df.px_mm.min():.4f}-{df.px_mm.max():.4f}); "
          f"BigBrain section of slice N ~ {(c + 70.01) / 0.02:.0f} + {b / 0.02:.2f} N; flips {df.flipped.sum()}")


if __name__ == "__main__":
    main()
