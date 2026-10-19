#!/usr/bin/env python3
"""Classical pose estimation on synthetic sections with known ground truth.

Queries: oblique planes resliced from BigBrain histology (400 um) at random poses, with histology artefacts.
Targets: the same histology volume (same modality) and BigBrain's own MRI (histology -> MRI).
Output: results/classical_synthetic.csv and a printed summary (median offset mm / tilt deg).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wms import volumes
from wms.augment import augment
from wms.classical import localise
from wms.planes import random_poses, pose_error, sample

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--targets", default="bigbrain_400,bigbrain_mri")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(a.seed)
    src = volumes.load("bigbrain_400", dev)
    lo, hi = src.bounds_mm
    poses = random_poses(a.n, (lo, hi), max_tilt=(10, 10, 10), margin_mm=(50, 15, 40), rng=rng)
    px = (0.4, 0.4)
    queries = []
    with torch.no_grad():
        for p in poses:
            img = sample(src, [p], (300, 350), 0.4)[0].cpu().numpy()          # 120 x 140 mm field of view
            queries.append(augment(img, rng))
    rows = []
    for tname in a.targets.split(","):
        tgt = src if tname == "bigbrain_400" else volumes.load(tname, dev)
        metric = "ncc" if tgt.modality == "histology" else "mi"
        for i, (p, q) in enumerate(zip(poses, queries)):
            t0 = time.time()
            best, cands = localise(q, px, tgt, metric=metric)
            off, tilt = pose_error(p, best[0].pose)
            off0, tilt0 = pose_error(p, cands[0].pose)
            rows.append(dict(target=tname, i=i, offset_mm=off, tilt_deg=tilt, coarse_offset_mm=off0,
                             coarse_tilt_deg=tilt0, score=best[0].score, seconds=time.time() - t0,
                             true=str(p), est=str(best[0].pose)))
            print(f"{tname:12s} #{i:02d} offset {off:6.2f} mm  tilt {tilt:5.2f} deg  (coarse {off0:5.1f} / {tilt0:4.1f})  "
                  f"{time.time() - t0:4.1f}s  true {p}\n{'':16s}est {best[0].pose}", flush=True)
        if tgt is not src:
            del tgt
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results/classical_synthetic.csv", index=False)
    summ = df.groupby("target").agg(median_offset_mm=("offset_mm", "median"), median_tilt_deg=("tilt_deg", "median"),
                                    within_2mm=("offset_mm", lambda s: (s < 2).mean()), sec=("seconds", "mean"))
    print(summ.round(2))
    if dev == "cuda":
        print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
