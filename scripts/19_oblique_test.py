#!/usr/bin/env python3
"""Oblique sections: does the search recover the cutting angle as well as the plane?

Queries are resliced from the BigBrain histology volume (400 um, with histology artefacts) at a fixed tilt away
from the coronal plane (0, 15, 30, 45 deg, random tilt direction, roll +-10 deg) and localised with no prior in
the BigBrain MRI (histology -> MRI) and in the histology volume. The coarse grid covers pitch/yaw +-48 deg.
Errors: angle between true and found plane normals, and distance from the true section centre to the found plane.
Output: results/oblique_test.csv, results/oblique_examples.npz (one example per tilt, for the figure)
"""
import argparse
import faulthandler
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wms import volumes
from wms.augment import augment
from wms.classical import localise
from wms.planes import Pose, pose_error, sample

ROOT = Path(__file__).resolve().parents[1]
ANGLES = tuple(np.arange(-48, 49, 8).tolist())
SHAPE, PX = (300, 350), 0.4


def tilt_from_coronal(p):
    return float(np.degrees(np.arccos(min(1.0, abs(p.normal[1])))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--tilts", type=float, nargs="*", default=[0, 15, 30, 45])
    ap.add_argument("--targets", default="bigbrain_mri,bigbrain_400")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(a.seed)
    src = volumes.load("bigbrain_400", dev)
    lo, hi = src.bounds_mm
    lo_c, hi_c = np.asarray(lo) + (50, 25, 40), np.asarray(hi) - (50, 25, 40)
    poses, queries, tilts = [], [], []
    for t in a.tilts:
        for _ in range(a.n):
            az = rng.uniform(0, 2 * np.pi)
            c = rng.uniform(lo_c, hi_c)
            p = Pose(*c, t * np.cos(az), t * np.sin(az), rng.uniform(-10, 10))
            with torch.no_grad():
                img = sample(src, [p], SHAPE, PX)[0].cpu().numpy()
            poses.append(p)
            queries.append(augment(img, rng))
            tilts.append(t)
    rows, examples = [], {}
    for tname in a.targets.split(","):
        tgt = src if tname == "bigbrain_400" else volumes.load(tname, dev)
        metric = "ncc" if tgt.modality == "histology" else "mi"
        for i, (p, q, t) in enumerate(zip(poses, queries, tilts)):
            faulthandler.dump_traceback_later(300, exit=True, file=sys.stderr)
            t0 = time.time()
            best, _ = localise(q, (PX, PX), tgt, metric=metric, coarse_kw=dict(angles=ANGLES))
            e = best[0].pose
            off, ang = pose_error(p, e)
            rows.append(dict(target=tname, i=i, tilt_set=t, tilt_true=tilt_from_coronal(p), tilt_found=tilt_from_coronal(e),
                             angle_err_deg=ang, offset_mm=off, centre_err_mm=float(np.linalg.norm(p.centre - e.centre)),
                             score=best[0].score, seconds=time.time() - t0, true=str(p), est=str(e)))
            print(f"{tname:12s} tilt {t:4.0f} #{i:02d}  true tilt {rows[-1]['tilt_true']:5.1f} found {rows[-1]['tilt_found']:5.1f} deg | "
                  f"angle err {ang:5.2f} deg  plane offset {off:5.2f} mm  {time.time() - t0:4.1f}s", flush=True)
            key = f"{tname}_{int(t)}"
            if tname == "bigbrain_mri" and key not in examples:
                with torch.no_grad():
                    found = sample(tgt, [e], SHAPE, PX)[0].cpu().numpy()
                    true = sample(tgt, [p], SHAPE, PX)[0].cpu().numpy()
                examples[f"query_{int(t)}"] = q
                examples[f"found_{int(t)}"] = found[:, ::-1] if best[0].flipped else found
                examples[f"true_{int(t)}"] = true
                examples[f"info_{int(t)}"] = np.array([rows[-1]["tilt_true"], rows[-1]["tilt_found"], ang, off])
        if tgt is not src:
            del tgt
            torch.cuda.empty_cache()
    faulthandler.cancel_dump_traceback_later()
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/oblique_test.csv", index=False)
    np.savez(ROOT / "results/oblique_examples.npz", **examples)
    summ = df.groupby(["target", "tilt_set"]).agg(
        n=("i", "size"), angle_err_deg=("angle_err_deg", "median"), offset_mm=("offset_mm", "median"),
        within_2mm_5deg=("offset_mm", lambda s: np.mean((s < 2) & (df.loc[s.index, "angle_err_deg"] < 5))),
        seconds=("seconds", "median"))
    print(summ.round(2).to_string())


if __name__ == "__main__":
    main()
