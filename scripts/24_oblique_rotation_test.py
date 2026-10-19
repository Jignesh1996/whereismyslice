#!/usr/bin/env python3
"""Full pose of an obliquely cut, arbitrarily rotated histology section in the MRI: tilt, in-plane rotation and
alignment, all scored.

Queries: resliced from the BigBrain histology volume (400 um, histology artefacts) at a tilt of 0/15/30/45 deg from
coronal (random direction) and an in-plane rotation drawn from the full 360 deg. Searched in the BigBrain MRI with
no prior; the coarse grid covers tilt +-48 deg and rotation over 360 deg (15 deg steps), then gradient refinement.
Scores: tilt error (angle between plane normals), rotation error (in-plane), plane offset, and alignment error =
mean 3D distance between where each tissue pixel truly is and where the found pose puts it.
Output: results/oblique_rotation_test.csv, results/oblique_rotation_examples.npz (poses + images for figures)
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
from wms.planes import Pose, as_params, plane_points, pose_error, rotation, sample

ROOT = Path(__file__).resolve().parents[1]
SHAPE, PX = (300, 350), 0.4
COARSE = dict(angles=tuple(range(-48, 49, 12)), rolls=tuple(range(-180, 180, 15)))


def tilt_from_coronal(p):
    return float(np.degrees(np.arccos(min(1.0, abs(p.normal[1])))))


def errors(true, match, mask):
    """tilt deg, rotation deg, plane offset mm, alignment mm (mean over tissue pixels)."""
    off, tilt = pose_error(true, match.pose)
    pt = plane_points(as_params([true], "cpu"), SHAPE, PX)[0].numpy()
    pe = plane_points(as_params([match.pose], "cpu"), SHAPE, PX)[0].numpy()
    if match.flipped:                                  # the search matched the mirrored section
        pe = pe[:, ::-1]
    align = float(np.linalg.norm(pt - pe, axis=-1)[mask].mean())
    Rt = rotation(torch.tensor([[true.pitch, true.yaw, true.roll]], dtype=torch.float32))[0].numpy()
    Re = rotation(torch.tensor([[match.pose.pitch, match.pose.yaw, match.pose.roll]], dtype=torch.float32))[0].numpy()
    ue = -Re[:, 0] if match.flipped else Re[:, 0]
    rot = float(np.degrees(np.arccos(np.clip(Rt[:, 0] @ ue, -1, 1))))
    return tilt, rot, off, align


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--tilts", type=float, nargs="*", default=[0, 15, 30, 45])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(a.seed)
    src = volumes.load("bigbrain_400", dev)
    mri = volumes.load("bigbrain_mri", dev)
    lo, hi = src.bounds_mm
    lo_c, hi_c = np.asarray(lo) + (50, 25, 40), np.asarray(hi) - (50, 25, 40)
    out = ROOT / "results/oblique_rotation_test.csv"
    rows, ex = [], {}
    for t in a.tilts:
        for i in range(a.n):
            faulthandler.dump_traceback_later(600, exit=True, file=sys.stderr)
            az = rng.uniform(0, 2 * np.pi)
            p = Pose(*rng.uniform(lo_c, hi_c), t * np.cos(az), t * np.sin(az), rng.uniform(-180, 180))
            with torch.no_grad():
                img = sample(src, [p], SHAPE, PX)[0].cpu().numpy()
            q = augment(img, rng)
            mask = img > 0.08
            t0 = time.time()
            best, _ = localise(q, (PX, PX), mri, metric="mi", coarse_kw=COARSE)
            m = best[0]
            tilt, rot, off, align = errors(p, m, mask)
            rows.append(dict(tilt_set=t, i=i, tilt_true=tilt_from_coronal(p), tilt_found=tilt_from_coronal(m.pose),
                             roll_true=p.roll, tilt_err_deg=tilt, rot_err_deg=rot, offset_mm=off, align_mm=align,
                             flipped=m.flipped, score=m.score, seconds=time.time() - t0, true=str(p), est=str(m.pose)))
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"tilt {t:4.0f} #{i}  true tilt {rows[-1]['tilt_true']:5.1f} rot {p.roll:7.1f} | found tilt "
                  f"{rows[-1]['tilt_found']:5.1f} | errors: tilt {tilt:5.2f} deg  rotation {rot:6.2f} deg  plane {off:5.2f} mm  "
                  f"alignment {align:5.2f} mm  {time.time() - t0:.0f}s", flush=True)
            if f"query_{int(t)}" not in ex:
                with torch.no_grad():
                    found = sample(mri, [m.pose], SHAPE, PX)[0].cpu().numpy()
                ex[f"query_{int(t)}"] = q
                ex[f"found_{int(t)}"] = found[:, ::-1] if m.flipped else found
                ex[f"true_pose_{int(t)}"] = p.as_array()
                ex[f"est_pose_{int(t)}"] = m.pose.as_array()
                ex[f"flipped_{int(t)}"] = np.array(m.flipped)
                ex[f"err_{int(t)}"] = np.array([tilt, rot, off, align])
                np.savez(ROOT / "results/oblique_rotation_examples.npz", **ex)
    faulthandler.cancel_dump_traceback_later()
    df = pd.DataFrame(rows)
    ok = (df.tilt_err_deg < 5) & (df.rot_err_deg < 5) & (df.align_mm < 2)
    df["ok"] = ok
    print(df.groupby("tilt_set").agg(n=("i", "size"), tilt_err=("tilt_err_deg", "median"), rot_err=("rot_err_deg", "median"),
                                     offset=("offset_mm", "median"), align=("align_mm", "median"), ok=("ok", "mean"),
                                     sec=("seconds", "median")).round(2).to_string())


if __name__ == "__main__":
    main()
