#!/usr/bin/env python3
"""Stress test of in-plane localisation with synthetic damage and exact ground truth.

Each real SbR BigBrain histology section (trusted reference, Dice >= 0.9) is damaged with a known distortion
(rotation over 360 deg, 5-10 % affine stretch/shear, 5-10 % elastic deformation, tears and missing chunks), then
localised in its MRI slice with no initial transform (affine) and refined with a B-spline.

Truth: an MRI pixel p belongs at original histology pixel T_ref p; our mapping sends p to a distorted pixel q,
whose original location is G(q) (known exactly). Error = |A_ref^-1 (G(q) - T_ref p)| in MRI pixels x 0.4 mm,
over the MRI brain mask. The clean condition measures the floor (method vs reference with no damage).
Output: results/stress_inplane.csv, figures/stress_inplane.png
"""
import argparse
import faulthandler
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from wms.classical import local_align
from wms.distort import apply, random_distortion
from wms.inplane import localise, warp

sbr = import_module("08_sbr_inplane")
ROOT = Path(__file__).resolve().parents[1]
PX_MM = 0.4
CONDITIONS = {
    "clean": dict(rotation=False),
    "rotation 360": dict(rotation=True),
    "affine 5%": dict(rotation=False, affine=0.05),
    "affine 10%": dict(rotation=False, affine=0.10),
    "elastic 5%": dict(rotation=False, elastic=0.05),
    "elastic 10%": dict(rotation=False, elastic=0.10),
    "cuts + tears": dict(rotation=False, cuts=True),
    "all 5%": dict(rotation=True, affine=0.05, elastic=0.05, cuts=True),
    "all 10%": dict(rotation=True, affine=0.10, elastic=0.10, cuts=True),
}


def mappings(T_aff, field, dist, mm):
    """Original-histology location (2, n) that our affine and affine+B-spline assign to each MRI brain pixel."""
    r, c = np.nonzero(mm)
    p = np.stack([c, r]).astype(float)
    out = {}
    for name, f in (("affine", None), ("bspline", field)):
        pp = p if f is None else p + np.stack([f[r, c, 0], f[r, c, 1]])
        q = T_aff[:2, :2] @ pp + T_aff[:2, 2:3]
        out[name] = np.stack(dist.G(q[0], q[1]))
    return out


def surviving(dist, T_ref, mm, orig_shape):
    """MRI brain pixels whose true histology point was not cut away (cut pixels mapped back to the original, dilated)."""
    from scipy.ndimage import binary_dilation
    yy, xx = np.nonzero(dist.cut)
    gone = np.zeros(orig_shape, bool)
    if len(xx):
        ox, oy = dist.G(xx.astype(float), yy.astype(float))
        ok = (ox >= 0) & (oy >= 0) & (ox < orig_shape[1] - 1) & (oy < orig_shape[0] - 1)
        gone[np.round(oy[ok]).astype(int), np.round(ox[ok]).astype(int)] = True
        gone = binary_dilation(gone, iterations=2)
    r, c = np.nonzero(mm)
    t = T_ref[:2, :2] @ np.stack([c, r]).astype(float) + T_ref[:2, 2:3]
    ti = np.clip(np.round(t).astype(int), 0, [[orig_shape[1] - 1], [orig_shape[0] - 1]])
    return ~gone[ti[1], ti[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ref = pd.read_csv(ROOT / "results/sbr_BigBrain_inplane.csv")
    ids = sorted(np.random.default_rng(a.seed).choice(ref[ref.dice_reference >= 0.9].slice.values, a.n, replace=False))
    out = ROOT / "results/stress_inplane.csv"
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {(int(r["slice"]), r["condition"]) for r in rows}
    angles = np.arange(-180, 180, 4)
    cache = ROOT / "results/stress_cache"
    cache.mkdir(parents=True, exist_ok=True)
    clean = {}
    for sid in ids:
        _, h, hm, m, mm = sbr.load_pieces("BigBrain", int(sid))[0]
        T_ref, _ = sbr.reference_affine("BigBrain", int(sid), "whole", h, hm, "cpu")
        for ci, cond in enumerate(a.conditions):
            if (int(sid), cond) in done:
                continue
            faulthandler.dump_traceback_later(300, exit=True, file=sys.stderr)
            rng = np.random.default_rng([a.seed, int(sid), ci])
            dist = random_distortion(h.shape, hm, rng, **CONDITIONS[cond])
            hd, hmd = apply(dist, h, hm)
            t0 = time.time()
            res = localise(hd, hmd, m, mm, device=dev, angles=angles)
            moved = warp(torch.tensor(hd)[None], torch.tensor(res.T[None], dtype=torch.float32), m.shape)[0].numpy()
            moved_mask = warp(torch.tensor(hmd, dtype=torch.float32)[None], torch.tensor(res.T[None], dtype=torch.float32),
                              m.shape)[0].numpy() > 0.5
            try:                                                            # metric only where both sections have tissue
                _, field, _ = local_align(moved, m, 1.0, mesh_mm=40.0, metric="mi", iterations=60, fixed_mask=mm & moved_mask)
            except RuntimeError:
                _, field, _ = local_align(moved, m, 1.0, mesh_mm=40.0, metric="mi", iterations=60)
            secs = time.time() - t0
            maps = mappings(res.T, field, dist, mm)
            keep = surviving(dist, T_ref, mm, h.shape)
            r_, c_ = np.nonzero(mm)
            target = T_ref[:2, :2] @ np.stack([c_, r_]).astype(float) + T_ref[:2, 2:3]
            Ainv = np.linalg.inv(T_ref[:2, :2])
            dist_mm = lambda a, b: np.linalg.norm(Ainv @ (a - b), axis=0)[keep] * PX_MM
            if cond == "clean":
                clean[int(sid)] = maps
                np.savez(cache / f"slice_{sid:03d}.npz", **maps)
            elif int(sid) not in clean and (cache / f"slice_{sid:03d}.npz").exists():
                clean[int(sid)] = dict(np.load(cache / f"slice_{sid:03d}.npz"))
            ref_aff, ref_bs = dist_mm(maps["affine"], target), dist_mm(maps["bspline"], target)
            cons = {k: dist_mm(maps[k], clean[int(sid)][k]) for k in maps} if int(sid) in clean else None
            row = dict(slice=int(sid), condition=cond, rotation_deg=dist.params["rotation_deg"], surviving=keep.mean(),
                       affine_mean_mm=ref_aff.mean(), affine_p95_mm=np.percentile(ref_aff, 95),
                       bspline_mean_mm=ref_bs.mean(), bspline_p95_mm=np.percentile(ref_bs, 95),
                       affine_vs_clean_mm=cons["affine"].mean() if cons else np.nan,
                       bspline_vs_clean_mm=cons["bspline"].mean() if cons else np.nan, seconds=secs)
            rows.append(row)
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"slice {sid:03d} {cond:13s} rot {row['rotation_deg']:7.1f}  vs truth: affine {row['affine_mean_mm']:5.2f} "
                  f"B-spline {row['bspline_mean_mm']:5.2f} mm | vs own clean: affine {row['affine_vs_clean_mm']:5.2f} "
                  f"B-spline {row['bspline_vs_clean_mm']:5.2f} mm  {secs:.1f}s", flush=True)
    faulthandler.cancel_dump_traceback_later()
    df = pd.DataFrame(rows)
    df["best_mm"] = df[["affine_mean_mm", "bspline_mean_mm"]].min(axis=1)
    summ = df.groupby("condition", sort=False).agg(
        n=("slice", "size"), affine_vs_truth=("affine_mean_mm", "median"), bspline_vs_truth=("bspline_mean_mm", "median"),
        affine_vs_clean=("affine_vs_clean_mm", "median"), bspline_vs_clean=("bspline_vs_clean_mm", "median"),
        best_within_1mm=("best_mm", lambda s: (s < 1).mean()), best_within_2mm=("best_mm", lambda s: (s < 2).mean()),
        failures_over_5mm=("affine_mean_mm", lambda s: int((s > 5).sum())))
    print("\n" + summ.reindex([c for c in CONDITIONS if c in summ.index]).round(2).to_string())


if __name__ == "__main__":
    main()
