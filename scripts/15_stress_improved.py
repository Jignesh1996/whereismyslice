#!/usr/bin/env python3
"""Stress test, improved whole-section in-plane alignment: same sections, damage and seeds as scripts/11.

* affine       scripts/11 affine (localise)
* cut_aware    MRI tissue the aligned section does not cover (cut chunks, tears) is dropped from the MRI mask and
               the affine refined again, so mask Dice stops pulling the section into the hole
* deform       regularised GPU displacement field on top of cut_aware (MI + mask Dice - strain), gated: kept only
               when it clearly improves the match (inplane.deform)
Errors vs truth (exact synthetic ground truth composed with the reference affine) and vs the same method on the
undamaged section (how much the damage moves the answer). The scripts/11 SimpleITK B-spline columns are merged in.
Output: results/stress_improved.csv (or --out)
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
from wms.distort import apply, random_distortion
from wms.inplane import deform, localise, localise_cut_aware

sbr = import_module("08_sbr_inplane")
s11 = import_module("11_synthetic_stress")
ROOT = Path(__file__).resolve().parents[1]
PX_MM = s11.PX_MM
METHODS = ("affine", "cut_aware", "deform")


def mapping(T, field, dist, mm):
    r, c = np.nonzero(mm)
    p = np.stack([c, r]).astype(float)
    if field is not None:
        p = p + np.stack([field[r, c, 0], field[r, c, 1]])
    q = T[:2, :2] @ p + T[:2, 2:3]
    return np.stack(dist.G(q[0], q[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", nargs="*", default=list(s11.CONDITIONS))
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--gate", type=float, default=0.01)
    ap.add_argument("--out", default="results/stress_improved.csv")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ref = pd.read_csv(ROOT / "results/sbr_BigBrain_inplane.csv")
    ids = sorted(np.random.default_rng(a.seed).choice(ref[ref.dice_reference >= 0.9].slice.values, 30, replace=False))[:a.n]
    out = ROOT / a.out
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {(int(r["slice"]), r["condition"]) for r in rows}
    angles = np.arange(-180, 180, 4)
    all_conds = list(s11.CONDITIONS)
    conds = ["clean"] + [c for c in a.conditions if c != "clean"]          # clean first: it is the consistency anchor
    for sid in ids:
        if all((int(sid), c) in done for c in conds):                     # resume: slice finished
            continue
        _, h, hm, m, mm = sbr.load_pieces("BigBrain", int(sid))[0]
        T_ref, _ = sbr.reference_affine("BigBrain", int(sid), "whole", h, hm, "cpu")
        r_, c_ = np.nonzero(mm)
        target = T_ref[:2, :2] @ np.stack([c_, r_]).astype(float) + T_ref[:2, 2:3]
        Ainv = np.linalg.inv(T_ref[:2, :2])
        clean = None
        for cond in conds:
            faulthandler.dump_traceback_later(300, exit=True, file=sys.stderr)
            ci = all_conds.index(cond)                                       # same rng stream as scripts/11
            rng = np.random.default_rng([a.seed, int(sid), ci])
            dist = random_distortion(h.shape, hm, rng, **s11.CONDITIONS[cond])
            hd, hmd = apply(dist, h, hm)
            t0 = time.time()
            res = localise(hd, hmd, m, mm, device=dev, angles=angles)
            res_c, mm_eff = localise_cut_aware(hd, hmd, m, mm, device=dev, res=res)
            t_aff = time.time() - t0
            field, gain = deform(hd, hmd, m * mm_eff, mm_eff, res_c.T, lam=a.lam, gate=a.gate, device=dev)
            maps = {"affine": mapping(res.T, None, dist, mm), "cut_aware": mapping(res_c.T, None, dist, mm),
                    "deform": mapping(res_c.T, field, dist, mm)}
            keep = s11.surviving(dist, T_ref, mm, h.shape)
            d = lambda x, y: np.linalg.norm(Ainv @ (x - y), axis=0)[keep] * PX_MM
            if cond == "clean":
                clean = maps
            row = dict(slice=int(sid), condition=cond, surviving=keep.mean(), cut_px=int((mm & ~mm_eff).sum()),
                       deform_gain=gain, deform_kept=bool(np.abs(field).max() > 0), seconds=time.time() - t0, affine_s=t_aff)
            for k in METHODS:
                e = d(maps[k], target)
                row[f"{k}_mean_mm"], row[f"{k}_p95_mm"] = e.mean(), np.percentile(e, 95)
                row[f"{k}_vs_clean_mm"] = d(maps[k], clean[k]).mean()
            if (int(sid), cond) not in done:
                rows.append(row)
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"slice {sid:03d} {cond:13s} vs clean: affine {row['affine_vs_clean_mm']:5.2f} cut-aware "
                  f"{row['cut_aware_vs_clean_mm']:5.2f} deform {row['deform_vs_clean_mm']:5.2f} (gain {gain:+.3f}"
                  f"{'' if row['deform_kept'] else ', gated off'}) | vs truth: {row['affine_mean_mm']:5.2f} "
                  f"{row['cut_aware_mean_mm']:5.2f} {row['deform_mean_mm']:5.2f}  {row['seconds']:.1f}s", flush=True)
    faulthandler.cancel_dump_traceback_later()
    summarise(pd.DataFrame(rows))


def summarise(df):
    old = pd.read_csv(ROOT / "results/stress_inplane.csv")
    df = df.merge(old[["slice", "condition", "bspline_mean_mm", "bspline_vs_clean_mm"]], on=["slice", "condition"], how="left")
    cols = ["affine", "cut_aware", "deform", "bspline"]
    order = [c for c in s11.CONDITIONS if c in set(df.condition)]
    print("\nmedian shift vs own clean result (mm)\n" + df.groupby("condition")[[f"{c}_vs_clean_mm" for c in cols]]
          .median().reindex(order).round(2).to_string())
    print("\nmedian error vs truth (mm)\n" + df.groupby("condition")[[f"{c}_mean_mm" for c in cols]]
          .median().reindex(order).round(2).to_string())
    print("\ndeformation kept (fraction)\n" + df.groupby("condition").deform_kept.mean().reindex(order).round(2).to_string())


if __name__ == "__main__":
    main()
