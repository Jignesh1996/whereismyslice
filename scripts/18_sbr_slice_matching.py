#!/usr/bin/env python3
"""Which MRI slice does a raw histology section belong to? Tested on the SbR BigBrain pairs, whose slice number N
gives the true MRI plane (y = y0 + dy * N, scripts/09), with no prior on position.

Two stages, plus a direct baseline:
* histology    stage 1: the raw section is located in the 3D BigBrain histology volume (same modality, NCC)
* hist_to_mri  stage 2: that pose starts a mutual-information refinement against the 3D MRI -> the MRI slice
* mri_direct   baseline: the raw section searched directly in the MRI (MI), as in scripts/14
Per method: y error, slice error |N_est - N| (one SbR slice = 0.4 mm), mean 3D error over the section's tissue
(truth from the reference affine). For the neighbourhood plots, the similarity of the section to MRI planes
shifted along the plane normal (+-6 mm) around the stage-2 pose is saved per slice.
Output: results/sbr_slice_matching.csv, results/sbr_slice_profiles.npz
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
from wms import volumes
from wms.classical import localise, refine, to_isotropic, _volume_level
from wms.metrics import mutual_information
from wms.patches import sbr_geometry, sbr_mri_world
from wms.planes import as_params, rotation, sample

sbr = import_module("08_sbr_inplane")
p14 = import_module("14_hierarchical_patch")
ROOT = Path(__file__).resolve().parents[1]
PX = p14.HISTO_PX
COARSE = dict(angles=(-10, -5, 0, 5, 10), rolls=(-10, 0, 10), try_flip=True)
OFFSETS = np.round(np.arange(-6.0, 6.01, 0.4), 2)


def profile(img, match, vol, px=1.0):
    """MI between the section and MRI planes moved along the plane normal by OFFSETS (mm)."""
    dev = vol.data.device
    p = match.pose
    n = rotation(torch.tensor([[p.pitch, p.yaw, p.roll]], dtype=torch.float32))[0].numpy()[:, 1]
    base = np.array([p.x, p.y, p.z, p.pitch, p.yaw, p.roll, p.log_s], np.float32)
    params = np.repeat(base[None], len(OFFSETS), 0)
    params[:, :3] += OFFSETS[:, None] * n[None]
    s = to_isotropic(img, (PX, PX), px, dev)
    s = s.flip(1) if match.flipped else s
    with torch.no_grad():
        planes = sample(_volume_level(vol, px), torch.tensor(params, device=dev), s.shape, px)
        return mutual_information(s[None].expand(len(params), -1, -1), planes).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    geom = sbr_geometry()
    y0, dy = geom[2], geom[3]
    vh = volumes.load("bigbrain_400", dev)
    vm = volumes.load("bigbrain_mri", dev)
    ref = pd.read_csv(ROOT / "results/sbr_BigBrain_inplane.csv")
    ids = sorted(ref.slice.unique())[::a.every]
    out = ROOT / "results/sbr_slice_matching.csv"
    prof_f = ROOT / "results/sbr_slice_profiles.npz"
    rows = pd.read_csv(out).to_dict("records") if out.exists() and a.resume else []
    profs = dict(np.load(prof_f)) if prof_f.exists() and a.resume else {}
    done = {int(r["slice"]) for r in rows}
    for sid in ids:
        if int(sid) in done:
            continue
        faulthandler.dump_traceback_later(300, exit=True, file=sys.stderr)
        _, h, hm, m, mm = sbr.load_pieces("BigBrain", int(sid))[0]
        T_ref, _ = sbr.reference_affine("BigBrain", int(sid), "whole", h, hm, "cpu")
        r_, c_ = np.nonzero(hm)
        pick = np.random.default_rng(0).choice(len(r_), min(4000, len(r_)), replace=False)
        hx, hy = c_[pick].astype(float), r_[pick].astype(float)
        mp = np.linalg.inv(T_ref) @ np.stack([hx, hy, np.ones_like(hx)])
        truth = sbr_mri_world(int(sid), mp[0], mp[1], m.shape, geom)
        row = dict(slice=int(sid), y_true=y0 + dy * int(sid), ref_dice=float(ref.set_index("slice").loc[sid, "dice_reference"]))
        t0 = time.time()
        best_h, _ = localise(h, (PX, PX), vh, metric="ncc", coarse_kw=COARSE)
        t_h = time.time() - t0
        best_hm = refine(h, (PX, PX), vm, [best_h[0]], metric="mi", levels=((1.0, 60, 0.2), (0.5, 40, 0.1)))
        t_hm = time.time() - t0
        best_m, _ = localise(h, (PX, PX), vm, metric="mi", coarse_kw=COARSE)
        t_m = time.time() - t0 - t_hm
        for name, match, secs in (("histology", best_h[0], t_h), ("hist_to_mri", best_hm[0], t_hm), ("mri_direct", best_m[0], t_m)):
            w = p14.section_px_to_world(match, hx, hy, h.shape, PX)
            # y of the section at its tissue centre; slice number read off the fitted SbR geometry
            y_est = float(w[:, 1].mean())
            row[f"{name}_y_signed_mm"] = y_est - truth[:, 1].mean()
            row[f"{name}_y_err_mm"] = abs(y_est - truth[:, 1].mean())
            row[f"{name}_slice_err"] = abs(int(round((y_est - y0) / dy)) - int(round((truth[:, 1].mean() - y0) / dy)))
            row[f"{name}_err3d_mm"] = float(np.linalg.norm(w - truth, axis=1).mean())
            row[f"{name}_score"] = match.score
            row[f"{name}_flipped"] = match.flipped
            row[f"{name}_s"] = secs
        profs[f"s{sid:03d}"] = profile(h, best_hm[0], vm)
        rows.append(row)
        pd.DataFrame(rows).to_csv(out, index=False)
        np.savez(prof_f, offsets=OFFSETS, **{k: v for k, v in profs.items() if k != "offsets"})
        print(f"slice {sid:03d} | slice error: histology {row['histology_slice_err']:3d}  hist->MRI {row['hist_to_mri_slice_err']:3d}  "
              f"MRI direct {row['mri_direct_slice_err']:3d} | 3D err {row['histology_err3d_mm']:5.2f} {row['hist_to_mri_err3d_mm']:5.2f} "
              f"{row['mri_direct_err3d_mm']:5.2f} mm | {time.time() - t0:.1f}s", flush=True)
    faulthandler.cancel_dump_traceback_later()
    summarise(pd.DataFrame(rows))


SBR_OFFSET_MM = 1.0      # SbR pairs use the released MRI, 1.0 mm off in y (scripts/02); all methods find +1.0 mm


def summarise(df):
    for name in ("histology", "hist_to_mri", "mri_direct"):
        s = df[f"{name}_y_signed_mm"] - SBR_OFFSET_MM
        e = np.abs(np.round(s / 0.4))
        print(f"{name:12s} offset-corrected: median |y| {s.abs().median():.2f} mm  exact slice {np.mean(e == 0):.0%}  "
              f"within 1 slice {np.mean(e <= 1):.0%}  within 1 mm {np.mean(s.abs() < 1):.0%}  within 2 mm {np.mean(s.abs() < 2):.0%}")
    for name in ("histology", "hist_to_mri", "mri_direct"):
        e, y = df[f"{name}_slice_err"], df[f"{name}_y_err_mm"]
        print(f"{name:12s} n={len(df)}  exact slice {np.mean(e == 0):.0%}  within 1 slice {np.mean(e <= 1):.0%}  "
              f"within 5 slices (2 mm) {np.mean(e <= 5):.0%}  median y err {y.median():.2f} mm  "
              f"median 3D err {df[f'{name}_err3d_mm'].median():.2f} mm  median signed y {df[f'{name}_y_signed_mm'].median():+.2f} mm")


if __name__ == "__main__":
    main()
