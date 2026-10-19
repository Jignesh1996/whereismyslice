#!/usr/bin/env python3
"""In-plane localisation of raw histology sections inside their matching MRI slice (SbR datasets,
Casamitjana et al. 2021, github.com/acasamitjana/SynthByReg). slice_N in histo/ and mri/ are a pair.

Ground truth: BigBrain histo/affine/slice_N.txt, NiftyReg affines mapping MRI pixel (x=col, y=row) ->
histology pixel (verified: warping with them reproduces histo/images_affine at NCC 0.999 on all 344).
Some of those reference registrations failed, so each slice also gets the reference's own mask Dice and
results are reported for all slices and for the trusted subset (reference Dice >= 0.9).

Our estimate uses only the image pair and the dataset's tissue masks (no initial transform).
Error = distance between where the reference and our estimate put each brain pixel, in MRI pixels
(x0.4 mm; see --mri-px-mm). Output: results/sbr_<dataset>_inplane.csv
"""
import argparse
import faulthandler
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from wms.inplane import localise, mapping_error, mask_dice, warp

ROOT = Path(__file__).resolve().parents[1]
SBR = ROOT / "data/sbr/repo/data"


def load_pieces(ds, sid):
    """[(piece, histo, histo mask, mri, mri mask)]. Allen slices cut into cerebrum + cerebellum come as separate
    files (slice_N.cerebrum.png, slice_N.cerebellum.png), each with its own reference affine; others are whole."""
    d = SBR / ds
    names = [n for n in ("cerebrum", "cerebellum") if (d / "histo/images" / f"slice_{sid:03d}.{n}.png").exists()]
    out = []
    for piece in names or ["whole"]:
        f = f"slice_{sid:03d}.png" if piece == "whole" else f"slice_{sid:03d}.{piece}.png"
        h = np.asarray(Image.open(d / "histo/images" / f).convert("L"), np.float32) / 255
        m = np.asarray(Image.open(d / "mri/images" / f).convert("L"), np.float32) / 255
        if ds == "BigBrain":
            h = 1 - h                                                       # white-background histology -> tissue bright
        hm = np.asarray(Image.open(d / "histo/labels" / f).convert("L")) > 0
        mm = np.asarray(Image.open(d / "mri/labels" / f).convert("L")) > 0
        out.append((piece, h * hm, hm, m * mm, mm))
    return out


def _piece_file(sid, piece):
    return f"slice_{sid:03d}.png" if piece == "whole" else f"slice_{sid:03d}.{piece}.png"


def reference_affine(ds, sid, piece, h, hm, dev):
    """BigBrain: the NiftyReg file. Allen (no files): recover the affine that produced histo/images_affine for this
    piece by registering the raw histology piece to it (same image resampled, so exact when it works); returns
    (T, NCC of the reproduction) and caches it under results/sbr_reference/<ds>/."""
    f = SBR / ds / "histo/affine" / f"slice_{sid:03d}.txt"
    if f.exists():
        return np.loadtxt(f)[[0, 1, 3]][:, [0, 1, 3]], np.nan
    cache = ROOT / "results/sbr_reference" / ds / f"slice_{sid:03d}_{piece}.txt"
    if cache.exists():
        v = np.loadtxt(cache)
        return v[:9].reshape(3, 3), float(v[9])
    f = _piece_file(sid, piece)
    ha = np.asarray(Image.open(SBR / ds / "histo/images_affine" / f).convert("L"), np.float32) / 255
    ham = (np.asarray(Image.open(SBR / ds / "histo/labels_affine" / f).convert("L")) > 0) & (ha > 0.02)
    ha = ha * ham
    T = localise(h, hm, ha, ham, device=dev).T
    w = warp(torch.tensor(h)[None], torch.tensor(T[None], dtype=torch.float32), ha.shape)[0].numpy()
    m = ham & (w > 0.02)
    r = float(np.corrcoef(w[m], ha[m])[0, 1])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(cache, np.r_[T.ravel(), r])
    return T, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BigBrain")
    ap.add_argument("--ids", type=int, nargs="*")
    ap.add_argument("--mri-px-mm", type=float, default=0.4)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gt_dir = SBR / a.dataset / "histo/affine"
    if a.ids:
        ids = a.ids
    elif gt_dir.exists():
        ids = sorted(int(p.stem[6:9]) for p in gt_dir.glob("slice_*.txt"))
    else:
        ids = [int(line.split(",")[0]) for line in (SBR / a.dataset / "slice_id.txt").read_text().split()[1:]]
    out = ROOT / f"results/sbr_{a.dataset}_inplane.csv"
    if a.ids:
        out = out.with_name(out.stem + "_subset.csv")
    done = pd.read_csv(out) if out.exists() and not a.ids else pd.DataFrame()
    rows = done.to_dict("records")
    done_keys = {(int(r["slice"]), r["piece"]) for r in rows}
    if done_keys:
        print(f"resuming: {len(done_keys)} items already in {out.name}", flush=True)
    for sid in ids:
        for piece, h, hm, m, mm in load_pieces(a.dataset, sid):
            if (sid, piece) in done_keys:
                continue
            # a stuck CUDA call never returns (seen when the driver runs out of memory): dump the stack and exit
            # so a wrapper can restart and resume from the CSV
            faulthandler.dump_traceback_later(300, exit=True, file=sys.stderr)
            T_gt, ref_ncc = reference_affine(a.dataset, sid, piece, h, hm, dev)
            if ref_ncc < 0.95:                                              # Allen: recovery failed, no usable reference
                print(f"slice {sid:03d} {piece}: reference affine not recoverable (NCC {ref_ncc:.3f}), skipped", flush=True)
                continue
            t0 = time.time()
            res = localise(h, hm, m, mm, device=dev)
            secs = time.time() - t0
            err = mapping_error(res.T, T_gt, mm) * a.mri_px_mm
            err_c = mapping_error(res.coarse_T, T_gt, mm) * a.mri_px_mm
            row = dict(slice=sid, piece=piece, reference_recovery_ncc=ref_ncc, dice_reference=mask_dice(T_gt, hm, mm),
                       dice_ours=mask_dice(res.T, hm, mm), err_mean_px=err.mean() / a.mri_px_mm, err_mean_mm=err.mean(),
                       err_median_mm=np.median(err), err_max_mm=err.max(), coarse_err_mean_mm=err_c.mean(),
                       score=res.score, seconds=secs, T=" ".join(f"{v:.5f}" for v in res.T[:2].ravel()))
            rows.append(row)
            pd.DataFrame(rows).to_csv(out, index=False)                    # written every item, so a crash loses nothing
            print(f"slice {sid:03d} {piece:10s} Dice reference {row['dice_reference']:.3f} ours {row['dice_ours']:.3f}  "
                  f"distance to reference mean {row['err_mean_mm']:.2f} mm (max {row['err_max_mm']:.2f}; coarse {row['coarse_err_mean_mm']:.2f})  {secs:.1f}s",
                  flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no slices with a usable reference")
        return
    for name, g in (("all", df), ("trusted reference (Dice >= 0.9)", df[df.dice_reference >= 0.9])):
        print(f"\n{name}: n={len(g)}  distance to reference: median of per-slice means {g.err_mean_mm.median():.2f} mm, "
              f"<1 mm {np.mean(g.err_mean_mm < 1):.0%}, <2 mm {np.mean(g.err_mean_mm < 2):.0%}; "
              f"Dice ours {g.dice_ours.median():.3f} vs reference {g.dice_reference.median():.3f}; "
              f"ours better Dice on {np.mean(g.dice_ours > g.dice_reference):.0%}")


if __name__ == "__main__":
    main()
