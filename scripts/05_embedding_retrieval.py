#!/usr/bin/env python3
"""Embedding baseline: frozen DINOv2 retrieval of sections from a slice bank, alone and as the proposal step
for classical refinement (hybrid).

Bank: planes of the reference volume over y (1 mm) x pitch x yaw (-15..15 deg, 5 deg steps).
Queries: real BigBrain section thumbnails (truth: the 20 um cutting line from scripts/04) and synthetic
oblique, artefacted histology planes (truth known). Output: results/embedding_<bank>.csv + printed summary.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wms import sections, volumes
from wms.augment import augment
from wms.classical import refine
from wms.dual_encoder import DualEncoder
from wms.embed import Embedder
from wms.planes import Pose, pose_error, random_poses, sample
from wms.retrieval import Bank, build_bank, query

ROOT = Path(__file__).resolve().parents[1]
Y0, DY = -70.01, 0.020                     # reference cutting line fitted in scripts/04_real_sections.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="bigbrain_400", help="reference volume: bigbrain_400 or bigbrain_mri")
    ap.add_argument("--arch", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--n-synth", type=int, default=30)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model", default=None, help="trained dual encoder (models/dual_encoder.pt); default frozen DINOv2")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vol = volumes.load(a.bank, dev)
    holdout = (np.inf, -np.inf)
    if a.model:
        ck = torch.load(ROOT / a.model, map_location=dev)
        de = DualEncoder(ck["arch"], pretrained=False)
        de.load_state_dict(ck["state_dict"])
        holdout = tuple(ck["holdout_y"])
        emb = Embedder(device=dev, model=de.backbone, head=de.heads["histology"])            # queries
        bank_emb = Embedder(device=dev, model=de.backbone, head=de.heads[vol.modality])      # reference planes
        tag = Path(a.model).stem
    else:
        emb = bank_emb = Embedder(a.arch, dev)
        tag = a.arch.split(".")[0]
    cache = ROOT / "results" / "banks" / f"{a.bank}__{tag}.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if cache.exists():
        bank = Bank.load(cache)
    else:
        bank = build_bank(vol, bank_emb)
        bank.save(cache)
    print(f"bank: {len(bank.poses)} planes from {a.bank} ({time.time() - t0:.0f}s)")

    queries = []                                                    # (kind, id, image, px, true Pose)
    for n in range(300, 7101, a.every):
        img, px = sections.load(n)
        queries.append(("real", n, img, px, Pose(0, Y0 + DY * n, 0)))
    rng = np.random.default_rng(0)                                  # same draws as scripts/03
    src = volumes.load("bigbrain_400", dev) if a.bank != "bigbrain_400" else vol
    lo, hi = src.bounds_mm
    for i, p in enumerate(random_poses(a.n_synth, (lo, hi), max_tilt=(10, 10, 10), margin_mm=(50, 15, 40), rng=rng)):
        with torch.no_grad():
            img = sample(src, [p], (300, 350), 0.4)[0].cpu().numpy()
        queries.append(("synthetic", i, augment(img, rng), (0.4, 0.4), p))

    metric = "ncc" if vol.modality == "histology" else "mi"
    rows = []
    for kind, qid, img, px, true in queries:
        t1 = time.time()
        hits = query(bank, emb, img, px, k=a.k, rolls=(-10, -5, 0, 5, 10), flips=(False, True) if kind == "real" else (False,))
        t_ret = time.time() - t1
        errs = [pose_error(true, h.pose) for h in hits]
        best = refine(img, px, vol, hits, metric=metric)[0]
        off_h, tilt_h = pose_error(true, best.pose)
        rows.append(dict(kind=kind, id=qid, held_out=bool(holdout[0] <= true.y <= holdout[1]), top1_offset_mm=errs[0][0], top1_tilt_deg=errs[0][1],
                         topk_best_offset_mm=min(e[0] for e in errs), hybrid_offset_mm=off_h, hybrid_tilt_deg=tilt_h,
                         sim=hits[0].score, retrieval_s=t_ret, total_s=time.time() - t1))
        print(f"{kind:9s} {qid:5d}  top1 {errs[0][0]:6.2f} mm {errs[0][1]:4.1f} deg | best of top{a.k} {rows[-1]['topk_best_offset_mm']:6.2f} mm"
              f" | hybrid {off_h:6.2f} mm {tilt_h:4.1f} deg  ({t_ret:.2f}s + refine)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / f"results/embedding_{a.bank}__{tag}.csv", index=False)
    summ = df.groupby(["kind", "held_out"]).agg(n=("id", "size"),
        top1_median_mm=("top1_offset_mm", "median"), top1_within_3mm=("top1_offset_mm", lambda s: (s < 3).mean()),
        topk_within_3mm=("topk_best_offset_mm", lambda s: (s < 3).mean()), top1_tilt=("top1_tilt_deg", "median"),
        hybrid_median_mm=("hybrid_offset_mm", "median"), hybrid_within_2mm=("hybrid_offset_mm", lambda s: (s < 2).mean()),
        retrieval_s=("retrieval_s", "mean"))
    print(f"\nbank {a.bank}, {tag}\n{summ.round(3).to_string()}")


if __name__ == "__main__":
    main()
