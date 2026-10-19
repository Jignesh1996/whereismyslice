#!/usr/bin/env python3
"""Train the JEPA-style patch localiser on synthetic pairs rendered on the fly from BigBrain histology + MRI.

Each sample: a random oblique plane; the MRI plane (0.8 mm, 224 px, centred on the brain) is the target; a 20-40 mm
histology patch from the same plane (0.4 mm) at a random tissue location, randomly rotated (360 deg), scaled
(+-10 %), with stain/tear/chunk/blur/noise artefacts, is the context. Pose centres in y in [0, 20] mm are held out.
Output: models/patch_jepa.pt, results/patch_jepa_log.csv
"""
import argparse
import faulthandler
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from wms import volumes
from wms.augment import augment_torch
from wms.jepa import PATCH_SIZE, TARGET_PX_MM, TARGET_SIZE, TOKEN_MM, PatchJEPA, footprint_tokens, losses, window_maps
from wms.planes import as_params, random_poses, rotation, sample

ROOT = Path(__file__).resolve().parents[1]
HIST_PX = 0.4


def poses(n, bounds, rng, holdout, inside):
    out = []
    while len(out) < n:
        for p in random_poses(4 * n, bounds, max_tilt=(20, 20, 0), margin_mm=(35, 15, 30), rng=rng):
            if (holdout[0] <= p.y <= holdout[1]) == inside:
                out.append(p.as_array())
    P = np.stack(out[:n])
    return P


def make_batch(hist, mri, P, rng, centre_xz, dev):
    """Render target MRI planes and cut context patches. Returns patches, planes, k, token positions, keep mask."""
    P = P.copy()
    # put every plane's centre at the brain centre projected into the plane, so the 179 mm field covers the brain
    R = rotation(torch.tensor(P[:, 3:6], dtype=torch.float32)).numpy()
    for i in range(len(P)):
        c = np.array([centre_xz[0], P[i, 1], centre_xz[1]])
        n = R[i][:, 1]
        P[i, :3] = c - np.dot(c - P[i, :3], n) * n
    params = torch.tensor(P, dtype=torch.float32, device=dev)
    n_h = int(TARGET_SIZE * TARGET_PX_MM / HIST_PX)
    with torch.no_grad():
        planes = sample(mri, params, (TARGET_SIZE, TARGET_SIZE), TARGET_PX_MM)
        hp = sample(hist, params, (n_h, n_h), HIST_PX)
    B = len(P)
    size_mm = rng.uniform(20, 40, B)
    ang = rng.uniform(0, 2 * np.pi, B)
    scale = rng.uniform(0.9, 1.1, B)
    tissue = (F.avg_pool2d((hp > 0.08).float()[:, None], 9, 1, 4)[:, 0] > 0.7).cpu().numpy()
    centres, keep = np.zeros((B, 2)), np.zeros(B, bool)
    for i in range(B):
        r, c = np.nonzero(tissue[i])
        margin = size_mm[i] / HIST_PX
        ok = (r > margin) & (c > margin) & (r < n_h - margin) & (c < n_h - margin)
        if ok.sum() > 0:
            j = rng.choice(np.nonzero(ok)[0])
            centres[i], keep[i] = (c[j], r[j]), True
    # one affine grid per sample: 112 px output covering size_mm, rotated and scaled, around the chosen centre
    theta = torch.zeros(B, 2, 3, device=dev)
    half = size_mm / 2 / HIST_PX * scale                              # half side in histology px
    for i in range(B):
        ca, sa = np.cos(ang[i]), np.sin(ang[i])
        sx = half[i] / ((n_h - 1) / 2)
        theta[i, 0] = torch.tensor([ca * sx, -sa * sx, 2 * centres[i, 0] / (n_h - 1) - 1])
        theta[i, 1] = torch.tensor([sa * sx, ca * sx, 2 * centres[i, 1] / (n_h - 1) - 1])
    grid = F.affine_grid(theta, (B, 1, PATCH_SIZE, PATCH_SIZE), align_corners=True)
    patches = F.grid_sample(hp[:, None], grid, align_corners=True)[:, 0]
    patches = augment_torch(patches)
    pos = torch.tensor(centres * HIST_PX / TOKEN_MM - 0.5, dtype=torch.float32, device=dev)   # token (col, row)
    k = torch.tensor([footprint_tokens(s) for s in size_mm], device=dev)
    keep_t = torch.tensor(keep, device=dev)
    return patches[keep_t], planes[keep_t], k[keep_t], pos[keep_t]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--holdout", type=float, nargs=2, default=(0.0, 20.0))
    ap.add_argument("--out", default="models/patch_jepa.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-val", type=int, default=128)
    a = ap.parse_args()
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    dev = a.device
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    hist, mri = volumes.load("bigbrain_400", dev), volumes.load("bigbrain_mri", dev)
    lo, hi = hist.bounds_mm
    centre_xz = ((lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2)
    model = PatchJEPA().to(dev)
    bb = [p for p in model.backbone.parameters() if p.requires_grad]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.lr * 0.1}, {"params": rest, "lr": a.lr}], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr * 0.1, a.lr], total_steps=a.steps, pct_start=0.05)
    vrng = np.random.default_rng(123)
    val = make_batch(hist, mri, poses(a.n_val, (lo, hi), vrng, a.holdout, True), vrng, centre_xz, dev)

    def validate():
        model.eval()
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            z = model.context(val[0]).float()
            maps = {kk: v.float() for kk, v in window_maps(model.target(val[1]).float()).items()}
            _, _, err = losses(model, z, maps, val[2], val[3])
        model.train()
        return float(err.median()), float((err < 10).float().mean())

    out = ROOT / a.out
    out.parent.mkdir(exist_ok=True)
    log, t0 = [], time.time()
    med, acc = validate()
    print(f"step 0  held-out: median error {med:.1f} mm, within 10 mm {acc:.2f}", flush=True)
    for step in range(1, a.steps + 1):
        faulthandler.dump_traceback_later(120, exit=True, file=sys.stderr)
        patches, planes, k, pos = make_batch(hist, mri, poses(a.batch, (lo, hi), rng, a.holdout, False), rng, centre_xz, dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            z = model.context(patches).float()
            maps = {kk: v.float() for kk, v in window_maps(model.target(planes).float()).items()}
        loss_loc, loss_jepa, err = losses(model, z, maps, k, pos)
        loss = loss_loc + loss_jepa
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == a.steps or a.steps < 10:
            row = dict(step=step, loss_loc=loss_loc.item(), loss_jepa=loss_jepa.item(), train_err_mm=err.median().item(),
                       minutes=(time.time() - t0) / 60)
            if step % 500 == 0 or step == a.steps:
                row["val_median_mm"], row["val_within_10mm"] = validate()
                torch.save({"state_dict": model.state_dict(), "arch": model.arch, "holdout_y": a.holdout, "step": step}, out)
                pd.DataFrame(log + [row]).to_csv(ROOT / "results/patch_jepa_log.csv", index=False)
            log.append(row)
            print("  ".join(f"{kk} {v:.3f}" if isinstance(v, float) else f"{kk} {v}" for kk, v in row.items()),
                  f" VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB" if dev == "cuda" else "", flush=True)
    faulthandler.cancel_dump_traceback_later()
    print("saved", out)


if __name__ == "__main__":
    main()
