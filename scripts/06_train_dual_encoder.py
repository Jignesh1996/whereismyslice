#!/usr/bin/env python3
"""Train the histology <-> MRI dual encoder on synthetic pairs resliced on the fly (GPU) from BigBrain
histology and BigBrain's MRI at random oblique poses. No manual labels.

A slab of the brain (default y in [0, 20] mm, roughly sections 3500-4500) is never used as a training pose
centre so retrieval there tests anatomy the encoder has not been trained on.
Output: models/dual_encoder.pt, results/train_log.csv
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
from wms.dual_encoder import DualEncoder, plane_distance, soft_infonce
from wms.embed import canonical_views
from wms.planes import random_poses, sample

ROOT = Path(__file__).resolve().parents[1]
RENDER_PX, FOV = 0.5, 160.0


def draw_poses(n, bounds, rng, holdout, inside=False):
    out = []
    while len(out) < n:
        for p in random_poses(4 * n, bounds, max_tilt=(20, 20, 10), margin_mm=(35, 6, 30), rng=rng):
            if (holdout[0] <= p.y <= holdout[1]) == inside:
                out.append(p.as_array())
    return torch.tensor(np.stack(out[:n]), dtype=torch.float32)


def render_pair(hist, mri, params, n_px):
    with torch.no_grad():
        h = sample(hist, params, (n_px, n_px), RENDER_PX)
        m = sample(mri, params, (n_px, n_px), RENDER_PX)
        vh, _ = canonical_views(h, RENDER_PX, FOV)
        vm, _ = canonical_views(m, RENDER_PX, FOV)
    keep = (vh > 0.08).float().mean((1, 2)) > 0.05                   # drop planes that barely touch the brain
    return vh, vm, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--holdout", type=float, nargs=2, default=(0.0, 20.0))
    ap.add_argument("--out", default="models/dual_encoder.pt")
    a = ap.parse_args()
    # hang diagnostics without ptrace: kill -USR1 <pid> dumps the stack, and any step slower than 120 s dumps it to the log
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    dev = "cuda"
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    hist, mri = volumes.load("bigbrain_400", dev), volumes.load("bigbrain_mri", dev)
    bounds = hist.bounds_mm
    n_px = int(np.max(bounds[1] - bounds[0]) * 1.15 / RENDER_PX)
    model = DualEncoder().to(dev)
    backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": backbone, "lr": a.lr * 0.1}, {"params": rest, "lr": a.lr}], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr * 0.1, a.lr], total_steps=a.steps, pct_start=0.05)
    val_params = draw_poses(256, bounds, np.random.default_rng(123), a.holdout, inside=True).to(dev)
    val_h, val_m, val_keep = render_pair(hist, mri, val_params, n_px)
    val_h, val_m, val_params = val_h[val_keep], val_m[val_keep], val_params[val_keep]

    def validate():
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            zh = torch.cat([model.encode(val_h[i:i + 64], "histology") for i in range(0, len(val_h), 64)]).float()
            zm = torch.cat([model.encode(val_m[i:i + 64], "mri") for i in range(0, len(val_m), 64)]).float()
        top1 = (zh @ zm.T).argmax(1)
        d = plane_distance(val_params)[torch.arange(len(top1)), top1]
        model.train()
        return float((d < 2).float().mean()), float(d.median())

    log, t0 = [], time.time()
    acc, med = validate()
    print(f"step 0  held-out slab retrieval within 2 mm {acc:.2f} (median {med:.1f})", flush=True)
    model.train()
    out = ROOT / a.out
    out.parent.mkdir(exist_ok=True)

    def save(step):
        torch.save({"state_dict": model.state_dict(), "arch": model.arch, "holdout_y": a.holdout, "fov_mm": FOV,
                    "step": step}, out)
        pd.DataFrame(log).to_csv(ROOT / "results/train_log.csv", index=False)

    for step in range(1, a.steps + 1):
        faulthandler.dump_traceback_later(120, repeat=True, file=sys.stderr)
        params = draw_poses(a.batch, bounds, rng, a.holdout).to(dev)
        vh, vm, keep = render_pair(hist, mri, params, n_px)
        vh, vm, params = augment_torch(vh[keep]), vm[keep], params[keep]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zh, zm = model.encode(vh, "histology"), model.encode(vm, "mri")
            pose_pred = model.pose_head(zh.float())
        target = torch.stack([params[:, 1] / 75, params[:, 3] / 20, params[:, 4] / 20], 1)
        loss_nce = soft_infonce(zh.float(), zm.float(), params, model.logit_scale)
        loss_pose = F.smooth_l1_loss(pose_pred, target)
        loss = loss_nce + 0.5 * loss_pose
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == a.steps:
            row = dict(step=step, loss=loss.item(), nce=loss_nce.item(), pose=loss_pose.item(),
                       temp=(1 / model.logit_scale.exp()).item(), minutes=(time.time() - t0) / 60)
            if step % 500 == 0 or step == a.steps:
                row["val_within_2mm"], row["val_median_mm"] = validate()
            log.append(row)
            print("  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()),
                  f" VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
            if step % 500 == 0:
                save(step)
    faulthandler.cancel_dump_traceback_later()
    save(a.steps)
    print("saved", out)


if __name__ == "__main__":
    main()
