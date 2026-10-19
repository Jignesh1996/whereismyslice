#!/usr/bin/env python3
"""Step-by-step picture of small-patch localisation for one real piece of tissue (raw SbR BigBrain section):
(1) the piece and where it was cut, (2) its parent section placed in the brain -> the slice (shown on a sagittal
MRI view), (3) the JEPA match map inside that slice -> the spot, i.e. the approximate 3D position.
Same route as 'section_tta' in scripts/14. Output: figures/patch_pipeline.png, results/patch_pipeline_example.npz (for the 3D view, scripts/25)
"""
import argparse
import sys
from importlib import import_module
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from wms import volumes
from wms.classical import localise
from wms.jepa import TARGET_PX_MM, TARGET_SIZE, TOKEN_MM, PatchJEPA, footprint_tokens, window_maps
from wms.patches import cut_patch, sbr_geometry, sbr_mri_world, truth_mri_px
from wms.planes import rotation, sample

sbr = import_module("08_sbr_inplane")
p13 = import_module("13_patch_localisation")
p14 = import_module("14_hierarchical_patch")
ROOT = Path(__file__).resolve().parents[1]
NAVY, TEAL, ORANGE, RED = "#002448", "#1F6E7A", "#D4A54D", "#E8312A"
SBR_OFFSET_MM = 1.0


def to_canvas(param, w, flipped):
    """World mm -> (col, row) on a TARGET_SIZE canvas at TARGET_PX_MM rendered with param (7,)."""
    R = rotation(torch.tensor(param[None, 3:6], dtype=torch.float32))[0].numpy()
    off = np.asarray(w) - param[:3]
    col = off @ R[:, 0] / TARGET_PX_MM + (TARGET_SIZE - 1) / 2
    row = (TARGET_SIZE - 1) / 2 - off @ R[:, 2] / TARGET_PX_MM
    return (TARGET_SIZE - 1 - col if flipped else col), row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=int, default=142)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="figures/patch_pipeline.png")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    geom = sbr_geometry()
    ck = torch.load(ROOT / "models/patch_jepa.pt", map_location=dev)
    jepa = PatchJEPA(ck["arch"], pretrained=False).to(dev).eval()
    jepa.load_state_dict(ck["state_dict"])
    vol = volumes.load("bigbrain_mri", dev)

    _, h, hm, m, mm = sbr.load_pieces("BigBrain", a.slice)[0]
    T_ref, _ = sbr.reference_affine("BigBrain", a.slice, "whole", h, hm, "cpu")
    p = cut_patch(h, hm, T_ref, np.random.default_rng([a.seed, a.slice]))
    tx, ty = truth_mri_px(p, T_ref)
    truth = sbr_mri_world(a.slice, tx, ty, m.shape, geom)
    size_mm = p.side_px * p.px_mm
    k = footprint_tokens(size_mm)

    # step 2: place the whole parent section in 3D
    sec = localise(h, (p14.HISTO_PX, p14.HISTO_PX), vol, metric="mi",
                   coarse_kw=dict(angles=p14.ANGLES, rolls=(-10, 0, 10), try_flip=True))[0][0]
    y_true = geom[2] + geom[3] * a.slice + SBR_OFFSET_MM
    # step 3: JEPA in that plane
    canvas, to_world = p14.section_canvas(vol, sec, dev)
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        maps = window_maps(jepa.target(canvas[None]).float())
    z = p14.embed(jepa, p13.patch_input(p).to(dev)[None], tta=True)
    sm = torch.einsum("d,dhw->hw", z, maps[k][0]).float().cpu().numpy()
    r, c = np.unravel_index(sm.argmax(), sm.shape)
    w_est = to_world(sm, r, c)
    err = float(np.linalg.norm(w_est - truth))
    pose = sec.pose
    param = np.array([pose.x, pose.y, pose.z, pose.pitch, pose.yaw, pose.roll, 0.0], np.float32)
    ec, er = to_canvas(param, w_est, sec.flipped)
    tc, tr = to_canvas(param, truth, sec.flipped)
    np.savez(ROOT / "results/patch_pipeline_example.npz", sec_pose=param, sec_log_s=pose.log_s, flipped=sec.flipped,
             w_est=w_est, truth=truth, size_mm=size_mm, y_true=y_true, err=err, patch=p.img, canvas=canvas.float().cpu().numpy(),
             sid=a.slice)
    print(f"slice {a.slice} seed {a.seed}: patch {size_mm:.0f} mm, slice y error {pose.y - y_true:+.2f} mm, 3D error {err:.1f} mm")

    fig, axes = plt.subplots(1, 4, figsize=(24, 6.6), facecolor="white",
                             gridspec_kw=dict(width_ratios=[1.25, 0.8, 1.35, 1.25], wspace=0.16))
    # 1: section with the cut
    ax = axes[0]
    ax.imshow(1 - h, cmap="gray", vmin=0, vmax=1)
    x0, y0 = [float(v[0]) for v in p.to_histo(np.array([0.0]), np.array([0.0]))]
    ax.add_patch(Rectangle((x0, y0), p.side_px, p.side_px, fill=False, ec=ORANGE, lw=4))
    ax.set_title("1 · a piece cut from\na raw section", fontsize=22, color=NAVY)
    ax.axis("off")
    ax = axes[1]
    ax.imshow(1 - p.img, cmap="gray", vmin=0, vmax=1)
    for sp in ax.spines.values():
        sp.set_edgecolor(ORANGE)
        sp.set_linewidth(5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"the piece\n({size_mm:.0f} mm)", fontsize=22, color=NAVY)
    # 2: which slice, on a sagittal MRI view
    ax = axes[2]
    lo, hi = vol.bounds_mm
    sag = np.array([pose.x, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2, 0, 90, 0, 0], np.float32)
    H, W, px = 300, 380, 0.5
    with torch.no_grad():
        img = sample(vol, torch.tensor(sag[None], device=dev), (H, W), px)[0].cpu().numpy()
    R = rotation(torch.tensor(sag[None, 3:6], dtype=torch.float32))[0].numpy()
    col_of_y = lambda y: (np.array([0, y - sag[1], 0]) @ R[:, 0]) / px + (W - 1) / 2
    ax.imshow(img, cmap="gray")
    ax.axvline(col_of_y(pose.y), color=TEAL, lw=5, label=f"slice we found (y {pose.y:+.1f} mm)")
    ax.axvline(col_of_y(y_true), color=ORANGE, lw=3, ls="--", label=f"true slice ({abs(pose.y - y_true):.1f} mm away)")
    ax.legend(fontsize=14, loc="lower center", framealpha=0.9)
    ax.set_title("2 · its section placed\nin the brain → the slice", fontsize=22, color=NAVY)
    ax.axis("off")
    # 3: match map in that slice
    ax = axes[3]
    cv = canvas.float().cpu().numpy()
    ax.imshow(cv, cmap="gray")
    heat = np.kron(sm, np.ones((int(TOKEN_MM / TARGET_PX_MM),) * 2))
    heat = (heat - heat.min()) / max(heat.max() - heat.min(), 1e-9)
    ax.imshow(np.ma.masked_less(heat, 0.6), cmap="viridis", alpha=0.45, extent=(-0.5, heat.shape[1] - 0.5, heat.shape[0] - 0.5, -0.5))
    s = size_mm / TARGET_PX_MM
    ax.add_patch(Rectangle((tc - s / 2, tr - s / 2), s, s, fill=False, ec=ORANGE, lw=3, ls="--", label="true spot"))
    ax.add_patch(Rectangle((ec - s / 2, er - s / 2), s, s, fill=False, ec=RED, lw=4, label="our spot"))
    ax.legend(fontsize=14, loc="lower center", framealpha=0.9)
    ax.set_title(f"3 · best match in that slice\n→ 3D spot, {err:.0f} mm from truth", fontsize=22, color=NAVY)
    ax.axis("off")
    fig.savefig(ROOT / a.out, dpi=90, bbox_inches="tight", facecolor="white")
    print(ROOT / a.out)


if __name__ == "__main__":
    main()
