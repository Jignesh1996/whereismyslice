#!/usr/bin/env python3
"""Localise small REAL histology patches (raw SbR BigBrain sections, 20-40 mm) in MRI, in 2D and 3D.

2D (patch inside its own MRI slice): classical template search over the whole slice; frozen DINOv2 dense features;
JEPA; JEPA top-3 positions refined by the classical search in a small window.
3D (patch inside the whole MRI volume): JEPA over a bank of MRI planes (y every 1 mm x pitch/yaw in {-10, 0, 10}),
then the classical search on the top candidate planes.
Truth: reference affine (histology -> MRI pixel) and the SbR slice geometry (scripts/09). Conditions: clean
patches and damaged ones (rotation 360, 10 % affine, 5 % elastic, tears/chunks).
Output: results/patch_localisation.csv
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from wms import volumes
from wms.jepa import PATCH_SIZE, TARGET_PX_MM, TARGET_SIZE, TOKEN_MM, PatchJEPA, footprint_tokens, window_maps, _prep
from wms.patches import Patch, classical_patch_search, cut_patch, sbr_geometry, sbr_mri_world, truth_mri_px
from wms.planes import plane_points, as_params, sample

sbr = import_module("08_sbr_inplane")
ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {"clean": None, "damaged": dict(rotation=True, affine=0.10, elastic=0.05, cuts=True)}


def canvas_from_slice(m, mri_px):
    """SbR MRI slice (H, W) at mri_px -> 224 x 224 canvas at 0.8 mm centred on the slice centre."""
    H, W = m.shape
    half = TARGET_SIZE * TARGET_PX_MM / 2
    theta = torch.tensor([[[half / (W * mri_px / 2), 0, 0], [0, half / (H * mri_px / 2), 0]]], dtype=torch.float32)
    grid = F.affine_grid(theta, (1, 1, TARGET_SIZE, TARGET_SIZE), align_corners=False)
    return F.grid_sample(torch.tensor(m)[None, None], grid, align_corners=False)[0, 0]


def canvas_px_to_slice(cx, cy, shape, mri_px):
    H, W = shape
    mm_x = (cx + 0.5) * TOKEN_MM - TARGET_SIZE * TARGET_PX_MM / 2
    mm_y = (cy + 0.5) * TOKEN_MM - TARGET_SIZE * TARGET_PX_MM / 2
    return (W - 1) / 2 + mm_x / mri_px, (H - 1) / 2 + mm_y / mri_px


def patch_input(p):
    """The central side_px square of the (possibly distorted, larger) patch canvas, resized to the model input, so a
    damaged patch looks like training data: a full square of rotated tissue, not a rotated square with empty corners."""
    H, W = p.img.shape
    s = min(p.side_px or min(H, W), H, W)
    y0, x0 = (H - s) // 2, (W - s) // 2
    t = torch.tensor(p.img[y0:y0 + s, x0:x0 + s])[None, None]
    return F.interpolate(t, size=(PATCH_SIZE, PATCH_SIZE), mode="area" if s > PATCH_SIZE else "bilinear")[0, 0]


def subcell(sims, r, c):
    """Parabolic peak interpolation around integer argmax (r, c) of a 2D similarity map -> (col, row) floats."""
    g_h, g_w = sims.shape
    def off(a, b, d):
        den = a - 2 * b + d
        return 0.0 if abs(den) < 1e-9 else float(np.clip(0.5 * (a - d) / den, -0.5, 0.5))
    dc = off(sims[r, c - 1], sims[r, c], sims[r, c + 1]) if 0 < c < g_w - 1 else 0.0
    dr = off(sims[r - 1, c], sims[r, c], sims[r + 1, c]) if 0 < r < g_h - 1 else 0.0
    return c + dc, r + dr


class Frozen:
    """Frozen DINOv2 baseline in the same interface: z = mean patch token, map = raw tokens."""
    def __init__(self, model):
        self.bb = model.backbone

    def context(self, x):
        tok = self.bb.forward_features(_prep(x))[:, self.bb.num_prefix_tokens:]
        return F.normalize(tok.mean(1), dim=1)

    def target(self, x):
        tok = self.bb.forward_features(_prep(x))[:, self.bb.num_prefix_tokens:]
        g = int(round(tok.shape[1] ** 0.5))
        return tok.transpose(1, 2).reshape(tok.shape[0], -1, g, g)


def top_positions(z, wmap, n=3, min_sep=2):
    sims = torch.einsum("d,dhw->hw", z, wmap)
    order = torch.argsort(sims.flatten(), descending=True).tolist()
    g = sims.shape[1]
    out = []
    for i in order:
        r, c = divmod(i, g)
        if all(abs(r - rr) + abs(c - cc) >= min_sep for _, cc, rr in out):
            out.append((float(sims.flatten()[i]), c, r))
        if len(out) == n:
            break
    return out


def local_search(p, image, x, y, radius_px, dev):
    """Classical patch search in a window of image around (x, y); returns (score, x, y) in image pixels."""
    H, W = image.shape
    x0, x1 = int(max(0, x - radius_px)), int(min(W, x + radius_px))
    y0, y1 = int(max(0, y - radius_px)), int(min(H, y + radius_px))
    side = int(np.ceil(max(p.img.shape) * p.px_mm / 0.4007 * 1.1 * 1.45)) | 1
    if x1 - x0 <= side or y1 - y0 <= side:                                 # window must hold the rotated template
        pad = side
        x0, x1, y0, y1 = max(0, x0 - pad), min(W, x1 + pad), max(0, y0 - pad), min(H, y1 + pad)
    crop = image[y0:y1, x0:x1]
    if crop.shape[0] <= side or crop.shape[1] <= side:
        return (-1, x, y)
    s, cx, cy, _, _ = classical_patch_search(p, crop, device=dev, top_k=1)[0]
    return (s, cx + x0, cy + y0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-slices", type=int, default=40)
    ap.add_argument("--per-slice", type=int, default=3)
    ap.add_argument("--model", default="models/patch_jepa.pt")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="skip slices already in results/patch_localisation.csv")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    geom = sbr_geometry()
    mri_px = geom[4]
    ck = torch.load(ROOT / a.model, map_location=dev)
    jepa = PatchJEPA(ck["arch"], pretrained=False).to(dev).eval()
    jepa.load_state_dict(ck["state_dict"])
    frozen = Frozen(PatchJEPA(pretrained=True).to(dev).eval())

    # 3D bank: MRI planes through the brain centre, y every 1 mm, pitch/yaw in {-10, 0, 10}
    vol = volumes.load("bigbrain_mri", dev)
    lo, hi = vol.bounds_mm
    cx, cz = geom[0], geom[1]
    bank = np.array([(cx, y, cz, p, w, 0, 0) for y in np.arange(lo[1] + 3, hi[1] - 3, 1.0)
                     for p in (-10, 0, 10) for w in (-10, 0, 10)], np.float32)
    feats = []
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        for i in range(0, len(bank), 32):
            planes = sample(vol, torch.tensor(bank[i:i + 32], device=dev), (TARGET_SIZE, TARGET_SIZE), TARGET_PX_MM)
            feats.append(jepa.target(planes).float().half())
    bank_maps = torch.cat(feats)                                           # (M, D, 16, 16)
    print(f"3D bank: {len(bank)} planes, {bank_maps.numel() * 2 / 2**20:.0f} MB", flush=True)

    ref = pd.read_csv(ROOT / "results/sbr_BigBrain_inplane.csv")
    rng = np.random.default_rng(a.seed)
    ids = sorted(rng.choice(ref[ref.dice_reference >= 0.9].slice.values, a.n_slices, replace=False))
    out = ROOT / "results/patch_localisation.csv"
    rows = pd.read_csv(out).to_dict("records") if out.exists() and a.resume else []
    done = {int(r["slice"]) for r in rows}
    if done:
        print(f"resuming: {len(done)} slices already in {out.name}", flush=True)
    for sid in ids:
        if int(sid) in done:
            continue
        rng = np.random.default_rng([a.seed, int(sid)])                       # per-slice patches: identical on resume
        _, h, hm, m, mm = sbr.load_pieces("BigBrain", int(sid))[0]
        T_ref, _ = sbr.reference_affine("BigBrain", int(sid), "whole", h, hm, "cpu")
        canvas = canvas_from_slice(m, mri_px).to(dev)
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            maps_j = window_maps(jepa.target(canvas[None]).float())
            maps_f = window_maps(frozen.target(canvas[None]).float())
        for cond, dist in CONDITIONS.items():
            for j in range(a.per_slice):
                faulthandler.dump_traceback_later(300, exit=True, file=sys.stderr)
                p = cut_patch(h, hm, T_ref, rng, distortion=dist)
                if p is None:
                    continue
                tx, ty = truth_mri_px(p, T_ref)
                truth3d = sbr_mri_world(int(sid), tx, ty, m.shape, geom)
                size_mm = p.side_px * p.px_mm
                k = footprint_tokens(size_mm)
                x_in = patch_input(p).to(dev)[None]
                row = dict(slice=int(sid), condition=cond, size_mm=size_mm)
                t0 = time.time()
                s, x, y, _, _ = classical_patch_search(p, m, mri_px_mm=mri_px, device=dev, top_k=1)[0]
                row["classical_2d_mm"] = np.hypot(x - tx, y - ty) * mri_px
                row["classical_s"] = time.time() - t0
                with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                    zj, zf = jepa.context(x_in).float()[0], frozen.context(x_in).float()[0]
                for name, z, maps in (("frozen", zf, maps_f), ("jepa", zj, maps_j)):
                    tops = top_positions(z, maps[k][0])
                    x, y = canvas_px_to_slice(tops[0][1], tops[0][2], m.shape, mri_px)
                    row[f"{name}_2d_mm"] = np.hypot(x - tx, y - ty) * mri_px
                    if name == "jepa":
                        sm = torch.einsum("d,dhw->hw", z, maps[k][0]).cpu().numpy()
                        x, y = canvas_px_to_slice(*subcell(sm, tops[0][2], tops[0][1]), m.shape, mri_px)
                        row["jepa_subcell_2d_mm"] = np.hypot(x - tx, y - ty) * mri_px
                t0 = time.time()
                radius = (k * TOKEN_MM / 2 + 12) / mri_px
                cands = [local_search(p, m, *canvas_px_to_slice(c, r, m.shape, mri_px), radius, dev)
                         for _, c, r in top_positions(zj, maps_j[k][0])]
                s, x, y = max(cands)
                row["jepa_refined_2d_mm"] = np.hypot(x - tx, y - ty) * mri_px
                row["jepa_refined_s"] = time.time() - t0
                # 3D: JEPA over the plane bank, then classical search on the top planes
                t0 = time.time()
                with torch.no_grad():
                    wm = window_maps(bank_maps.float(), ks=(k,))[k] if k > 1 else F.normalize(bank_maps.float(), dim=1)
                    sims = torch.einsum("d,mdhw->mhw", zj, wm)
                flat = torch.argsort(sims.flatten(), descending=True)[:200].tolist()
                picks, seen = [], set()
                for f in flat:
                    mi, rem = divmod(f, sims.shape[1] * sims.shape[2])
                    r, c = divmod(rem, sims.shape[2])
                    if mi not in seen:
                        seen.add(mi)
                        picks.append((mi, c, r))
                    if len(picks) == 5:
                        break
                pts = plane_points(as_params(bank[[pk[0] for pk in picks]], dev), (TARGET_SIZE, TARGET_SIZE), TARGET_PX_MM)
                tok_px = TOKEN_MM / TARGET_PX_MM
                w0 = pts[0, int((picks[0][2] + 0.5) * tok_px), int((picks[0][1] + 0.5) * tok_px)].cpu().numpy()
                row["jepa_3d_mm"] = float(np.linalg.norm(w0 - truth3d))
                mi0, c0, r0 = picks[0]
                cs, rs = subcell(sims[mi0].float().cpu().numpy(), r0, c0)
                P0 = plane_points(as_params(bank[mi0:mi0 + 1], dev), (TARGET_SIZE, TARGET_SIZE), TARGET_PX_MM)[0]
                fy, fx = (rs + 0.5) * tok_px, (cs + 0.5) * tok_px                 # fractional canvas pixel
                y0_, x0_ = int(np.clip(np.floor(fy), 0, TARGET_SIZE - 2)), int(np.clip(np.floor(fx), 0, TARGET_SIZE - 2))
                ay, ax_ = fy - y0_, fx - x0_
                wsub = ((1 - ay) * ((1 - ax_) * P0[y0_, x0_] + ax_ * P0[y0_, x0_ + 1]) +
                        ay * ((1 - ax_) * P0[y0_ + 1, x0_] + ax_ * P0[y0_ + 1, x0_ + 1])).cpu().numpy()
                row["jepa_subcell_3d_mm"] = float(np.linalg.norm(wsub - truth3d))
                best = (-1, None)
                for (mi, c, r) in picks:
                    plane = sample(vol, torch.tensor(bank[mi:mi + 1], device=dev), (int(TARGET_SIZE * TARGET_PX_MM / mri_px),) * 2, mri_px)[0]
                    pl = plane.cpu().numpy()
                    cx_px, cy_px = (c + 0.5) * TOKEN_MM / mri_px, (r + 0.5) * TOKEN_MM / mri_px
                    s, x, y = local_search(p, pl, cx_px, cy_px, radius, dev)
                    if s > best[0]:
                        P3 = plane_points(as_params(bank[mi:mi + 1], dev), pl.shape, mri_px)[0]
                        best = (s, P3[int(np.clip(round(y), 0, pl.shape[0] - 1)), int(np.clip(round(x), 0, pl.shape[1] - 1))].cpu().numpy())
                row["jepa_refined_3d_mm"] = float(np.linalg.norm(best[1] - truth3d))
                row["jepa_3d_s"] = time.time() - t0
                rows.append(row)
                print(f"slice {sid:03d} {cond:7s} {size_mm:4.1f} mm | 2D classical {row['classical_2d_mm']:5.1f} frozen {row['frozen_2d_mm']:5.1f} "
                      f"JEPA {row['jepa_2d_mm']:5.1f} sub {row['jepa_subcell_2d_mm']:5.1f} JEPA+refine {row['jepa_refined_2d_mm']:5.1f} | 3D JEPA {row['jepa_3d_mm']:5.1f} "
                      f"+refine {row['jepa_refined_3d_mm']:5.1f} mm", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
    faulthandler.cancel_dump_traceback_later()
    df = pd.DataFrame(rows)
    cols = ["classical_2d_mm", "frozen_2d_mm", "jepa_2d_mm", "jepa_subcell_2d_mm", "jepa_refined_2d_mm",
            "jepa_3d_mm", "jepa_subcell_3d_mm", "jepa_refined_3d_mm"]
    for thr in (5, 10):
        print(f"\nfraction within {thr} mm\n" + df.groupby("condition")[cols].agg(lambda s: (s < thr).mean()).round(2).to_string())
    print("\nmedian error mm\n" + df.groupby("condition")[cols].median().round(1).to_string())


if __name__ == "__main__":
    main()
