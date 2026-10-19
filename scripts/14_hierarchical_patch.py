#!/usr/bin/env python3
"""Slice first, then pose: hierarchical 3D localisation of small REAL histology patches (raw SbR BigBrain).

Compared on the same patches (clean / damaged as in scripts/13):
* flat        JEPA joint argmax over all (plane, position) of the 1 mm bank (scripts/13 3D baseline)
* flat_dense  the same over a denser bank (y every 0.5 mm, pitch/yaw in {-10, -5, 0, 5, 10})
* hier        patch only, two stages: (1) which slice: per-y peak score, smoothed over neighbouring planes
              (a true match persists over +-1 mm, a chance peak does not), top candidates >= 6 mm apart;
              (2) where and at what tilt: position and orientation searched only around the chosen slice, on
              the y-averaged similarity map, sub-cell peak
* tta         patch embedding averaged over the 8 rotations/flips of the patch (model is rotation-trained)
* section     parent section first: the whole raw histology section is placed in 3D (classical MI pipeline),
              then the patch is found in 2D inside that estimated plane (needs the section the patch came from)
* oracle_2d   the patch in its true MRI slice (upper bound for 'section')
Every method also reports a confidence; the summary gives accuracy of the most confident half.
Output: results/hierarchical_patch.csv
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
from wms.classical import localise
from wms.jepa import TARGET_PX_MM, TARGET_SIZE, TOKEN_MM, PatchJEPA, footprint_tokens, window_maps
from wms.patches import cut_patch, sbr_geometry, sbr_mri_world, truth_mri_px
from wms.planes import as_params, rotation, sample

sbr = import_module("08_sbr_inplane")
p13 = import_module("13_patch_localisation")
ROOT = Path(__file__).resolve().parents[1]
ANGLES = (-10, -5, 0, 5, 10)
TOK_PX = TOKEN_MM / TARGET_PX_MM
HISTO_PX = 0.44                     # nominal scanner pixel size of the raw SbR sections; the pose search fits scale


def world_at(param, col, row, H=TARGET_SIZE, W=TARGET_SIZE, px=TARGET_PX_MM):
    """World mm of fractional pixel (col, row) of a plane rendered with params (7,) at px mm."""
    R = rotation(torch.tensor(np.asarray(param, np.float32)[None, 3:6]))[0].numpy()
    s = np.exp(param[6])
    return np.asarray(param[:3]) + s * ((col - (W - 1) / 2) * px * R[:, 0] + ((H - 1) / 2 - row) * px * R[:, 2])


def token_world(param, sim_map, r, c):
    cs, rs = p13.subcell(sim_map, r, c)
    return world_at(param, (cs + 0.5) * TOK_PX - 0.5, (rs + 0.5) * TOK_PX - 0.5)


def embed(jepa, x, tta):
    """x (1, 112, 112) -> z (D,); tta: mean over the 8 dihedral transforms."""
    xs = [x] if not tta else [torch.rot90(v, q, (1, 2)) for v in (x, x.flip(2)) for q in range(4)]
    with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        z = jepa.context(torch.cat(xs)).float()
    return F.normalize(z.mean(0), dim=0)


def bank_sims(z, maps, k, chunk=1024):
    """Similarity maps (M, g, g) of z against every bank plane, window-pooled to the patch footprint k."""
    out = []
    with torch.no_grad():
        for i in range(0, len(maps), chunk):
            m = maps[i:i + chunk].float()
            wm = window_maps(m, ks=(k,))[k] if k > 1 else F.normalize(m, dim=1)
            out.append(torch.einsum("d,mdhw->mhw", z, wm))
    return torch.cat(out)


def margin(scores, ys, best, sep=6.0):
    """Best score minus best score of planes >= sep mm away in y (separation of the winner from the runner-up)."""
    far = np.abs(ys - ys[best]) >= sep
    return float(scores[best] - scores[far].max()) if far.any() else 0.0


def flat(sims, params, ys):
    """Joint argmax over planes and positions."""
    M, g, _ = sims.shape
    best = sims.flatten(1).max(1).values.cpu().numpy()
    mi = int(best.argmax())
    r, c = divmod(int(sims[mi].flatten().argmax()), g)
    return token_world(params[mi], sims[mi].cpu().numpy(), r, c), params[mi][1], margin(best, ys, mi)


def hierarchical(sims, params, ny, no, sigma_mm=1.0, dy=0.5, half_mm=1.0):
    """(1) slice: per-y peak (max over orientation, position), Gaussian-smoothed over y; (2) pose: around the chosen
    y, average the similarity maps over +-half_mm per orientation, take the best orientation and position."""
    g = sims.shape[-1]
    S = sims.view(ny, no, g, g)
    peak = S.flatten(2).max(2).values.max(1).values                         # (ny,)
    t = torch.arange(-4, 5, device=sims.device, dtype=torch.float32) * dy
    kern = torch.exp(-t ** 2 / (2 * sigma_mm ** 2))
    smooth = F.conv1d(F.pad(peak[None, None], (4, 4), mode="replicate"), (kern / kern.sum())[None, None])[0, 0]
    sm = smooth.cpu().numpy()
    yi = int(sm.argmax())
    ys = params.reshape(ny, no, 7)[:, 0, 1]
    conf = margin(sm, ys, yi)
    h = int(round(half_mm / dy))
    lo, hi = max(0, yi - h), min(ny, yi + h + 1)
    avg = S[lo:hi].mean(0)                                                  # (no, g, g) y-averaged maps
    o = int(avg.flatten(1).max(1).values.argmax())
    r, c = divmod(int(avg[o].flatten().argmax()), g)
    # the plane: the chosen orientation at the y within the window whose map peaks highest at that position
    yb = lo + int(S[lo:hi, o, r, c].argmax())
    param = params.reshape(ny, no, 7)[yb, o]
    return token_world(param, avg[o].cpu().numpy(), r, c), float(ys[yi]), conf


def section_px_to_world(match, x, y, shape, px):
    """World mm of raw-section pixels (x, y) under a classical Match (pose of the section centre, fitted scale, flip)."""
    H, W = shape
    p = match.pose
    R = rotation(torch.tensor([[p.pitch, p.yaw, p.roll]], dtype=torch.float32))[0].numpy()
    u = (np.asarray(x, float) - (W - 1) / 2) * px * (-1 if match.flipped else 1)
    v = ((H - 1) / 2 - np.asarray(y, float)) * px
    return p.centre + np.exp(p.log_s) * (u[..., None] * R[:, 0] + v[..., None] * R[:, 2])


def section_canvas(vol, match, dev):
    """MRI plane at the section's estimated pose as a JEPA target canvas (0.8 mm, no scale), mirrored like the
    section if the match is flipped. Returns the canvas and a function token (c, r) -> world mm."""
    p = match.pose
    param = np.array([p.x, p.y, p.z, p.pitch, p.yaw, p.roll, 0.0], np.float32)
    with torch.no_grad():
        canvas = sample(vol, torch.tensor(param[None], device=dev), (TARGET_SIZE, TARGET_SIZE), TARGET_PX_MM)[0]
    if match.flipped:
        canvas = canvas.flip(-1)

    def to_world(sim_map, r, c):
        cs, rs = p13.subcell(sim_map, r, c)
        col = (cs + 0.5) * TOK_PX - 0.5
        col = TARGET_SIZE - 1 - col if match.flipped else col
        return world_at(param, col, (rs + 0.5) * TOK_PX - 0.5)
    return canvas, to_world


def peak_margin(sm, r, c, excl=2):
    """2D map confidence: peak minus the best value >= excl tokens away."""
    g = sm.shape[0]
    yy, xx = np.mgrid[0:g, 0:g]
    far = np.maximum(np.abs(yy - r), np.abs(xx - c)) >= excl
    return float(sm[r, c] - sm[far].max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-slices", type=int, default=40)
    ap.add_argument("--per-slice", type=int, default=2)
    ap.add_argument("--model", default="models/patch_jepa.pt")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    geom = sbr_geometry()
    mri_px = geom[4]
    ck = torch.load(ROOT / a.model, map_location=dev)
    jepa = PatchJEPA(ck["arch"], pretrained=False).to(dev).eval()
    jepa.load_state_dict(ck["state_dict"])
    vol = volumes.load("bigbrain_mri", dev)
    lo, hi = vol.bounds_mm
    cx, cz = geom[0], geom[1]

    # dense bank (y every 0.5 mm x 25 orientations); the 1 mm x {-10, 0, 10}^2 subset is the scripts/13 bank
    ys = np.arange(lo[1] + 3, hi[1] - 3, 0.5)
    orient = [(p, w) for p in ANGLES for w in ANGLES]
    params = np.array([(cx, y, cz, p, w, 0, 0) for y in ys for p, w in orient], np.float32)
    ny, no = len(ys), len(orient)
    feats = []
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        for i in range(0, len(params), 64):
            planes = sample(vol, torch.tensor(params[i:i + 64], device=dev), (TARGET_SIZE, TARGET_SIZE), TARGET_PX_MM)
            feats.append(jepa.target(planes).float().half())
    maps = torch.cat(feats)
    coarse_idx = np.array([yi * no + oi for yi in range(0, ny, 2) for oi, (p, w) in enumerate(orient)
                           if p in (-10, 0, 10) and w in (-10, 0, 10)])
    print(f"bank: {len(params)} planes ({ny} y x {no} orientations), {maps.numel() * 2 / 2**20:.0f} MB; "
          f"coarse subset {len(coarse_idx)}", flush=True)

    ref = pd.read_csv(ROOT / "results/sbr_BigBrain_inplane.csv")
    ids = sorted(np.random.default_rng(a.seed).choice(ref[ref.dice_reference >= 0.9].slice.values, a.n_slices, replace=False))
    out = ROOT / "results/hierarchical_patch.csv"
    rows = pd.read_csv(out).to_dict("records") if out.exists() and a.resume else []
    done = {int(r["slice"]) for r in rows}
    for sid in ids:
        if int(sid) in done:
            continue
        faulthandler.dump_traceback_later(900, exit=True, file=sys.stderr)
        rng = np.random.default_rng([a.seed, int(sid)])
        _, h, hm, m, mm = sbr.load_pieces("BigBrain", int(sid))[0]
        T_ref, _ = sbr.reference_affine("BigBrain", int(sid), "whole", h, hm, "cpu")
        y_true = geom[2] + geom[3] * int(sid)

        # stage 1 of 'section': place the whole raw section in 3D (no prior), error over its tissue pixels
        t0 = time.time()
        best, _ = localise(h, (HISTO_PX, HISTO_PX), vol, metric="mi",
                           coarse_kw=dict(angles=ANGLES, rolls=(-10, 0, 10), try_flip=True))
        sec = best[0]
        sec_s = time.time() - t0
        r_, c_ = np.nonzero(hm)
        pick = np.random.default_rng(0).choice(len(r_), min(4000, len(r_)), replace=False)
        hx, hy = c_[pick].astype(float), r_[pick].astype(float)
        mp = np.linalg.inv(T_ref) @ np.stack([hx, hy, np.ones_like(hx)])                # histology -> MRI px
        sec_err = np.linalg.norm(section_px_to_world(sec, hx, hy, h.shape, HISTO_PX)
                                 - sbr_mri_world(int(sid), mp[0], mp[1], m.shape, geom), axis=1)
        canvas_sec, sec_world = section_canvas(vol, sec, dev)
        canvas_true = p13.canvas_from_slice(m, mri_px).to(dev)
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            maps_sec = window_maps(jepa.target(canvas_sec[None]).float())
            maps_true = window_maps(jepa.target(canvas_true[None]).float())
        print(f"slice {sid:03d}: section placed in 3D, mean err {sec_err.mean():.2f} mm, y {sec.pose.y:.1f} vs "
              f"{y_true:.1f}, flip={sec.flipped}, score {sec.score:.3f}, {sec_s:.0f}s", flush=True)

        for cond, dist in p13.CONDITIONS.items():
            for j in range(a.per_slice):
                p = cut_patch(h, hm, T_ref, rng, distortion=dist)
                if p is None:
                    continue
                tx, ty = truth_mri_px(p, T_ref)
                truth = sbr_mri_world(int(sid), tx, ty, m.shape, geom)
                size_mm = p.side_px * p.px_mm
                k = footprint_tokens(size_mm)
                x_in = p13.patch_input(p).to(dev)[None]
                row = dict(slice=int(sid), condition=cond, size_mm=size_mm, section_err_mm=sec_err.mean(),
                           section_y_err_mm=abs(sec.pose.y - y_true), section_score=sec.score, section_s=sec_s)
                t0 = time.time()
                for tag, tta in (("", False), ("_tta", True)):
                    z = embed(jepa, x_in, tta)
                    # 2D in the true slice (oracle) and in the estimated section plane
                    for name, mp_, to_w in (("oracle_2d", maps_true, None), ("section", maps_sec, sec_world)):
                        sm = torch.einsum("d,dhw->hw", z, mp_[k][0]).float().cpu().numpy()
                        r, c = np.unravel_index(sm.argmax(), sm.shape)
                        if to_w is None:
                            xs, ys_ = p13.canvas_px_to_slice(*p13.subcell(sm, r, c), m.shape, mri_px)
                            row[f"{name}{tag}_mm"] = np.hypot(xs - tx, ys_ - ty) * mri_px
                        else:
                            row[f"{name}{tag}_mm"] = float(np.linalg.norm(to_w(sm, r, c) - truth))
                        row[f"{name}{tag}_conf"] = peak_margin(sm, r, c)
                    sims = bank_sims(z, maps, k)
                    for name, idx in (("flat", coarse_idx), ("flat_dense", np.arange(len(params)))):
                        w, y_est, conf = flat(sims[torch.as_tensor(idx, device=sims.device)], params[idx], params[idx][:, 1])
                        row[f"{name}{tag}_mm"] = float(np.linalg.norm(w - truth))
                        row[f"{name}{tag}_y_err"] = abs(y_est - y_true)
                        row[f"{name}{tag}_conf"] = conf
                    w, y_est, conf = hierarchical(sims, params, ny, no)
                    row[f"hier{tag}_mm"] = float(np.linalg.norm(w - truth))
                    row[f"hier{tag}_y_err"] = abs(y_est - y_true)
                    row[f"hier{tag}_conf"] = conf
                row["patch_s"] = time.time() - t0
                rows.append(row)
                print(f"  {cond:7s} {size_mm:4.1f} mm | flat {row['flat_mm']:5.1f} dense {row['flat_dense_mm']:5.1f} "
                      f"hier {row['hier_mm']:5.1f} hier+tta {row['hier_tta_mm']:5.1f} | section {row['section_mm']:5.1f} "
                      f"+tta {row['section_tta_mm']:5.1f} | oracle 2D {row['oracle_2d_mm']:5.1f} +tta {row['oracle_2d_tta_mm']:5.1f}",
                      flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
    faulthandler.cancel_dump_traceback_later()
    summarise(pd.DataFrame(rows))


METHODS = ["flat", "flat_dense", "hier", "flat_tta", "hier_tta", "section", "section_tta", "oracle_2d", "oracle_2d_tta"]


def summarise(df):
    for thr in (5, 10):
        print(f"\nfraction within {thr} mm\n" + df.groupby("condition")[[f"{m}_mm" for m in METHODS]]
              .agg(lambda s: (s < thr).mean()).round(2).to_string())
    print("\nmedian error mm\n" + df.groupby("condition")[[f"{m}_mm" for m in METHODS]].median().round(1).to_string())
    ycols = [c for c in df.columns if c.endswith("_y_err")] + ["section_y_err_mm"]
    print("\nslice found (|y err| <= 2 mm)\n" + df.groupby("condition")[ycols].agg(lambda s: (s <= 2).mean()).round(2).to_string())
    # confidence: within 10 mm among the most confident half (per condition), vs all
    res = {}
    for m in METHODS:
        conf = f"{m}_conf"
        for cond, d in df.groupby("condition"):
            top = d[d[conf] >= d[conf].median()]
            res[(m, cond)] = ((d[f"{m}_mm"] < 10).mean(), (top[f"{m}_mm"] < 10).mean())
    print("\nwithin 10 mm: all -> most confident half")
    for (m, cond), (al, tp) in res.items():
        print(f"  {m:14s} {cond:7s} {al:.2f} -> {tp:.2f}")


if __name__ == "__main__":
    main()
