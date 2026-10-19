#!/usr/bin/env python3
"""Can the 'section first' patch route (scripts/14) be made more precise? Same 320 real patches, same draws.

* jepa          baseline: JEPA (+ rotation/flip averaging) on the MRI plane at the section's estimated pose,
                11.2 mm token grid with sub-cell peak
* jepa_shift    the same on 4 copies of the MRI plane shifted by half a token (5.6 mm grid), best peak overall
* in_section    the piece is found inside its own parent histology section (same modality: rotation x scale
                template search, NCC), then carried into 3D by the section's estimated pose (MRI search)
* in_section_refined   the same, but the section is first aligned inside the MRI plane at its estimated pose
                (in-plane affine, wms.inplane, as in scripts/08) before the piece is carried into 3D
Errors against the SbR truth and against the offset-corrected truth (SbR plane + 1.0 mm in y, scripts/18).
Output: results/patch_improve.csv
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
from wms.classical import localise
from wms.jepa import TARGET_PX_MM, TARGET_SIZE, TOKEN_MM, PatchJEPA, footprint_tokens, window_maps
from wms.inplane import localise as inplane_localise
from wms.patches import classical_patch_search, cut_patch, sbr_geometry, sbr_mri_world, truth_mri_px
from wms.planes import rotation, sample

sbr = import_module("08_sbr_inplane")
p13 = import_module("13_patch_localisation")
p14 = import_module("14_hierarchical_patch")
ROOT = Path(__file__).resolve().parents[1]
SBR_OFFSET = np.array([0.0, 1.0, 0.0])
TOK_PX = TOKEN_MM / TARGET_PX_MM


def shifted_maps(jepa, vol, match, dev):
    """JEPA target maps on 4 copies of the section's MRI plane shifted by (0|half token) along u and v."""
    p = match.pose
    R = rotation(torch.tensor([[p.pitch, p.yaw, p.roll]], dtype=torch.float32))[0].numpy()
    out = []
    for du in (0.0, TOKEN_MM / 2):
        for dv in (0.0, TOKEN_MM / 2):
            c = np.array([p.x, p.y, p.z]) + du * R[:, 0] + dv * R[:, 2]
            param = np.array([*c, p.pitch, p.yaw, p.roll, 0.0], np.float32)
            with torch.no_grad():
                cv = sample(vol, torch.tensor(param[None], device=dev), (TARGET_SIZE, TARGET_SIZE), TARGET_PX_MM)[0]
                if match.flipped:
                    cv = cv.flip(-1)
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                    maps = window_maps(jepa.target(cv[None]).float())
            out.append((param, maps))
    return out


def best_shifted(z, k, shifted, flipped):
    best = None
    for param, maps in shifted:
        sm = torch.einsum("d,dhw->hw", z, maps[k][0]).float().cpu().numpy()
        r, c = np.unravel_index(sm.argmax(), sm.shape)
        if best is None or sm[r, c] > best[0]:
            cs, rs = p13.subcell(sm, r, c)
            col = (cs + 0.5) * TOK_PX - 0.5
            col = TARGET_SIZE - 1 - col if flipped else col
            best = (sm[r, c], p14.world_at(param, col, (rs + 0.5) * TOK_PX - 0.5))
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-slices", type=int, default=80)
    ap.add_argument("--per-slice", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    geom = sbr_geometry()
    mri_px = geom[4]
    ck = torch.load(ROOT / "models/patch_jepa.pt", map_location=dev)
    jepa = PatchJEPA(ck["arch"], pretrained=False).to(dev).eval()
    jepa.load_state_dict(ck["state_dict"])
    vol = volumes.load("bigbrain_mri", dev)
    ref = pd.read_csv(ROOT / "results/sbr_BigBrain_inplane.csv")
    ids = sorted(np.random.default_rng(a.seed).choice(ref[ref.dice_reference >= 0.9].slice.values, a.n_slices, replace=False))
    out = ROOT / "results/patch_improve.csv"
    rows = []
    for sid in ids:
        faulthandler.dump_traceback_later(600, exit=True, file=sys.stderr)
        rng = np.random.default_rng([a.seed, int(sid)])
        _, h, hm, m, mm = sbr.load_pieces("BigBrain", int(sid))[0]
        T_ref, _ = sbr.reference_affine("BigBrain", int(sid), "whole", h, hm, "cpu")
        sec = localise(h, (p14.HISTO_PX, p14.HISTO_PX), vol, metric="mi",
                       coarse_kw=dict(angles=p14.ANGLES, rolls=(-10, 0, 10), try_flip=True))[0][0]
        canvas, to_world = p14.section_canvas(vol, sec, dev)
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            maps = window_maps(jepa.target(canvas[None]).float())
        shifted = shifted_maps(jepa, vol, sec, dev)
        # in-plane alignment of the section inside the MRI plane at its estimated pose
        PH, PW, PPX = 360, 400, 0.4
        sp = sec.pose
        pparam = np.array([sp.x, sp.y, sp.z, sp.pitch, sp.yaw, sp.roll, 0.0], np.float32)
        with torch.no_grad():
            plane = sample(vol, torch.tensor(pparam[None], device=dev), (PH, PW), PPX)[0].cpu().numpy()
        if sec.flipped:
            plane = plane[:, ::-1].copy()
        T_in = inplane_localise(h, hm, plane, plane > 0.05, device=dev).T          # plane px -> histology px
        T_inv = np.linalg.inv(T_in)

        def histo_to_world(x, y):
            q = T_inv @ np.array([x, y, 1.0])
            col = PW - 1 - q[0] if sec.flipped else q[0]
            return p14.world_at(pparam, col, q[1], PH, PW, PPX)
        for cond, dist in p13.CONDITIONS.items():
            for j in range(a.per_slice):
                p = cut_patch(h, hm, T_ref, rng, distortion=dist)
                if p is None:
                    continue
                tx, ty = truth_mri_px(p, T_ref)
                truth = sbr_mri_world(int(sid), tx, ty, m.shape, geom)
                k = footprint_tokens(p.side_px * p.px_mm)
                z = p14.embed(jepa, p13.patch_input(p).to(dev)[None], tta=True)
                sm = torch.einsum("d,dhw->hw", z, maps[k][0]).float().cpu().numpy()
                r, c = np.unravel_index(sm.argmax(), sm.shape)
                est = {"jepa": to_world(sm, r, c), "jepa_shift": best_shifted(z, k, shifted, sec.flipped)}
                t0 = time.time()
                pad = int(max(p.img.shape) * 1.1 * 1.45) + 2                 # room for the rotated template
                s_, x, y, _, _ = classical_patch_search(p, np.pad(h, pad), mri_px_mm=p.px_mm, device=dev, top_k=1,
                                                        angles=np.arange(0, 360, 5))[0]
                x, y = x - pad, y - pad
                est["in_section"] = p14.section_px_to_world(sec, np.array([x]), np.array([y]), h.shape, p14.HISTO_PX)[0]
                est["in_section_refined"] = histo_to_world(x, y)
                row = dict(slice=int(sid), condition=cond, size_mm=p.side_px * p.px_mm, in_section_s=time.time() - t0,
                           in_section_score=s_)
                for name, w in est.items():
                    row[f"{name}_mm"] = float(np.linalg.norm(w - truth))
                    row[f"{name}_corr_mm"] = float(np.linalg.norm(w - (truth + SBR_OFFSET)))
                rows.append(row)
                print(f"slice {sid:03d} {cond:7s} | JEPA {row['jepa_corr_mm']:5.1f}  shifted {row['jepa_shift_corr_mm']:5.1f}  "
                      f"in section {row['in_section_corr_mm']:5.1f}  + in-plane {row['in_section_refined_corr_mm']:5.1f} mm (offset-corrected truth)", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
    faulthandler.cancel_dump_traceback_later()
    df = pd.DataFrame(rows)
    cols = [f"{n}{s}_mm" for n in ("jepa", "jepa_shift", "in_section", "in_section_refined") for s in ("", "_corr")]
    print("\nwithin 10 mm\n" + df.groupby("condition")[cols].agg(lambda s: (s < 10).mean()).round(2).to_string())
    print("\nwithin 5 mm\n" + df.groupby("condition")[cols].agg(lambda s: (s < 5).mean()).round(2).to_string())
    print("\nwithin 2 mm\n" + df.groupby("condition")[cols].agg(lambda s: (s < 2).mean()).round(2).to_string())
    print("\nmedian mm\n" + df.groupby("condition")[cols].median().round(2).to_string())


if __name__ == "__main__":
    main()
