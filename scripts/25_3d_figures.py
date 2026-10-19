#!/usr/bin/env python3
"""3D views: the brain surface (from the BigBrain MRI) with the planes we found.

patch:   the histology piece (scripts/23 example): its parent section's MRI slice in place in 3D, with our spot
         and the true spot of the piece -> figures/patch_3d.png
oblique: an obliquely cut, rotated histology section (scripts/24 example) drawn in 3D at the pose we found,
         with the true plane outlined -> figures/oblique_3d.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from skimage.measure import marching_cubes

from wms import volumes
from wms.planes import plane_points, rotation

ROOT = Path(__file__).resolve().parents[1]
NAVY, TEAL, ORANGE, RED = "#002448", "#1F6E7A", "#D4A54D", "#E8312A"


def brain_mesh(vol, mm=2.0, level=0.12):
    vox = float(np.abs(np.diag(vol.affine)[:3]).mean())
    k = max(1, int(round(mm / vox)))
    d = F.avg_pool3d(vol.data[None, None].float(), k, stride=k)[0, 0].numpy()
    verts, faces, _, _ = marching_cubes(d, level=level, step_size=1)
    ijk = verts * k + (k - 1) / 2
    world = ijk @ vol.affine[:3, :3].T + vol.affine[:3, 3]
    return world[faces]


def scene(ax, tris, elev=16, azim=-58):
    ax.computed_zorder = False                          # draw in call order: markers drawn last stay on top
    ax.add_collection3d(Poly3DCollection(tris, facecolor=(0.45, 0.52, 0.62, 0.09), edgecolor="none"))
    lo, hi = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def textured_plane(ax, pts, img, cmap="gray", thr=0.06, step=3, alpha=0.95, invert=False):
    P = pts[::step, ::step]
    I = img[::step, ::step]
    rgba = plt.get_cmap(cmap)(np.clip(1 - I if invert else I, 0, 1))
    rgba[..., 3] = np.where(I > thr, alpha, 0.0)
    ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], facecolors=rgba, rstride=1, cstride=1, shade=False, linewidth=0)


def outline(ax, pts, color, ls="-", lw=2.5):
    c = np.array([pts[0, 0], pts[0, -1], pts[-1, -1], pts[-1, 0], pts[0, 0]])
    ax.plot(c[:, 0], c[:, 1], c[:, 2], color=color, ls=ls, lw=lw)


def square(ax, centre, R, s, color, ls="-", lw=3):
    u, v = R[:, 0] * s / 2, R[:, 2] * s / 2
    c = np.array([centre - u - v, centre + u - v, centre + u + v, centre - u + v, centre - u - v])
    ax.plot(c[:, 0], c[:, 1], c[:, 2], color=color, ls=ls, lw=lw)


def patch_view(tris):
    e = np.load(ROOT / "results/patch_pipeline_example.npz")
    param = e["sec_pose"].astype(np.float32)
    canvas = e["canvas"]
    pts = plane_points(torch.tensor(param[None]), canvas.shape, 0.8)[0].numpy()
    if bool(e["flipped"]):
        pts = pts[:, ::-1]
    R = rotation(torch.tensor(param[None, 3:6]))[0].numpy()
    fig = plt.figure(figsize=(16, 7.5), facecolor="white")
    ax0 = fig.add_axes([0.0, 0.18, 0.2, 0.62])
    ax0.imshow(1 - e["patch"], cmap="gray", vmin=0, vmax=1)
    ax0.set_title(f"histology piece\n({float(e['size_mm']):.0f} mm)", fontsize=20, color=NAVY)
    ax0.set_xticks([]); ax0.set_yticks([])
    for sp in ax0.spines.values():
        sp.set_edgecolor(ORANGE); sp.set_linewidth(5)
    ax = fig.add_axes([0.2, 0.0, 0.8, 1.0], projection="3d")
    scene(ax, tris)
    textured_plane(ax, pts, canvas / max(canvas.max(), 1e-6), thr=0.08, step=1, alpha=0.9)
    s = float(e["size_mm"])
    n = R[:, 1] * 1.0                                   # draw the squares just in front of the slice
    square(ax, e["truth"] - n, R, s, ORANGE, ls="--", lw=4)
    square(ax, e["w_est"] - n, R, s, RED, lw=5)
    ax.scatter(*(e["w_est"] - n), color=RED, s=60, depthshade=False)
    ax.plot([], [], color="#888888", lw=8, label="MRI slice its section was placed in")
    ax.plot([], [], color=RED, lw=4, label=f"where we put the piece ({float(e['err']):.0f} mm from truth)")
    ax.plot([], [], color=ORANGE, lw=3, ls="--", label="where it truly is")
    ax.legend(loc="upper left", fontsize=15, frameon=False, bbox_to_anchor=(0.02, 0.98))
    fig.savefig(ROOT / "figures/patch_3d.png", dpi=90, facecolor="white", bbox_inches="tight")


def oblique_view(tris, tilt):
    e = np.load(ROOT / "results/oblique_rotation_examples.npz")
    q = e[f"query_{tilt}"]
    tp = e[f"true_pose_{tilt}"].astype(np.float32)
    ep = e[f"est_pose_{tilt}"].astype(np.float32)
    flipped = bool(e[f"flipped_{tilt}"])
    tilt_err, rot_err, off, align = e[f"err_{tilt}"]
    pt = plane_points(torch.tensor(tp[None]), q.shape, 0.4)[0].numpy()
    pe = plane_points(torch.tensor(ep[None]), q.shape, 0.4)[0].numpy()
    if flipped:
        pe = pe[:, ::-1]
    fig = plt.figure(figsize=(16, 7.5), facecolor="white")
    ax0 = fig.add_axes([0.0, 0.26, 0.24, 0.6])
    ax0.imshow(1 - q, cmap="gray", vmin=0, vmax=1)
    ax0.set_title("histology cut at an\noblique angle, rotated", fontsize=20, color=NAVY)
    ax0.axis("off")
    ax = fig.add_axes([0.22, 0.0, 0.78, 1.0], projection="3d")
    scene(ax, tris)
    textured_plane(ax, pe, q, thr=0.06, step=3, invert=True)
    outline(ax, pe, TEAL, lw=3)
    outline(ax, pt, ORANGE, ls="--", lw=3)
    ax.plot([], [], color=TEAL, lw=3, label="pose we found (section drawn in place)")
    ax.plot([], [], color=ORANGE, lw=3, ls="--", label="true pose")
    ax.legend(loc="upper left", fontsize=15, frameon=False, bbox_to_anchor=(0.02, 0.98))
    fig.text(0.12, 0.08, f"cut {tilt}° off coronal\nerrors: angle {tilt_err:.1f}°,\nrotation {rot_err:.1f}°, "
                         f"alignment {align:.1f} mm", ha="center", fontsize=17, color=NAVY)
    fig.savefig(ROOT / f"figures/oblique_3d_{tilt}.png", dpi=90, facecolor="white", bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["patch", "oblique"])
    ap.add_argument("--tilt", type=int, default=45)
    a = ap.parse_args()
    tris = brain_mesh(volumes.load("bigbrain_mri", "cpu"))
    patch_view(tris) if a.what == "patch" else oblique_view(tris, a.tilt)
    print("done")


if __name__ == "__main__":
    main()
