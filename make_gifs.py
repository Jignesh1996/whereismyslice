#!/usr/bin/env python3
"""Where's My Slice? demo GIFs from real BigBrain data (fallback: MNI152)."""
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from scipy.ndimage import map_coordinates, rotate, gaussian_filter, zoom


def load_volume(res_mm):
    try:
        import siibra
        img = siibra.get_template("bigbrain").fetch(resolution_mm=res_mm)
        name = "BigBrain"
    except Exception as err:
        print(f"Could not fetch BigBrain via siibra ({err}); falling back to MNI152.")
        from nilearn import datasets
        img = datasets.load_mni152_template(resolution=1)
        name = "MNI152"
    img = nib.as_closest_canonical(img)
    vox = float(abs(img.header.get_zooms()[1]))
    vol = np.asarray(img.dataobj, dtype=np.float32)
    lo, hi = np.percentile(vol, [1, 99.5])
    vol = np.clip((vol - lo) / (hi - lo + 1e-6), 0, 1)
    corners = vol[[0, -1]][:, [0, -1]][:, :, [0, -1]]
    return vol, float(np.median(corners)), name, vox


def oblique_slice(vol, yc, theta_deg, bg):
    """Plane through (x-centre, yc, z-centre); tilted by theta around the left-right axis."""
    X, Y, Z = vol.shape
    th = np.deg2rad(theta_deg)
    L = int(np.hypot(Y, Z))
    S, T = np.meshgrid(np.arange(X) - (X - 1) / 2, np.arange(L) - L / 2)
    px = (X - 1) / 2 + S
    py = yc - T * np.sin(th)
    pz = (Z - 1) / 2 + T * np.cos(th)
    return map_coordinates(vol, [px, py, pz], order=1, cval=bg)


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def tissue_bbox(imgs, bg, thr=0.08):
    m = np.zeros(imgs[0].shape, bool)
    for im in imgs:
        m |= np.abs(im - bg) > thr
    rows, cols = np.where(m.any(1))[0], np.where(m.any(0))[0]
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def make_search_gif(vol, bg, name, vox, out, fps):
    X, Y, Z = vol.shape
    yq, thq = 0.58 * Y, 12.0
    rng = np.random.default_rng(0)
    q = oblique_slice(vol, yq, thq, bg)
    q = rotate(q, 4, reshape=False, order=1, mode="constant", cval=bg)
    q = np.clip(q + rng.normal(0, 0.04, q.shape), 0, 1)
    q_s = gaussian_filter(q, 1.5)

    poses = [(y, 0.0) for y in np.linspace(0.2 * Y, 0.85 * Y, 40)]
    cands = [oblique_slice(vol, y, t, bg) for y, t in poses]
    scores = [ncc(gaussian_filter(c, 1.5), q_s) for c in cands]
    n1 = len(poses)
    yb = poses[int(np.argmax(scores))][0]
    for t in np.linspace(-20, 20, 25):
        c = oblique_slice(vol, yb, t, bg)
        poses.append((yb, t)); cands.append(c); scores.append(ncc(gaussian_filter(c, 1.5), q_s))
    best = n1 + int(np.argmax(scores[n1:]))
    n_search, n_hold = len(poses), 20

    r0, r1, c0, c1 = tissue_bbox(cands + [q], bg)
    crop = lambda im: im[r0:r1, c0:c1]
    sag, Zc, L = vol[X // 2].T, (Z - 1) / 2, cands[0].shape[0]
    smin, smax = min(scores), max(scores)

    fig = plt.figure(figsize=(12, 6.75), dpi=80, facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.1], hspace=0.35, wspace=0.08)
    ax_sag, ax_c, ax_q = (fig.add_subplot(gs[0, i]) for i in range(3))
    ax_s = fig.add_subplot(gs[1, :])

    def update(i):
        done = i >= n_search
        k = best if done else i
        y, th = poses[k]
        col = "limegreen" if done else "orange"
        for ax in (ax_sag, ax_c, ax_q, ax_s):
            ax.clear()
        ax_sag.imshow(sag, cmap="gray", origin="lower", vmin=0, vmax=1)
        t, r = np.array([-L / 2, L / 2]), np.deg2rad(th)
        ax_sag.plot(y - t * np.sin(r), Zc + t * np.cos(r), color=col, lw=3)
        ax_sag.set_xlim(0, Y - 1); ax_sag.set_ylim(0, Z - 1)
        ax_sag.set_title(f"3D reference ({name}), side view", fontsize=11)
        ax_c.imshow(crop(cands[k]), cmap="gray", origin="lower", vmin=0, vmax=1)
        ax_c.set_title("Slice at this cutting plane", fontsize=11, color="green" if done else "black")
        ax_q.imshow(crop(q), cmap="gray", origin="lower", vmin=0, vmax=1)
        ax_q.set_title("Query: loose histology section", fontsize=11)
        for ax in (ax_sag, ax_c, ax_q):
            ax.axis("off")
        n = n_search if done else i + 1
        ax_s.plot(range(n), scores[:n], color="tab:orange", lw=2)
        ax_s.axvline(n1 - 0.5, color="gray", ls="--", lw=1)
        ax_s.set_xlim(0, n_search); ax_s.set_ylim(smin - 0.02, smax + 0.04)
        ax_s.set_ylabel("match score"); ax_s.set_xticks([])
        ax_s.text(n1 / 2, smax + 0.03, "step 1: scan position", ha="center", va="top")
        ax_s.text(n1 + (n_search - n1) / 2, smax + 0.03, "step 2: refine tilt", ha="center", va="top")
        if done:
            ax_s.plot(best, scores[best], "o", color="limegreen", ms=10)
            status = f"Found: {y * vox:.1f} mm, tilt {th:.0f}°  (true: {yq * vox:.1f} mm, {thq:.0f}°)"
        else:
            status = "Searching for where this section came from…"
        fig.suptitle(f"Where's My Slice?   {status}", fontsize=13)

    FuncAnimation(fig, update, frames=n_search + n_hold).save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)


def make_stack_gif(vol, bg, out, fps, n=12):
    X, Y, Z = vol.shape
    raw = [oblique_slice(vol, y, 0.0, bg) for y in np.linspace(0.25 * Y, 0.8 * Y, n)]
    r0, r1, c0, c1 = tissue_bbox(raw, bg)
    f = 110 / (c1 - c0)
    thumbs = [np.clip(zoom(im[r0:r1, c0:c1], f, order=1), 0, 1) for im in raw]
    order = list(range(n)); order[3], order[8] = order[8], order[3]  # slot -> section
    start = {sec: slot for slot, sec in enumerate(order)}
    flipped = {6}
    bad = {k for k in range(n) if start[k] != k} | flipped
    w, gap = 1.0, 0.1
    h = w * thumbs[0].shape[0] / thumbs[0].shape[1]
    A, B, C = 15, 30, 20

    fig, ax = plt.subplots(figsize=(12, 4), dpi=80, facecolor="white")

    def update(i):
        ax.clear(); ax.axis("off")
        p = np.clip((i - A) / B, 0, 1); p = p * p * (3 - 2 * p)
        for k in range(n):
            s = start[k]
            x = (s + (k - s) * p) * (w + gap)
            lift = 0.5 * h * np.sin(np.pi * p) if k in bad else 0.0
            img = thumbs[k][:, ::-1] if (k in flipped and p < 0.5) else thumbs[k]
            ax.imshow(img, cmap="gray", origin="lower", vmin=0, vmax=1,
                      extent=[x, x + w, lift, lift + h])
            if k in bad and i >= 5:
                ax.add_patch(Rectangle((x, lift), w, h, fill=False, lw=2.5,
                                       ec="limegreen" if p >= 1 else "red"))
            ax.text(x + w / 2, -0.08 * h, f"#{k + 1}", ha="center", va="top", fontsize=10)
        ax.set_xlim(-0.1, n * (w + gap)); ax.set_ylim(-0.35 * h, 1.6 * h)
        msg = ("Corrupted stack: 2 sections swapped, 1 flipped" if i < 5 else
               "Detected: sections out of place (red)" if i < A else
               "Restoring order and orientation…" if p < 1 else "Stack restored")
        ax.set_title(f"Where's My Slice?   {msg}", fontsize=13)

    FuncAnimation(fig, update, frames=A + B + C).save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=float, default=0.64, help="BigBrain resolution in mm")
    ap.add_argument("--out-dir", default="assets")
    ap.add_argument("--fps", type=int, default=10)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    vol, bg, name, vox = load_volume(a.res)
    print(f"Loaded {name}: shape {vol.shape}, voxel {vox:.2f} mm")
    make_search_gif(vol, bg, name, vox, out / "wheres_my_slice_search.gif", a.fps)
    make_stack_gif(vol, bg, out / "wheres_my_slice_stack.gif", a.fps)
    print(f"Saved GIFs to {out}/")


if __name__ == "__main__":
    main()
