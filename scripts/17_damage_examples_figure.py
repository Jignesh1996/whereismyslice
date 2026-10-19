#!/usr/bin/env python3
"""What the synthetic damage looks like: one real SbR BigBrain section, undamaged and under each damage type of
the stress test (scripts/11, 15). Output: figures/damage_examples.png
"""
import argparse
import sys
from importlib import import_module
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from wms.distort import apply, random_distortion

sbr = import_module("08_sbr_inplane")
ROOT = Path(__file__).resolve().parents[1]
PANELS = [("undamaged", dict(rotation=False)), ("rotated", dict(rotation=True)),
          ("stretched 10 %", dict(rotation=False, affine=0.10)), ("cuts + tears", dict(rotation=False, cuts=True)),
          ("warped 10 %", dict(rotation=False, elastic=0.10)),
          ("all at once", dict(rotation=True, affine=0.10, elastic=0.10, cuts=True))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=int, default=94)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="figures/damage_examples.png")
    a = ap.parse_args()
    _, h, hm, _, _ = sbr.load_pieces("BigBrain", a.slice)[0]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(3.3 * len(PANELS), 3.6), facecolor="white")
    for ax, (name, kw) in zip(axes, PANELS):
        dist = random_distortion(h.shape, hm, np.random.default_rng(a.seed), **kw)
        hd, hmd = apply(dist, h, hm)
        r, c = np.nonzero(hmd)
        pad = 6
        crop = (1 - hd)[max(r.min() - pad, 0):r.max() + pad, max(c.min() - pad, 0):c.max() + pad]   # stain look
        ax.imshow(crop, cmap="gray", vmin=0, vmax=1)
        ax.set_title(name, fontsize=17, color="#002448", pad=8)
        ax.axis("off")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(ROOT / a.out, dpi=110, facecolor="white")
    print(ROOT / a.out)


if __name__ == "__main__":
    main()
