"""Real BigBrain 2D coronal sections (the 7404 sections the 3D volume was reconstructed from).

pm0001 is the occipital pole and numbers increase towards the frontal pole (posterior -> anterior).
Full resolution is 20 um/px (6572 x 5711); thumbnails are the same field of view at 500 x 434.
Images are returned tissue-bright on a zero background, like the volumes.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from .volumes import DATA

SECTION_UM = 20.0
N_SECTIONS = 7404
FULL_SHAPE = (5711, 6572)                                     # rows, cols at 20 um
PX_MM = {
    "full_res": (0.02, 0.02),
    "thumbnail": (FULL_SHAPE[0] * 0.02 / 434, FULL_SHAPE[1] * 0.02 / 500),   # (row, col) ~0.263 mm
}


def path(n, res="thumbnail"):
    return DATA / "sections" / res / f"pm{n:04d}o.png"


def available(res="thumbnail"):
    return sorted(int(p.stem[2:6]) for p in (DATA / "sections" / res).glob("pm*o.png"))


def to_tissue_bright(gray_u8):
    """White-background histology (uint8) -> float32 in [0, 1], tissue bright, background ~0."""
    v = 1 - np.asarray(gray_u8, np.float32) / 255.0
    bg = np.median(np.concatenate([v[:4].ravel(), v[-4:].ravel(), v[:, :4].ravel(), v[:, -4:].ravel()]))
    v = np.clip(v - bg, 0, None)
    hi = np.percentile(v[v > 0.02], 99.5) if (v > 0.02).any() else 1.0
    return np.clip(v / max(hi, 1e-6), 0, 1).astype(np.float32)


def load(n, res="thumbnail", reduce=1):
    """Section n -> (image (H, W) float32 tissue-bright, px_mm (row, col)).

    reduce > 1 box-downsamples in PIL before converting to float (e.g. reduce=5 turns a 20 um section
    into 0.1 mm, ~25x less memory).
    """
    im = Image.open(path(n, res)).convert("L")
    if reduce > 1:
        sx, sy = im.size[0] / (im.size[0] // reduce), im.size[1] / (im.size[1] // reduce)
        im = im.reduce(reduce)
        px = (PX_MM[res][0] * sy, PX_MM[res][1] * sx)
    else:
        px = PX_MM[res]
    return to_tissue_bright(im), px
