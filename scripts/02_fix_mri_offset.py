#!/usr/bin/env python3
"""The released BigBrain MRI (mri_to_blk_aligned) sits ~1 mm off the 3D histology along y. Estimate the
translation that maximises tissue-mask Dice (modality-free) and save it; volumes.load applies it.
Output: data/bigbrain/mri/offset_mm.json
"""
import json
import itertools

import numpy as np
import torch
import torch.nn.functional as F

from wms import volumes


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    h = volumes.load("bigbrain_400", dev)
    m = volumes.load("bigbrain_mri", dev, apply_offset=False)
    ii = torch.stack(torch.meshgrid(*[torch.arange(s, device=dev, dtype=torch.float32) for s in h.shape], indexing="ij"), -1)
    A = torch.tensor(h.affine, dtype=torch.float32, device=dev)
    inv = torch.tensor(np.linalg.inv(m.affine), dtype=torch.float32, device=dev)
    world = ii @ A[:3, :3].T + A[:3, 3]
    hm = (h.data > 0.08).float()
    size = torch.tensor(m.shape, dtype=torch.float32, device=dev)

    def dice(t):
        v = (world + torch.tensor(t, dtype=torch.float32, device=dev)) @ inv[:3, :3].T + inv[:3, 3]
        mm = F.grid_sample(m.data[None, None], (2 * v / (size - 1) - 1).flip(-1)[None], align_corners=True)[0, 0]
        mk = (mm > 0.08).float()
        return float(2 * (hm * mk).sum() / (hm.sum() + mk.sum()))

    best = np.zeros(3)
    for step in (0.5, 0.25, 0.1):
        cands = [best + step * np.array(d) for d in itertools.product((-2, -1, 0, 1, 2), repeat=3)]
        scores = [dice(c) for c in cands]
        best = cands[int(np.argmax(scores))]
        print(f"step {step} mm: MRI content found at histology + {np.round(best, 2)} mm, Dice {max(scores):.4f} (no shift {dice([0, 0, 0]):.4f})")
    # MRI voxel content at world p + t matches histology at p, so the MRI affine must move by -t
    out = volumes.DATA / "bigbrain/mri/offset_mm.json"
    out.write_text(json.dumps({"translate_mm": (-best).round(3).tolist(),
                               "note": "added to the MRI affine translation; estimated by tissue-mask Dice vs full8_400um"}))
    print("saved", out)


if __name__ == "__main__":
    main()
