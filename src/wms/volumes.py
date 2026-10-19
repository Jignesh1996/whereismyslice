"""Registry and loader for the 3D volumes. Every volume is held as float32 in [0, 1] with its voxel->mm affine.

Two paired settings:
  * "histological" space: BigBrain histology <-> BigBrain's own post-mortem MRI (same brain, aligned)
  * "icbm2009b_sym" space: BigBrain warped to MNI <-> ICBM152 2009b sym T1 (cross-subject MRI target)
"""
import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

DATA = Path(__file__).resolve().parents[2] / "data"

VOLUMES = {
    # name: (path, space, modality)
    "bigbrain_200": ("bigbrain/histological/full8_200um_optbal.nii.gz", "histological", "histology"),
    "bigbrain_400": ("bigbrain/histological/full8_400um_optbal.nii.gz", "histological", "histology"),
    "bigbrain_1000": ("bigbrain/histological/full8_1000um_optbal.nii.gz", "histological", "histology"),
    "bigbrain_mri": ("bigbrain/mri/mri_to_blk_aligned.nii.gz", "histological", "mri"),
    "bigbrain_mni_200": ("bigbrain/icbm2009b_sym/full8_200um_2009b_sym.nii.gz", "icbm2009b_sym", "histology"),
    "bigbrain_mni_400": ("bigbrain/icbm2009b_sym/full8_400um_2009b_sym.nii.gz", "icbm2009b_sym", "histology"),
    "icbm_t1": ("mri/icbm2009b_sym/mni_icbm152_t1_tal_nlin_sym_09b_hires.nii.gz", "icbm2009b_sym", "mri"),
}


@dataclass
class Volume:
    name: str
    data: torch.Tensor          # (X, Y, Z) float32 in [0, 1], RAS voxel order, tissue bright / background 0
    affine: np.ndarray          # voxel -> mm
    space: str
    modality: str

    @property
    def shape(self):
        return tuple(self.data.shape)

    @property
    def bounds_mm(self):
        c = np.array([[0, 0, 0], [s - 1 for s in self.shape]], float)
        w = nib.affines.apply_affine(self.affine, c)
        return w.min(0), w.max(0)

    def to(self, device):
        self.data = self.data.to(device)
        return self


def load(name, device="cpu", robust_pct=(0.5, 99.5), apply_offset=True):
    """Load a registered volume, reorient to RAS and rescale robustly to [0, 1].

    If <volume dir>/offset_mm.json exists (see scripts/02_fix_mri_offset.py) its translation is added to the affine.
    """
    rel, space, modality = VOLUMES[name]
    img = nib.as_closest_canonical(nib.load(DATA / rel))
    affine = img.affine.copy()
    fix = (DATA / rel).parent / "offset_mm.json"
    if apply_offset and fix.exists():
        affine[:3, 3] += json.loads(fix.read_text())["translate_mm"]
    v = np.asarray(img.dataobj, dtype=np.float32)
    sub = v.ravel()[:: max(v.size // 2_000_000, 1)]                  # percentiles from a subsample: no full-size copies
    lo, hi = np.percentile(sub[sub > 0] if (sub > 0).any() else sub, robust_pct)
    v -= lo
    v /= max(hi - lo, 1e-6)
    np.clip(v, 0, 1, out=v)
    if modality == "histology":
        np.subtract(1, v, out=v)      # histology is dark tissue on white: flip so tissue is bright, background 0
    return Volume(name, torch.from_numpy(v).to(device), affine, space, modality)
