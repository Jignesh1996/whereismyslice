import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from wms.patches import Patch, _rot_scale, classical_patch_search


def test_rot_scale_maps_centre_and_zoom():
    img = torch.zeros(1, 21, 21)
    img[0, 10, 15] = 1.0                                   # 5 px right of centre
    out = _rot_scale(img, 90.0, 2.0, 61)[0]
    r, c = np.unravel_index(int(out.argmax()), out.shape)
    # rotated by +90 deg (x right -> y down in image rows) and zoomed x2: 10 px from centre
    assert abs(np.hypot(r - 30, c - 30) - 10) <= 1.5


def test_patch_search_recovers_known_position_rotation_scale():
    rng = np.random.default_rng(0)
    big = gaussian_filter(rng.random((200, 220)), 3).astype(np.float32)
    big = (big - big.min()) / np.ptp(big)
    cy, cx, half = 90, 120, 20
    crop = torch.tensor(big[cy - half:cy + half + 1, cx - half:cx + half + 1])[None]
    # present the patch rotated by -40 deg and at 1/1.1 scale: the search must undo both
    patch_img = _rot_scale(crop, -40.0, 1 / 1.1, 41)[0].numpy()
    valid = _rot_scale(torch.ones_like(crop), -40.0, 1 / 1.1, 41)[0].numpy() > 0.99
    p = Patch(patch_img, valid, None, px_mm=1.1, sid=-1)
    best = classical_patch_search(p, big, mri_px_mm=1.0, angles=np.arange(0, 360, 10), scales=(1.0,), device="cpu")[0]
    _, x, y, a, s = best
    assert np.hypot(x - cx, y - cy) <= 3
    assert min(abs(a - 40) % 360, 360 - abs(a - 40) % 360) <= 10
