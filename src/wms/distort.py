"""Synthetic section damage with exact ground truth: rotation, affine stretch/shear, smooth elastic deformation,
missing chunks and tears.

A distortion is stored as its backward map G: distorted pixel (x, y) -> original pixel, so the distorted image is
Hd(x) = H(G(x)) and the true location of any distorted point in the original section is known exactly.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


@dataclass
class Distortion:
    shape: tuple                     # distorted canvas (H, W)
    A: np.ndarray                    # 2x2 linear part (distorted -> original, about the centres)
    c_dist: np.ndarray               # canvas centre (x, y)
    c_orig: np.ndarray               # original image centre (x, y)
    u: np.ndarray                    # (2, H, W) extra displacement (x, y) in original px: elastic + tears
    cut: np.ndarray                  # (H, W) bool, tissue removed in the distorted section
    params: dict = field(default_factory=dict)

    def G(self, x, y):
        """Distorted pixel coordinates -> original pixel coordinates (arrays of any shape)."""
        dx, dy = x - self.c_dist[0], y - self.c_dist[1]
        ox = self.A[0, 0] * dx + self.A[0, 1] * dy + self.c_orig[0]
        oy = self.A[1, 0] * dx + self.A[1, 1] * dy + self.c_orig[1]
        coords = np.stack([y, x])
        ox = ox + map_coordinates(self.u[0], coords, order=1, mode="nearest")
        oy = oy + map_coordinates(self.u[1], coords, order=1, mode="nearest")
        return ox, oy


def random_distortion(img_shape, tissue_mask, rng, rotation=True, affine=0.0, elastic=0.0, cuts=False):
    """affine, elastic: fractional severity (e.g. 0.05-0.10). elastic = max displacement / tissue extent."""
    H, W = img_shape
    r, c = np.nonzero(tissue_mask)
    extent = max(np.ptp(r), np.ptp(c))
    side = int(np.ceil(np.hypot(H, W) * (1 + affine))) + 8
    shape = (side, side)
    theta = rng.uniform(-np.pi, np.pi) if rotation else 0.0
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    S = np.eye(2)
    if affine > 0:
        s = rng.uniform(-affine, affine, 2)                       # anisotropic stretch
        sh = rng.uniform(-affine, affine)                         # shear
        S = np.array([[1 + s[0], sh], [0, 1 + s[1]]])
    A = R @ S
    c_dist = np.array([(side - 1) / 2, (side - 1) / 2])
    c_orig = np.array([c.mean(), r.mean()])                       # rotate about the tissue centroid
    u = np.zeros((2, side, side))
    if elastic > 0:
        for k in range(2):
            f = gaussian_filter(rng.standard_normal((side, side)), sigma=side / 8)
            u[k] = f / (np.abs(f).max() + 1e-9)
        u *= elastic * extent / max(np.linalg.norm(u, axis=0).max(), 1e-9)
    cut = np.zeros(shape, bool)
    info = dict(rotation_deg=np.degrees(theta), affine=affine, elastic=elastic, cuts=cuts)
    if cuts:
        yy, xx = np.mgrid[0:side, 0:side].astype(float)
        # a tear: tissue on one side of a line through the tissue is pulled apart by gap px, gap band is empty
        phi = rng.uniform(0, np.pi)
        n = np.array([np.cos(phi), np.sin(phi)])
        d = (xx - c_dist[0]) * n[0] + (yy - c_dist[1]) * n[1] - rng.uniform(-0.2, 0.2) * extent
        gap = rng.uniform(0.01, 0.03) * extent
        side_pos = d > 0
        u[0] -= side_pos * gap * n[0]                            # original point = distorted point - gap on that side
        u[1] -= side_pos * gap * n[1]
        cut |= (d > 0) & (d < gap)
        # a missing chunk near the tissue edge, 5-15 % of the tissue area
        ang = rng.uniform(0, 2 * np.pi)
        centre = c_dist + 0.45 * extent * np.array([np.cos(ang), np.sin(ang)])
        rad = np.sqrt(rng.uniform(0.05, 0.15) * tissue_mask.sum() / np.pi)
        cut |= (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2 < rad ** 2
        info.update(tear_gap_px=gap, chunk_radius_px=rad)
    return Distortion(shape, A, c_dist, c_orig, u, cut, info)


def apply(dist, img, mask):
    """Distorted image and mask (cut regions removed from both, as a real tissue mask would be)."""
    yy, xx = np.mgrid[0:dist.shape[0], 0:dist.shape[1]].astype(float)
    ox, oy = dist.G(xx, yy)
    im = map_coordinates(img, [oy, ox], order=1, cval=0.0)
    mk = map_coordinates(mask.astype(float), [oy, ox], order=1, cval=0.0) > 0.5
    im[dist.cut] = 0
    mk &= ~dist.cut
    return im * mk, mk
