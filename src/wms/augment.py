"""Histology section artefacts for synthetic queries: stain jitter, blur, noise, tears, missing chunks, folds.

Works on tissue-bright float images in [0, 1] (numpy, (H, W)). Every function takes a numpy Generator.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.draw import polygon


def stain_jitter(x, rng):
    gamma = rng.uniform(0.7, 1.4)
    gain = rng.uniform(0.75, 1.25)
    bias = rng.uniform(-0.05, 0.05)
    tissue = x > 0.05
    y = np.clip(gain * np.power(np.clip(x, 0, 1), gamma) + bias * tissue, 0, 1)
    ramp = 1 + rng.uniform(-0.15, 0.15) * np.linspace(-1, 1, x.shape[1])[None, :]   # uneven staining across the slide
    return np.clip(y * ramp, 0, 1)


def blur_noise(x, rng):
    y = gaussian_filter(x, rng.uniform(0, 1.2))
    return np.clip(y + rng.normal(0, rng.uniform(0, 0.04), x.shape) * (y > 0.03), 0, 1)


def tear(x, rng, max_gap_px=4):
    """A crack: tissue on one side of a random line is pulled apart by a few pixels."""
    H, W = x.shape
    c = np.array([rng.uniform(0.3, 0.7) * H, rng.uniform(0.3, 0.7) * W])
    th = rng.uniform(0, np.pi)
    n = np.array([np.cos(th), np.sin(th)])
    rr, cc = np.mgrid[0:H, 0:W]
    side = ((rr - c[0]) * n[0] + (cc - c[1]) * n[1]) > 0
    gap = rng.uniform(1, max_gap_px)
    coords = np.stack([rr - side * gap * n[0], cc - side * gap * n[1]])
    return map_coordinates(x, coords, order=1, cval=0.0)


def missing_chunk(x, rng, max_frac=0.12):
    """Remove a random polygon of tissue (lost piece at the section edge or a hole)."""
    H, W = x.shape
    r0, c0 = rng.uniform(0, H), rng.uniform(0, W)
    rad = np.sqrt(max_frac * H * W / np.pi) * rng.uniform(0.4, 1)
    ang = np.sort(rng.uniform(0, 2 * np.pi, 7))
    rr, cc = polygon(r0 + rad * rng.uniform(0.5, 1, 7) * np.sin(ang), c0 + rad * rng.uniform(0.5, 1, 7) * np.cos(ang), x.shape)
    y = x.copy()
    y[rr, cc] = 0
    return y


def fold(x, rng):
    """A fold: a strip of doubled tissue (darker stain = brighter here)."""
    H, W = x.shape
    th = rng.uniform(0, np.pi)
    rr, cc = np.mgrid[0:H, 0:W]
    d = (rr - rng.uniform(0.2, 0.8) * H) * np.cos(th) + (cc - rng.uniform(0.2, 0.8) * W) * np.sin(th)
    strip = np.abs(d) < rng.uniform(1, 4)
    y = x.copy()
    y[strip] = np.clip(y[strip] * 1.6, 0, 1)
    return y


def augment(x, rng, p_tear=0.4, p_chunk=0.4, p_fold=0.3):
    y = stain_jitter(x, rng)
    if rng.random() < p_tear:
        y = tear(y, rng)
    if rng.random() < p_chunk:
        y = missing_chunk(y, rng)
    if rng.random() < p_fold:
        y = fold(y, rng)
    return blur_noise(y, rng).astype(np.float32)


def augment_torch(x, gen=None):
    """GPU batch version for training: x (N, H, W) tissue-bright -> augmented copy.

    Stain gamma/gain and a staining ramp, scale jitter (shrinkage), tears (half-plane pulled apart),
    missing elliptical chunks, blur and noise.
    """
    import torch
    import torch.nn.functional as F
    N, H, W = x.shape
    dev = x.device
    rnd = lambda *s: torch.rand(*s, device=dev, generator=gen)
    gamma = 0.7 + 0.7 * rnd(N, 1, 1)
    gain = 0.75 + 0.5 * rnd(N, 1, 1)
    ramp = 1 + (rnd(N, 1, 1) - 0.5) * 0.3 * torch.linspace(-1, 1, W, device=dev)[None, None]
    y = (gain * x.clamp(0, 1) ** gamma * ramp).clamp(0, 1)
    # scale jitter +-8 % and tears via one sampling grid
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=dev), torch.linspace(-1, 1, W, device=dev), indexing="ij")
    s = 0.92 + 0.16 * rnd(N, 1, 1)
    gx, gy = xx[None] * s, yy[None] * s
    th = rnd(N, 1, 1) * 3.1416
    d = (gx - (rnd(N, 1, 1) - 0.5)) * th.cos() + (gy - (rnd(N, 1, 1) - 0.5)) * th.sin()
    gap = (rnd(N, 1, 1) < 0.4).float() * rnd(N, 1, 1) * 0.04
    gx = gx - (d > 0).float() * gap * th.cos()
    gy = gy - (d > 0).float() * gap * th.sin()
    y = F.grid_sample(y[:, None], torch.stack([gx, gy], -1), align_corners=True)[:, 0]
    # missing chunks
    cx, cy = rnd(N, 1, 1) * 2 - 1, rnd(N, 1, 1) * 2 - 1
    rx, ry = 0.05 + 0.25 * rnd(N, 1, 1), 0.05 + 0.25 * rnd(N, 1, 1)
    hole = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 < 1) & (rnd(N, 1, 1) < 0.4)
    y = y.masked_fill(hole, 0)
    # blur + noise
    k = torch.tensor([1, 4, 6, 4, 1], device=dev, dtype=y.dtype)
    k = (k[:, None] * k[None]) / 256
    blurred = F.conv2d(y[:, None], k[None, None], padding=2)[:, 0]
    a = rnd(N, 1, 1)
    y = a * blurred + (1 - a) * y
    y = y + torch.randn(N, H, W, device=dev, generator=gen) * 0.04 * rnd(N, 1, 1) * (y > 0.03)
    return y.clamp(0, 1)
