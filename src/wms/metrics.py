"""Batched, differentiable image similarities for (N, H, W) tensors."""
import torch
import torch.nn.functional as F


def ncc(a, b, mask=None, eps=1e-6):
    """Normalised cross-correlation per image; mask (N or 1, H, W) restricts the pixels used."""
    if mask is None:
        mask = torch.ones_like(a)
    m = mask.expand_as(a).float()
    n = m.sum((1, 2)).clamp_min(1)
    am = (a * m).sum((1, 2)) / n
    bm = (b * m).sum((1, 2)) / n
    da, db = (a - am[:, None, None]) * m, (b - bm[:, None, None]) * m
    return (da * db).sum((1, 2)) / ((da.square().sum((1, 2)) * db.square().sum((1, 2))).sqrt() + eps)


def mutual_information(a, b, mask=None, bins=24, sigma=None, normalised=True):
    """Parzen-window (soft histogram) mutual information per image, for cross-modal matching.

    a, b are expected in [0, 1]. normalised=True returns (H(a) + H(b)) / H(a, b) in [1, 2].
    """
    sigma = sigma or 1.0 / bins
    centres = torch.linspace(0, 1, bins, device=a.device)
    N = a.shape[0]
    a, b = a.reshape(N, -1, 1), b.reshape(N, -1, 1)
    m = torch.ones_like(a) if mask is None else mask.expand(N, *mask.shape[1:]).reshape(N, -1, 1).float()
    wa = torch.exp(-0.5 * ((a - centres) / sigma) ** 2) * m
    wb = torch.exp(-0.5 * ((b - centres) / sigma) ** 2)
    wa = wa / (wa.sum(-1, keepdim=True) + 1e-8) * m
    wb = wb / (wb.sum(-1, keepdim=True) + 1e-8)
    joint = wa.transpose(1, 2) @ wb
    joint = joint / joint.sum((1, 2), keepdim=True).clamp_min(1e-8)
    pa, pb = joint.sum(2), joint.sum(1)
    ent = lambda p: -(p * (p + 1e-10).log()).flatten(1).sum(1)
    ha, hb, hab = ent(pa), ent(pb), ent(joint)
    return (ha + hb) / hab if normalised else ha + hb - hab


def gaussian_blur(x, sigma_px):
    """Separable Gaussian blur of (N, H, W)."""
    if sigma_px <= 0:
        return x
    r = int(3 * sigma_px + 0.5)
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma_px) ** 2)
    k = k / k.sum()
    y = F.conv2d(F.pad(x[:, None], (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    return y[:, 0]


def resize(x, shape):
    """Area-average / bilinear resize of (N, H, W) or (H, W)."""
    squeeze = x.dim() == 2
    x = x[None] if squeeze else x
    mode = "area" if shape[0] < x.shape[1] else "bilinear"
    y = F.interpolate(x[:, None], size=tuple(shape), mode=mode, **({} if mode == "area" else {"align_corners": False}))[:, 0]
    return y[0] if squeeze else y
