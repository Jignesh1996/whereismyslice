"""In-plane localisation: where does a histology section sit inside its (known) MRI slice?

Estimates the 2D affine T that maps MRI pixel (x = col, y = row) -> histology pixel, the same convention as
NiftyReg/SbR affine files, in three steps:
1. coarse: rotation x scale grid; each candidate warps the histology tissue mask into the MRI frame, and FFT
   cross-correlation with the MRI mask gives the best translation (score = soft mask Dice). Modality-free.
2. refine: batched gradient refinement of all 6 affine parameters of the best candidates, maximising
   mutual information of intensities inside the MRI mask plus soft mask Dice.
3. local (optional): B-spline on top (classical.local_align).
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import gaussian_blur, mutual_information


@dataclass
class InplaneResult:
    T: np.ndarray            # 3x3, MRI px -> histology px
    score: float
    coarse_T: np.ndarray


def _t(x, dev):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)


def warp(img, T, out_shape, disp=None):
    """Warp img (B or 1, H, W) into an out_shape frame: out(p) = img(T (p + disp(p))); T (B, 3, 3) on (x, y) pixels,
    disp (B, h, w, 2) optional displacement in output pixels (x, y)."""
    B = T.shape[0]
    H, W = img.shape[-2:]
    h, w = out_shape
    ys, xs = torch.meshgrid(torch.arange(h, device=T.device, dtype=torch.float32),
                            torch.arange(w, device=T.device, dtype=torch.float32), indexing="ij")
    p = torch.stack([xs, ys, torch.ones_like(xs)], -1).view(1, -1, 3)
    if disp is not None:
        p = p + F.pad(disp.reshape(B, -1, 2), (0, 1))
    q = (p @ T.transpose(1, 2))[..., :2]                                   # (B, h*w, 2) histology x, y
    grid = torch.stack([2 * q[..., 0] / (W - 1) - 1, 2 * q[..., 1] / (H - 1) - 1], -1).view(B, h, w, 2)
    src = img[:, None] if img.dim() == 3 else img[None, None]
    src = src.expand(B, -1, -1, -1) if src.shape[0] == 1 else src
    return F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0]


def similarity(theta_deg, scale, c_mri, c_histo, shift=(0.0, 0.0)):
    """T for q = c_histo + scale * R(theta) (p - c_mri - shift)."""
    th = np.deg2rad(theta_deg)
    A = scale * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    T = np.eye(3)
    T[:2, :2] = A
    T[:2, 2] = np.asarray(c_histo) - A @ (np.asarray(c_mri) + np.asarray(shift))
    return T


def centroid(mask):
    r, c = np.nonzero(mask)
    return np.array([c.mean(), r.mean()])


def coarse(histo_mask, mri_mask, angles=np.arange(-90, 91, 3), scales=np.exp(np.linspace(np.log(0.5), np.log(2.0), 29)),
           top_k=12, device="cuda", chunk=128, sigma=1.0, down=2):
    """Grid over rotation x scale; FFT gives the translation. Returns [(score, T)] best first, diverse in angle/scale.

    Runs on masks box-downsampled by `down`; the returned T are in full-resolution pixels.
    """
    pool = lambda m: F.avg_pool2d(_t(m, device)[None, None], down, ceil_mode=True)[0, 0] if down > 1 else _t(m, device)
    hm = gaussian_blur(pool(histo_mask)[None], sigma)[0]
    mm = gaussian_blur(pool(mri_mask)[None], sigma)[0]
    histo_mask, mri_mask = hm.cpu().numpy() > 0.5, mm.cpu().numpy() > 0.5
    h, w = mri_mask.shape
    H2, W2 = 2 * h, 2 * w
    c_m, c_h = centroid(mri_mask), centroid(histo_mask)
    c_frame = np.array([(W2 - 1) / 2, (H2 - 1) / 2])                      # warp histology centred in a padded frame
    fm = torch.fft.rfft2(F.pad(mm, (0, W2 - w, 0, H2 - h)))
    norm_m = mm.square().sum()
    cands = [(a, s) for a in angles for s in scales]
    out = []
    for i in range(0, len(cands), chunk):
        part = cands[i:i + chunk]
        T = _t(np.stack([similarity(a, s, c_frame, c_h) for a, s in part]), device)
        wh = warp(hm, T, (H2, W2))
        corr = torch.fft.irfft2(fm * torch.fft.rfft2(wh).conj(), s=(H2, W2))           # corr[d] = sum mri(p) wh(p - d)
        dice = 2 * corr / (norm_m + wh.square().sum((1, 2))[:, None, None])
        val, idx = dice.flatten(1).max(1)
        for j, (a, s) in enumerate(part):
            dy, dx = divmod(int(idx[j]), W2)
            dy = dy - H2 if dy > H2 // 2 else dy
            dx = dx - W2 if dx > W2 // 2 else dx
            # wh(p - d) places the frame centre at c_frame + d; express directly in MRI pixel coordinates
            out.append((float(val[j]), a, s, similarity(a, s, c_frame + np.array([dx, dy]), c_h)))
    out.sort(key=lambda o: -o[0])
    picked = []
    for o in out:
        if len(picked) == top_k:
            break
        if all(abs(o[1] - p[1]) >= 9 or abs(np.log(o[2] / p[2])) >= 0.12 for p in picked):
            picked.append(o)
    S = np.diag([down, down, 1.0])
    S[:2, 2] = (down - 1) / 2                                               # full px = down * coarse px + (down - 1) / 2
    return [(o[0], S @ o[3] @ np.linalg.inv(S)) for o in picked]


def refine(histo, histo_mask, mri, mri_mask, inits, iters=(100, 100), sigmas=(3.0, 1.0), lr=(0.02, 0.01),
           w_dice=0.5, keep=4, device="cuda"):
    """Batched gradient refinement of 6-parameter affines. inits: list of 3x3. Returns [(score, T)] best first.
    After the first (blurred) level only the best `keep` candidates continue."""
    hi, hmk = _t(histo, device), _t(histo_mask, device)
    mi, mmk = _t(mri, device), _t(mri_mask, device)
    shape = mri.shape
    T0 = _t(np.stack(inits), device)
    # parameterise as T = T0 @ (I + D), D = [[a b tx] [c d ty]] with translations in units of 10 px
    z = torch.zeros(len(inits), 6, device=device, requires_grad=True)
    roi = (gaussian_blur(mmk[None], 4.0)[0] > 0.05).float()[None]
    score = None
    for level, (n_it, sg, step) in enumerate(zip(iters, sigmas, lr)):
        if level == 1 and len(T0) > keep:
            top = torch.argsort(score.detach(), descending=True)[:keep]
            T0, z = T0[top], z.detach()[top].clone().requires_grad_(True)
        n = len(T0)
        hs = gaussian_blur(hi[None], sg)[0]
        ms = gaussian_blur(mi[None], sg)[0].expand(n, -1, -1)
        hms = gaussian_blur(hmk[None], sg)[0]
        mms = gaussian_blur(mmk[None], sg)[0][None]
        opt = torch.optim.Adam([z], lr=step)
        for _ in range(n_it):
            D = torch.zeros(n, 3, 3, device=device)
            D[:, 0, 0], D[:, 0, 1], D[:, 0, 2] = 1 + z[:, 0], z[:, 1], 10 * z[:, 4]
            D[:, 1, 0], D[:, 1, 1], D[:, 1, 2] = z[:, 2], 1 + z[:, 3], 10 * z[:, 5]
            D[:, 2, 2] = 1
            T = T0 @ D
            wi = warp(hs, T, shape)
            wm = warp(hms, T, shape)
            mi_val = mutual_information(wi.clamp(0, 1), ms.clamp(0, 1), mask=roi)
            dice = 2 * (wm * mms).sum((1, 2)) / (wm.square().sum((1, 2)) + mms.square().sum((1, 2)) + 1e-6)
            score = mi_val + w_dice * dice
            opt.zero_grad()
            (-score.sum()).backward()
            opt.step()
    T = T.detach().cpu().numpy()
    s = score.detach().cpu().numpy()
    order = np.argsort(-s)
    return [(float(s[i]), T[i]) for i in order]


def localise(histo, histo_mask, mri, mri_mask, device="cuda", top_k=12, **coarse_kw):
    c = coarse(histo_mask, mri_mask, top_k=top_k, device=device, **coarse_kw)
    r = refine(histo, histo_mask, mri, mri_mask, [T for _, T in c], device=device)
    return InplaneResult(r[0][1], r[0][0], c[0][1])


def missing_tissue(T, histo_mask, mri_mask, open_px=4, min_frac=0.005):
    """MRI tissue that the aligned histology does not cover in solid blobs (cut or torn-off tissue, or tissue never
    on the section): uncovered pixels, opened to drop boundary slivers, components >= min_frac of the MRI mask."""
    from scipy import ndimage as ndi
    moved = warp(_t(histo_mask, "cpu"), _t(T[None], "cpu"), mri_mask.shape)[0].numpy() > 0.5
    gap = ndi.binary_opening(mri_mask & ~moved, structure=np.ones((3, 3)), iterations=open_px)
    lab, n = ndi.label(gap)
    if n == 0:
        return gap
    sizes = ndi.sum(gap, lab, range(1, n + 1))
    big = np.isin(lab, 1 + np.nonzero(sizes >= min_frac * mri_mask.sum())[0])
    return ndi.binary_dilation(big, iterations=2) & mri_mask


def localise_cut_aware(histo, histo_mask, mri, mri_mask, device="cuda", res=None, **kw):
    """localise (or take its result `res`), then drop missing MRI tissue from the MRI mask and refine the affine again
    from that answer, so mask Dice no longer pulls the section towards tissue it lost.
    Returns (InplaneResult, effective MRI mask)."""
    res = res or localise(histo, histo_mask, mri, mri_mask, device=device, **kw)
    gone = missing_tissue(res.T, histo_mask, mri_mask)
    if not gone.any():
        return res, mri_mask
    mm = mri_mask & ~gone
    r = refine(histo, histo_mask, mri * mm, mm, [res.T], device=device)
    return InplaneResult(r[0][1], r[0][0], res.coarse_T), mm


def deform(histo, histo_mask, mri, mri_mask, T, grids=(6, 12), iters=(150, 150), sigmas=(2.0, 1.0), lr=0.5,
           w_dice=1.0, lam=2.0, gate=0.01, device="cuda"):
    """Regularised deformable refinement on top of affine T (MRI px -> histology px).

    A displacement field on a coarse control grid (in MRI px), upsampled bicubically, coarse to fine; maximises
    MI (inside the MRI mask) + w_dice * mask Dice - lam * mean squared strain of the control grid. The field is kept
    only if the data term improves by more than `gate` over the affine (no drift when there is nothing to fix).
    Returns (disp (H, W, 2) in MRI px (x, y), data-term gain)."""
    hi, hmk = _t(histo, device), _t(histo_mask, device)
    mi, mmk = _t(mri, device), _t(mri_mask, device)
    Tt = _t(T[None], device)
    H, W = mri.shape
    roi = (gaussian_blur(mmk[None], 4.0)[0] > 0.05).float()[None]

    def data(disp, sg):
        hs, hms = gaussian_blur(hi[None], sg)[0], gaussian_blur(hmk[None], sg)[0]
        ms, mms = gaussian_blur(mi[None], sg), gaussian_blur(mmk[None], sg)
        wi, wm = warp(hs, Tt, (H, W), disp), warp(hms, Tt, (H, W), disp)
        dice = 2 * (wm * mms).sum((1, 2)) / (wm.square().sum((1, 2)) + mms.square().sum((1, 2)) + 1e-6)
        return mutual_information(wi.clamp(0, 1), ms.clamp(0, 1), mask=roi)[0] + w_dice * dice[0]

    with torch.no_grad():
        base = float(data(None, sigmas[-1]))
    c = None
    for g, n_it, sg in zip(grids, iters, sigmas):
        gh, gw = max(2, round(g * H / max(H, W))), max(2, round(g * W / max(H, W)))
        c = torch.zeros(1, 2, gh, gw, device=device) if c is None else \
            F.interpolate(c.detach(), size=(gh, gw), mode="bilinear", align_corners=True)
        c.requires_grad_(True)
        opt = torch.optim.Adam([c], lr=lr)
        spacing = max(H / (gh - 1), W / (gw - 1))
        for _ in range(n_it):
            disp = F.interpolate(c, size=(H, W), mode="bicubic", align_corners=True).permute(0, 2, 3, 1)
            strain = ((c[..., 1:] - c[..., :-1]).square().mean() + (c[..., 1:, :] - c[..., :-1, :]).square().mean()) / spacing ** 2
            loss = -data(disp, sg) + lam * 100 * strain
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        disp = F.interpolate(c, size=(H, W), mode="bicubic", align_corners=True).permute(0, 2, 3, 1)
        gain = float(data(disp, sigmas[-1])) - base
    if gain < gate:
        return np.zeros((H, W, 2), np.float32), gain
    return disp[0].cpu().numpy(), gain


def mapping_error(T_est, T_gt, mask):
    """Per-pixel distance (MRI px) over mask between where GT and estimate put the same histology point:
    || T_est^-1 (T_gt p) - p ||."""
    r, c = np.nonzero(mask)
    p = np.stack([c, r, np.ones_like(c)]).astype(float)
    q = np.linalg.inv(T_est) @ (T_gt @ p)
    return np.linalg.norm(q[:2] - p[:2], axis=0)


def mask_dice(T, histo_mask, mri_mask):
    w = warp(_t(histo_mask, "cpu"), _t(T[None], "cpu"), mri_mask.shape)[0].numpy() > 0.5
    m = mri_mask > 0
    return 2 * (w & m).sum() / max(w.sum() + m.sum(), 1)
