"""Classical section -> volume localisation: global pose search, gradient pose refinement, local deformable alignment.

1. coarse_search: render a grid of candidate planes (y, pitch, yaw, roll) as large canvases and slide the
   section over each with FFT normalised cross-correlation, which gives the in-plane offset for free.
   For histology -> MRI the coarse step compares smoothed tissue masks (shape of the slice outline and
   ventricles), which does not depend on stain or MR contrast.
2. refine: batched gradient ascent on all 7 pose parameters (x, y, z, pitch, yaw, roll, log-scale) through
   the differentiable plane sampler, coarse to fine, with NCC (same modality) or soft mutual information.
3. local_align: 2D B-spline registration of the section onto the plane rendered at the refined pose
   (SimpleITK, Mattes MI), for tears, shrinkage and folds that a rigid plane cannot explain.
"""
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from .metrics import gaussian_blur, mutual_information, ncc, resize
from .planes import NPARAM, Pose, as_params, plane_points, rotation, sample
from .volumes import Volume

TISSUE = 0.08                              # tissue/background threshold on the [0, 1] tissue-bright scale


@dataclass
class Match:
    pose: Pose
    score: float
    flipped: bool = False


def to_isotropic(img, px_mm, out_px, device):
    """(H, W) array at (row, col) px_mm -> tensor at isotropic out_px."""
    t = torch.as_tensor(np.ascontiguousarray(img), dtype=torch.float32, device=device)
    shape = (max(int(round(img.shape[0] * px_mm[0] / out_px)), 8), max(int(round(img.shape[1] * px_mm[1] / out_px)), 8))
    return resize(t, shape)


def tissue_mask(x, sigma_px=1.0):
    return gaussian_blur((x > TISSUE).float(), sigma_px)


def _volume_level(volume, px_mm, _cache={}):
    """Volume pre-smoothed for sampling at px_mm (box average over ~px_mm), cached."""
    vox = float(np.abs(np.diag(volume.affine)[:3]).mean())
    k = int(round(px_mm / vox)) | 1                         # odd box so the smoothing stays centred
    if k <= 1:
        return volume
    key = (id(volume.data), k)
    if key not in _cache:
        d = F.avg_pool3d(volume.data[None, None], k, stride=1, padding=k // 2, count_include_pad=False)[0, 0]
        _cache.clear()
        _cache[key] = Volume(volume.name, d, volume.affine, volume.space, volume.modality)
    return _cache[key]


def fft_ncc(canvas, template):
    """Valid-mode normalised cross-correlation of one template (h, w) over canvases (N, Hc, Wc)."""
    N, Hc, Wc = canvas.shape
    h, w = template.shape
    t = template - template.mean()
    tn = t.square().sum().sqrt()
    fc = torch.fft.rfft2(canvas)
    ft = torch.fft.rfft2(t, s=(Hc, Wc))
    num = torch.fft.irfft2(fc * ft.conj(), s=(Hc, Wc))[:, : Hc - h + 1, : Wc - w + 1]
    ii = F.pad(canvas, (1, 0, 1, 0)).cumsum(1).cumsum(2)
    ii2 = F.pad(canvas.square(), (1, 0, 1, 0)).cumsum(1).cumsum(2)
    box = lambda s: s[:, h:, w:] - s[:, :-h, w:] - s[:, h:, :-w] + s[:, :-h, :-w]
    s1, s2 = box(ii), box(ii2)
    var = (s2 - s1.square() / (h * w)).clamp_min(0)
    out = num / (tn * var.sqrt() + 1e-6)
    return out.masked_fill(var < 1e-3 * var.amax(), -1)       # windows of empty space cannot win


def coarse_search(img, px_mm, volume, y_step=1.5, angles=(-10, -5, 0, 5, 10), rolls=(-10, 0, 10),
                  work_px=2.0, feature="intensity", try_flip=False, top_k=24, min_sep_mm=3.0, chunk=192):
    """Grid search over plane position/angles; returns up to top_k Matches, at most one per min_sep_mm band of y
    (the best-scoring plane in a band wins), so refinement sees hypotheses spread through the brain."""
    dev = volume.data.device
    vol = _volume_level(volume, work_px)
    tmpl = to_isotropic(img, px_mm, work_px, dev)
    lo, hi = volume.bounds_mm
    span = float(np.max(hi - lo))
    Hc = int(span / work_px) + tmpl.shape[0]
    Wc = int(span / work_px) + tmpl.shape[1]
    cx, cz = (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2
    ys = np.arange(lo[1] + 2, hi[1] - 2, y_step)
    grid = np.array([(cx, y, cz, p, w, r, 0.0) for y in ys for p in angles for w in angles for r in rolls], np.float32)
    feat = (lambda x: tissue_mask(x, 1.0)) if feature == "mask" else (lambda x: x)
    templates = [(False, feat(tmpl[None])[0])] + ([(True, feat(tmpl.flip(1)[None])[0])] if try_flip else [])
    h, w = tmpl.shape

    best = []                                                   # (score, grid index, row, col, flipped)
    for i in range(0, len(grid), chunk):
        params = torch.as_tensor(grid[i:i + chunk], device=dev)
        canv = feat(sample(vol, params, (Hc, Wc), work_px))
        for flipped, t in templates:
            score = fft_ncc(canv, t)
            flat = score.flatten(1)
            val, idx = flat.max(1)
            r, c = idx // score.shape[2], idx % score.shape[2]
            best += [(float(val[j]), i + j, int(r[j]), int(c[j]), flipped) for j in range(len(params))]
    best.sort(key=lambda b: -b[0])

    out, taken = [], []
    for val, gi, r, c, flipped in best:
        if len(out) == top_k:
            break
        if any(abs(grid[gi][1] - y) < min_sep_mm for y in taken):
            continue
        taken.append(grid[gi][1])
        g = grid[gi].astype(float)
        R = rotation(torch.tensor(g[None, 3:6], dtype=torch.float32))[0].numpy()
        dc = (c + (w - 1) / 2 - (Wc - 1) / 2) * work_px
        dr = (r + (h - 1) / 2 - (Hc - 1) / 2) * work_px
        g[:3] = g[:3] + dc * R[:, 0] - dr * R[:, 2]
        out.append(Match(Pose.from_array(g), val, flipped))
    return out


def refine(img, px_mm, volume, init, metric="ncc", levels=((2.0, 80, 0.4), (1.0, 60, 0.2), (0.5, 40, 0.1)),
           fit_scale=True, keep=6):
    """Gradient refinement of a batch of initial poses (list[Match] or list[Pose]); returns Matches sorted best first.

    levels: (pixel size mm, iterations, Adam step) from coarse to fine. Steps act on scaled parameters
    (1 unit = 1 mm, 1 deg, or 0.01 in log-scale). After the first level only the best `keep` continue.
    """
    dev = volume.data.device
    matches = [m if isinstance(m, Match) else Match(m, float("nan")) for m in init]
    flips = [m.flipped for m in matches]
    p0 = as_params([m.pose for m in matches], dev)
    unit = torch.tensor([1, 1, 1, 1, 1, 1, 0.01], device=dev)
    z = torch.zeros_like(p0, requires_grad=True)
    sim = ncc if metric == "ncc" else mutual_information
    score = None
    for li, (px, iters, lr) in enumerate(levels):
        if li == 1 and len(flips) > keep:
            top = torch.argsort(score.detach(), descending=True)[:keep]
            p0, flips = p0[top], [flips[i] for i in top.tolist()]
            z = z.detach()[top].clone().requires_grad_(True)
        vol = _volume_level(volume, px)
        s = to_isotropic(img, px_mm, px, dev)
        tgt = torch.stack([s.flip(1) if f else s for f in flips])
        opt = torch.optim.Adam([z], lr=lr)
        for _ in range(iters):
            dz = z if fit_scale else z * torch.tensor([1] * 6 + [0], device=dev)
            rendered = sample(vol, p0 + dz * unit, tgt.shape[1:], px)
            score = sim(tgt, rendered)
            opt.zero_grad()
            (-score.sum()).backward()
            opt.step()
    with torch.no_grad():
        final = (p0 + (z if fit_scale else z * torch.tensor([1] * 6 + [0], device=dev)) * unit).cpu().numpy()
        score = score.detach().cpu().numpy()
    out = [Match(Pose.from_array(final[i]), float(score[i]), flips[i]) for i in range(len(flips))]
    return sorted(out, key=lambda m: -m.score)


def localise(img, px_mm, volume, metric="ncc", feature=None, coarse_kw=None, refine_kw=None):
    """Full classical pose estimate: coarse grid search -> batched refinement of the best candidates."""
    feature = feature or ("intensity" if metric == "ncc" else "mask")
    cands = coarse_search(img, px_mm, volume, feature=feature, **(coarse_kw or {}))
    return refine(img, px_mm, volume, cands, metric=metric, **(refine_kw or {})), cands


def render_like(img, px_mm, volume, match, out_px):
    """The volume plane at a pose, on the section's pixel grid resampled to out_px; returns (section, plane) arrays."""
    dev = volume.data.device
    s = to_isotropic(img, px_mm, out_px, dev)
    s = s.flip(1) if match.flipped else s
    with torch.no_grad():
        plane = sample(_volume_level(volume, out_px), [match.pose], s.shape, out_px)[0]
    return s.cpu().numpy(), plane.cpu().numpy()


def local_align(section, plane, px_mm, mesh_mm=12.0, metric="mi", iterations=100, fixed_mask=None, moving_mask=None):
    """2D B-spline registration moving=section onto fixed=plane (both isotropic, same grid). Optional tissue masks
    restrict the metric, so missing or torn tissue does not drag the deformation.

    Returns warped section, the dense displacement field (H, W, 2) in mm, and the final metric value.
    """
    fixed = sitk.GetImageFromArray(plane.astype(np.float32))
    moving = sitk.GetImageFromArray(section.astype(np.float32))
    for im in (fixed, moving):
        im.SetSpacing((px_mm, px_mm))
    mesh = [max(int(round(s * px_mm / mesh_mm)), 1) for s in fixed.GetSize()]
    tx = sitk.BSplineTransformInitializer(fixed, mesh, order=3)
    reg = sitk.ImageRegistrationMethod()
    if metric == "mi":
        reg.SetMetricAsMattesMutualInformation(32)
    else:
        reg.SetMetricAsCorrelation()
    for m, setter in ((fixed_mask, reg.SetMetricFixedMask), (moving_mask, reg.SetMetricMovingMask)):
        if m is not None:
            mk = sitk.GetImageFromArray(m.astype(np.uint8))
            mk.SetSpacing((px_mm, px_mm))
            setter(mk)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.25, seed=1)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=iterations,
                             maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=1000)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SetInitialTransform(tx, inPlace=True)
    reg.Execute(fixed, moving)
    warped = sitk.GetArrayFromImage(sitk.Resample(moving, fixed, tx, sitk.sitkLinear, 0.0))
    field = sitk.TransformToDisplacementField(tx, sitk.sitkVectorFloat64, fixed.GetSize(), fixed.GetOrigin(),
                                              fixed.GetSpacing(), fixed.GetDirection())
    return warped, sitk.GetArrayFromImage(field), reg.GetMetricValue()


def dice(a, b, thr=TISSUE):
    a, b = a > thr, b > thr
    return 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)
