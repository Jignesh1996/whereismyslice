"""Oblique plane poses and differentiable sampling of 2D sections from 3D volumes (torch, CPU or GPU).

A pose is a section plane in world mm, stored as 7 numbers:
    x, y, z            plane centre (mm), i.e. where the centre pixel of the section lands
    pitch, yaw, roll   degrees; R = Rz(yaw) @ Rx(pitch) @ Ry(roll)
    log_s              log of the isotropic scale (histology shrinks a few % relative to the volume)
With all angles zero the plane is the coronal slice y = const (BigBrain cutting direction):
image columns run to +x (right), image rows run from superior (row 0) to inferior, normal = +y (anterior).
roll is the in-plane rotation, pitch tilts the top of the section anterior/posterior, yaw is a left/right
obliquity. The normal does not depend on roll.
"""
from dataclasses import dataclass, astuple

import numpy as np
import torch
import torch.nn.functional as F

NPARAM = 7


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    log_s: float = 0.0

    @property
    def centre(self):
        return np.array([self.x, self.y, self.z])

    @property
    def normal(self):
        return rotation(torch.tensor(self.as_array()[None, 3:6]))[0, :, 1].numpy()

    def as_array(self):
        return np.array(astuple(self), float)

    @classmethod
    def from_array(cls, a):
        a = np.asarray(a, float).ravel()
        return cls(*map(float, a[:NPARAM]), *([0.0] * (NPARAM - len(a))))

    def __str__(self):
        return (f"Pose(x={self.x:.1f}, y={self.y:.2f}, z={self.z:.1f} mm, pitch={self.pitch:.1f}, "
                f"yaw={self.yaw:.1f}, roll={self.roll:.1f} deg, s={np.exp(self.log_s):.3f})")


def as_params(poses, device="cpu"):
    """Pose | list[Pose] | array (N, <=7) -> float32 tensor (N, 7)."""
    if isinstance(poses, torch.Tensor):
        p = poses
    else:
        if isinstance(poses, Pose):
            poses = [poses]
        arr = np.stack([q.as_array() if isinstance(q, Pose) else np.asarray(q, float) for q in poses])
        p = torch.as_tensor(arr, dtype=torch.float32)
    if p.shape[-1] < NPARAM:
        p = torch.cat([p, p.new_zeros(*p.shape[:-1], NPARAM - p.shape[-1])], -1)
    return p.to(device)


def rotation(angles):
    """angles (N, 3) = pitch, yaw, roll in degrees -> (N, 3, 3) with columns = world directions of (x, y, z)."""
    p, w, r = torch.deg2rad(angles).unbind(-1)
    one, zero = torch.ones_like(p), torch.zeros_like(p)
    rx = torch.stack([one, zero, zero, zero, p.cos(), -p.sin(), zero, p.sin(), p.cos()], -1).view(-1, 3, 3)
    ry = torch.stack([r.cos(), zero, r.sin(), zero, one, zero, -r.sin(), zero, r.cos()], -1).view(-1, 3, 3)
    rz = torch.stack([w.cos(), -w.sin(), zero, w.sin(), w.cos(), zero, zero, zero, one], -1).view(-1, 3, 3)
    return rz @ rx @ ry


def plane_points(params, shape, px_mm):
    """World mm of every pixel centre: params (N, 7), shape (H, W), px_mm float or (row, col) -> (N, H, W, 3)."""
    H, W = shape
    pr, pc = (px_mm, px_mm) if np.isscalar(px_mm) else px_mm
    R = rotation(params[:, 3:6])
    s = params[:, 6].exp()[:, None, None, None]
    dev = params.device
    cu = (torch.arange(W, device=dev, dtype=torch.float32) - (W - 1) / 2) * pc
    cv = ((H - 1) / 2 - torch.arange(H, device=dev, dtype=torch.float32)) * pr
    u, v = R[:, None, None, :, 0], R[:, None, None, :, 2]                     # in-plane right, up
    pts = s * (cu[None, None, :, None] * u + cv[None, :, None, None] * v)
    return pts + params[:, None, None, :3]


def sample(volume, poses, shape=(300, 350), px_mm=0.4, mode="bilinear"):
    """Render planes from a Volume -> (N, H, W) on the volume's device; zero outside the volume.

    Differentiable with respect to the pose parameters when they are passed as a tensor with grad.
    """
    dev = volume.data.device
    params = as_params(poses, dev)
    pts = plane_points(params, shape, px_mm)
    inv = torch.as_tensor(np.linalg.inv(volume.affine), dtype=torch.float32, device=dev)
    vox = pts @ inv[:3, :3].T + inv[:3, 3]
    size = torch.tensor(volume.shape, dtype=torch.float32, device=dev)
    grid = (2 * vox / (size - 1) - 1).flip(-1)                                # grid_sample wants (k, j, i)
    vol = volume.data[None, None].expand(len(params), -1, -1, -1, -1)
    return F.grid_sample(vol, grid[:, None], mode=mode, padding_mode="zeros", align_corners=True)[:, 0, 0]


def random_poses(n, bounds_mm, max_tilt=(20.0, 20.0, 180.0), margin_mm=(10.0, 8.0, 10.0), rng=None):
    """Random section poses: centre uniform inside the shrunk bounds, pitch/yaw ~ N(0, max/2) clipped, roll uniform."""
    rng = np.random.default_rng(rng)
    lo, hi = np.asarray(bounds_mm[0]) + margin_mm, np.asarray(bounds_mm[1]) - margin_mm
    c = rng.uniform(lo, hi, size=(n, 3))
    t = np.asarray(max_tilt[:2])
    tilt = np.clip(rng.normal(0, t / 2, size=(n, 2)), -t, t)
    roll = rng.uniform(-max_tilt[2], max_tilt[2], size=n)
    return [Pose(*c[i], *tilt[i], roll[i]) for i in range(n)]


def pose_error(true, est):
    """(offset mm, tilt deg): distance from the true section centre to the estimated plane along its normal,
    and the angle between the plane normals. In-plane shift and roll are not counted (sections are cropped freely)."""
    nt, ne = true.normal, est.normal
    offset = abs(np.dot(true.centre - est.centre, ne))
    tilt = np.degrees(np.arccos(np.clip(abs(np.dot(nt, ne)), 0, 1)))
    return float(offset), float(tilt)
