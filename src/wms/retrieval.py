"""Slice bank: planes rendered from a reference volume over a pose grid, embedded and indexed with faiss.

Querying a section returns bank poses ranked by cosine similarity, each shifted in-plane so the query's
tissue centroid lands on the bank plane's tissue centroid. Those poses can go straight into
classical.refine (the hybrid: embeddings propose, intensity registration disposes).
"""
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import torch

from .classical import Match, to_isotropic
from .embed import canonical_views
from .planes import Pose, rotation, sample


@dataclass
class Bank:
    poses: np.ndarray            # (M, 7)
    centroids: np.ndarray        # (M, 2) tissue centroid offset of each rendered plane, mm (right, up)
    emb: np.ndarray              # (M, D) L2-normalised
    volume: str
    fov_mm: float

    def __post_init__(self):
        self.index = faiss.IndexFlatIP(self.emb.shape[1])
        self.index.add(np.ascontiguousarray(self.emb, dtype=np.float32))

    def save(self, path):
        np.savez_compressed(path, poses=self.poses, centroids=self.centroids, emb=self.emb, volume=self.volume, fov_mm=self.fov_mm)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["poses"], d["centroids"], d["emb"], str(d["volume"]), float(d["fov_mm"]))


def bank_poses(volume, y_step=1.0, angles=(-15, -10, -5, 0, 5, 10, 15)):
    lo, hi = volume.bounds_mm
    cx, cz = (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2
    ys = np.arange(lo[1] + 3, hi[1] - 3, y_step)
    return np.array([(cx, y, cz, p, w, 0.0, 0.0) for y in ys for p in angles for w in angles], np.float32)


def render_views(volume, poses, fov_mm=160.0, render_px=0.5, chunk=64):
    """Render planes large enough to hold the whole cross-section, then canonicalise -> (views, centroids)."""
    lo, hi = volume.bounds_mm
    n = int(np.max(hi - lo) * 1.15 / render_px)
    views, cents = [], []
    with torch.no_grad():
        for i in range(0, len(poses), chunk):
            planes = sample(volume, torch.as_tensor(poses[i:i + chunk], device=volume.data.device), (n, n), render_px)
            v, c = canonical_views(planes, render_px, fov_mm)
            views.append(v.cpu())
            cents.append(c.cpu())
    return torch.cat(views), torch.cat(cents).numpy()


def build_bank(volume, embedder, fov_mm=160.0, **grid_kw):
    poses = bank_poses(volume, **grid_kw)
    views, cents = render_views(volume, poses, fov_mm)
    return Bank(poses, cents, embedder(views), volume.name, fov_mm)


def query_views(img, px_mm, fov_mm, rolls=(0.0,), flips=(False,), work_px=0.5, device="cpu"):
    s = to_isotropic(img, px_mm, work_px, device)[None]
    out = []
    for f in flips:
        for r in rolls:
            v, c = canonical_views(s, work_px, fov_mm, roll_deg=r, flip=f)
            out.append((r, f, v[0], c[0].cpu().numpy()))
    return out


def query(bank, embedder, img, px_mm, k=10, rolls=(0.0,), flips=(False,)):
    """Top-k Matches for one section. Each test-time roll/flip is searched; the best per bank plane is kept."""
    qs = query_views(img, px_mm, bank.fov_mm, rolls, flips, device=embedder.device)
    emb = embedder(torch.stack([q[2] for q in qs]))
    sims, idx = bank.index.search(np.ascontiguousarray(emb, dtype=np.float32), k)
    hits = {}
    for qi, (roll, flip, _, qc) in enumerate(qs):
        for s, j in zip(sims[qi], idx[qi]):
            if j not in hits or s > hits[j][0]:
                hits[j] = (float(s), roll, flip, qc)
    out = []
    for j, (s, roll, flip, qc) in sorted(hits.items(), key=lambda kv: -kv[1][0])[:k]:
        p = bank.poses[j].astype(float).copy()
        R0 = rotation(torch.tensor(p[None, 3:6], dtype=torch.float32))[0].numpy()
        bc = bank.centroids[j]
        world_c = p[:3] + bc[0] * R0[:, 0] + bc[1] * R0[:, 2]           # bank plane's tissue centroid in mm
        p[5] = roll
        R = rotation(torch.tensor(p[None, 3:6], dtype=torch.float32))[0].numpy()
        qx, qz = (-qc[0] if flip else qc[0]), qc[1]                     # centroid of the (flipped) section
        p[:3] = world_c - (qx * R[:, 0] + qz * R[:, 2])
        out.append(Match(Pose.from_array(p), s, flip))
    return out
