"""Histology <-> MRI dual encoder: shared DINOv2 backbone (last blocks fine-tuned), one projection head per
modality, soft-target InfoNCE (planes closer than ~1 mm count as partial positives), and an auxiliary
pose-regression head on the histology branch."""
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .embed import dino_features
from .planes import rotation


def mlp(d_in, d_out=256, hidden=512):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out))


class DualEncoder(nn.Module):
    def __init__(self, arch="vit_small_patch14_dinov2.lvd142m", train_blocks=4, dim=256, pretrained=True):
        super().__init__()
        self.arch = arch
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0, img_size=224)
        for p in self.backbone.parameters():
            p.requires_grad = False
        for blk in self.backbone.blocks[-train_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.backbone.norm.parameters():
            p.requires_grad = True
        d = 2 * self.backbone.num_features
        self.heads = nn.ModuleDict({"histology": mlp(d, dim), "mri": mlp(d, dim)})
        self.pose_head = nn.Linear(dim, 3)                      # y / 75 mm, pitch / 20 deg, yaw / 20 deg
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def encode(self, x, modality):
        return F.normalize(self.heads[modality](dino_features(self.backbone, x)), dim=1)


def plane_distance(params):
    """Pairwise distance between section planes (N, 7) -> (N, N): offset of j's centre from i's plane (mm)
    plus normal angle / 5 deg (so 5 deg of tilt counts like 1 mm)."""
    n = rotation(params[:, 3:6])[:, :, 1]
    c = params[:, :3]
    off = ((c[None, :, :] - c[:, None, :]) * n[:, None, :]).sum(-1).abs()
    ang = torch.rad2deg(torch.arccos((n @ n.T).abs().clamp(max=1)))
    return off + ang / 5


def soft_infonce(zh, zm, params, logit_scale, sigma_mm=1.0):
    logits = logit_scale.exp().clamp(max=100) * zh @ zm.T
    t = torch.exp(-(plane_distance(params) / sigma_mm) ** 2)
    t = t / t.sum(1, keepdim=True)
    return 0.5 * (torch.sum(-t * F.log_softmax(logits, 1), 1).mean() + torch.sum(-t.T * F.log_softmax(logits.T, 1), 1).mean())
