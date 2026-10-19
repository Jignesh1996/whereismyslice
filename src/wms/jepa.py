"""JEPA-style patch localisation: a small histology patch predicts the latent MRI features at its location.

* Context branch: ViT (DINOv2 init, shared weights) on the histology patch -> pooled tokens -> predictor -> z.
* Target branch: the same ViT on the full MRI plane -> dense token map -> per-token head -> F (h x w x D).
  F is pooled over windows the size of the patch footprint, so each position describes one candidate area.
* Losses: (1) JEPA regression: z must match the stop-gradient target features at the true footprint;
  (2) localisation InfoNCE over all positions of the map (soft target around the true position), which keeps the
  latent space discriminative and prevents collapse (used instead of an EMA target).
At test time the similarity map of z against F gives the location; rotation and fine position come from the
classical patch search in a small window around the best positions.
"""
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
TARGET_PX_MM = 0.8               # MRI plane rendered at 0.8 mm/px, 224 px -> 179 mm field of view, 16 x 16 tokens
TARGET_SIZE = 224
PATCH_SIZE = 112                 # histology patch input (8 x 8 tokens)
TOKEN_MM = TARGET_PX_MM * 14


def _prep(x):
    x = x[:, None].repeat(1, 3, 1, 1)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


class PatchJEPA(nn.Module):
    def __init__(self, arch="vit_small_patch14_dinov2.lvd142m", dim=256, train_blocks=4, pretrained=True):
        super().__init__()
        self.arch = arch
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0, img_size=224, dynamic_img_size=True)
        for p in self.backbone.parameters():
            p.requires_grad = False
        for blk in self.backbone.blocks[-train_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.backbone.norm.parameters():
            p.requires_grad = True
        d = self.backbone.num_features
        self.token_head = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, dim))
        self.predictor = nn.Sequential(nn.Linear(2 * d, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, dim))
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def context(self, patches):
        """patches (B, 112, 112) -> z (B, D), unit norm."""
        tok = self.backbone.forward_features(_prep(patches))
        n = self.backbone.num_prefix_tokens
        return F.normalize(self.predictor(torch.cat([tok[:, 0], tok[:, n:].mean(1)], 1)), dim=1)

    def target(self, planes):
        """planes (B, 224, 224) -> dense map (B, D, 16, 16), not normalised (normalise after window pooling)."""
        tok = self.backbone.forward_features(_prep(planes))[:, self.backbone.num_prefix_tokens:]
        B, N, _ = tok.shape
        g = int(round(N ** 0.5))
        return self.token_head(tok).transpose(1, 2).reshape(B, -1, g, g)


def window_maps(Fd, ks=(1, 2, 3, 4, 5)):
    """Dense map (B, D, g, g) -> {k: unit-norm features averaged over k x k token windows, same g x g grid}."""
    out = {}
    for k in ks:
        if k == 1:
            m = Fd
        else:
            m = F.avg_pool2d(F.pad(Fd, ((k - 1) // 2, k // 2, (k - 1) // 2, k // 2), mode="replicate"), k, stride=1)
        out[k] = F.normalize(m, dim=1)
    return out


def footprint_tokens(size_mm):
    return int(np.clip(np.round(size_mm / TOKEN_MM), 1, 5))


def losses(model, z, maps, k, pos, sigma=0.6):
    """z (B, D); maps {k: (B, D, g, g)}; k (B,) window sizes; pos (B, 2) continuous token (col, row) of the patch centre."""
    B = z.shape[0]
    g = maps[1].shape[-1]
    sims = torch.stack([torch.einsum("d,dhw->hw", z[i], maps[int(k[i])][i]) for i in range(B)])      # (B, g, g)
    yy, xx = torch.meshgrid(torch.arange(g, device=z.device), torch.arange(g, device=z.device), indexing="ij")
    d2 = (xx[None] - pos[:, 0, None, None]) ** 2 + (yy[None] - pos[:, 1, None, None]) ** 2
    target = torch.softmax(-d2.flatten(1) / (2 * sigma ** 2), 1)
    logits = model.logit_scale.exp().clamp(max=100) * sims.flatten(1)
    loss_loc = -(target * F.log_softmax(logits, 1)).sum(1).mean()
    # JEPA regression to the (stop-gradient) target feature at the true position
    ci = pos.round().long().clamp(0, g - 1)
    tgt = torch.stack([maps[int(k[i])][i, :, ci[i, 1], ci[i, 0]] for i in range(B)]).detach()
    loss_jepa = (2 - 2 * (z * tgt).sum(1)).mean()
    with torch.no_grad():
        best = sims.flatten(1).argmax(1)
        err_tok = torch.hypot((best % g).float() - pos[:, 0], (best // g).float() - pos[:, 1])
    return loss_loc, loss_jepa, err_tok * TOKEN_MM
