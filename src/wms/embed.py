"""Image embeddings for retrieval: canonical views (fixed physical scale, tissue-centred) + DINOv2 features.

A canonical view is the section resampled to a fixed field of view (fov_mm) at out_px pixels, centred on the
tissue centroid, so a histology section and an MRI plane of the same anatomy look alike regardless of how
the slide was cropped. The centroid offset (mm, in the image frame) is returned so a retrieved plane pose can
be shifted to put the query's tissue where the bank plane's tissue is.
"""
import numpy as np
import timm
import torch
import torch.nn.functional as F

from .metrics import resize

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
TISSUE = 0.08


def canonical_views(imgs, px_mm, fov_mm=160.0, out_px=224, roll_deg=0.0, flip=False):
    """imgs (N, H, W) tensor, px_mm scalar (isotropic) -> views (N, out_px, out_px), centroid offsets (N, 2) mm
    as (right, up) from the image centre."""
    N, H, W = imgs.shape
    dev = imgs.device
    m = (imgs > TISSUE).float()
    rr = torch.arange(H, device=dev, dtype=torch.float32)
    cc = torch.arange(W, device=dev, dtype=torch.float32)
    tot = m.sum((1, 2)).clamp_min(1)
    r0 = (m.sum(2) * rr).sum(1) / tot
    c0 = (m.sum(1) * cc).sum(1) / tot
    off = torch.stack([(c0 - (W - 1) / 2) * px_mm, ((H - 1) / 2 - r0) * px_mm], 1)
    # affine_grid maps output [-1, 1] to input [-1, 1]; build it in mm then normalise by the input extent
    half = fov_mm / 2
    t = torch.linspace(-half, half, out_px, device=dev)
    gy, gx = torch.meshgrid(-t, t, indexing="ij")                                   # up, right in mm
    th = np.deg2rad(roll_deg)
    rx = np.cos(th) * gx - np.sin(th) * gy
    ry = np.sin(th) * gx + np.cos(th) * gy
    if flip:
        rx = -rx
    xs = (off[:, 0, None, None] + rx) / (W * px_mm / 2)
    ys = -(off[:, 1, None, None] + ry) / (H * px_mm / 2)
    grid = torch.stack([xs, ys], -1)
    views = F.grid_sample(imgs[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0]
    return views, off


def dino_features(model, x):
    """x (B, 224, 224) grey in [0, 1] -> (B, 2D) unnormalised features: CLS ++ mean patch token."""
    x = x[:, None].repeat(1, 3, 1, 1)
    x = (x - MEAN.to(x.device)) / STD.to(x.device)
    tok = model.forward_features(x)
    return torch.cat([tok[:, 0], tok[:, model.num_prefix_tokens:].mean(1)], 1)


class Embedder:
    def __init__(self, arch="vit_small_patch14_dinov2.lvd142m", device=None, model=None, head=None):
        """Frozen DINOv2 by default; pass a fine-tuned backbone and a projection head (e.g. one branch of
        the trained dual encoder) to embed into the learned histology/MRI space instead."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model or timm.create_model(arch, pretrained=True, num_classes=0, img_size=224)
        self.model.eval().to(self.device)
        self.head = head.eval().to(self.device) if head is not None else None

    def features(self, x):
        return dino_features(self.model, x)

    @torch.inference_mode()
    def __call__(self, views, batch=128):
        out = []
        for i in range(0, len(views), batch):
            x = views[i:i + batch].to(self.device).float()
            if x.shape[-1] != 224:
                x = resize(x, (224, 224))
            with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
                f = self.features(x)
                if self.head is not None:
                    f = self.head(f)
            out.append(F.normalize(f.float(), dim=1).cpu())
        return torch.cat(out).numpy()
