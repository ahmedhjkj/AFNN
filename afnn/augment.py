"""Batch-level augmentation: Mixup / CutMix and a fast GPU crop+flip."""
import math

import torch
import torch.nn.functional as F


class MixupCutMix:
    """Randomly applies Mixup or CutMix to a batch.

    Usage: ``x, y_a, y_b, lam = mixup(x, y)``. When nothing is applied, ``y_b`` and
    ``lam`` are None.
    """
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0,
                 prob=0.5, enable_mixup=True, enable_cutmix=True):
        self.ma = mixup_alpha
        self.ca = cutmix_alpha
        self.prob = prob
        self.em = enable_mixup
        self.ec = enable_cutmix

    def __call__(self, x, y):
        """Mix the batch with probability ``prob``; pick Mixup or CutMix at random among
        the enabled ones.
        """
        if torch.rand(1).item() > self.prob:
            return x, y, None, None
        choices = []
        if self.em: choices.append("mixup")
        if self.ec: choices.append("cutmix")
        if not choices:
            return x, y, None, None
        pick = choices[torch.randint(0, len(choices), (1,)).item()]
        return (self._mixup(x, y) if pick == "mixup" else self._cutmix(x, y))

    @staticmethod
    def _torch_beta(alpha: float, device) -> float:
        """Sample from Beta(alpha, alpha) using torch's RNG (so ``torch.manual_seed`` makes
        runs reproducible).
        """
        dist = torch.distributions.Beta(
            torch.tensor(float(alpha), device=device),
            torch.tensor(float(alpha), device=device),
        )
        return dist.sample().item()

    def _mixup(self, x, y):
        """Blend each image with a shuffled partner: ``lam * x + (1 - lam) * x_shuffled``."""
        lam = self._torch_beta(self.ma, x.device)
        idx = torch.randperm(x.size(0), device=x.device)
        return lam * x + (1 - lam) * x[idx], y, y[idx], lam

    def _cutmix(self, x, y):
        """Paste a random box from a shuffled partner and correct ``lam`` to the real
        pasted area.
        """
        lam = self._torch_beta(self.ca, x.device)
        B, C, H, W = x.shape
        idx = torch.randperm(B, device=x.device)
        r = math.sqrt(1 - lam)
        cw, ch = int(W * r), int(H * r)
        cx = torch.randint(0, W, (1,)).item()
        cy = torch.randint(0, H, (1,)).item()
        x1, y1 = max(cx - cw // 2, 0), max(cy - ch // 2, 0)
        x2, y2 = min(cx + cw // 2, W), min(cy + ch // 2, H)
        x_cut = x.clone()
        x_cut[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
        return x_cut, y, y[idx], lam


def gpu_augment(x: torch.Tensor) -> torch.Tensor:
    """Random horizontal flip and random shifted crop for a whole batch on the GPU."""
    B, C, H, W = x.shape
    dev = x.device
    flip = torch.rand(B, device=dev) < 0.5
    x = torch.where(flip.view(B, 1, 1, 1), x.flip(3), x)
    pad = max(2, H // 8)
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    i = torch.randint(0, 2 * pad + 1, (B,), device=dev)
    j = torch.randint(0, 2 * pad + 1, (B,), device=dev)
    rows = (i.view(B, 1) + torch.arange(H, device=dev).view(1, H)).view(B, 1, H, 1)
    cols = (j.view(B, 1) + torch.arange(W, device=dev).view(1, W)).view(B, 1, 1, W)
    bidx = torch.arange(B, device=dev).view(B, 1, 1, 1)
    cidx = torch.arange(C, device=dev).view(1, C, 1, 1)
    return xp[bidx, cidx, rows, cols]
