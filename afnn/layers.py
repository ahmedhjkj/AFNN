"""Basic building blocks: normalization/activation factories, ConvBlock, WeightedMerge."""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


_GROUP_LOOKUP = {
    16: 2, 32: 4, 48: 4, 64: 8, 96: 8, 128: 8,
    192: 16, 256: 16, 384: 16, 512: 32, 768: 32, 1024: 32, 2048: 32,
}


def _select_groups(c: int) -> int:
    """Pick a GroupNorm group count that divides ``c`` (lookup table first, then the
    largest sensible divisor).
    """
    if c in _GROUP_LOOKUP:
        return _GROUP_LOOKUP[c]
    for g in (32, 24, 16, 12, 8, 6, 4, 3, 2):
        if c % g == 0:
            return g
    for g in range(min(32, c), 1, -1):
        if c % g == 0:
            return g
    return 1


def get_norm(name: str, c: int) -> nn.Module:
    """Return BatchNorm2d, GroupNorm or Identity for ``c`` channels."""
    if name == "batchnorm":
        return nn.BatchNorm2d(c)
    if name == "groupnorm":
        return nn.GroupNorm(_select_groups(c), c)
    return nn.Identity()


def get_act(name: str) -> nn.Module:
    """Return the activation module called ``name`` (relu, gelu, silu or mish)."""
    return {
        "relu": lambda: nn.ReLU(inplace=True),
        "gelu": lambda: nn.GELU(),
        "silu": lambda: nn.SiLU(inplace=True),
        "mish": lambda: nn.Mish(inplace=True),
    }[name]()


class ConvBlock(nn.Module):
    """Two 3x3 convolutions, each followed by normalization and activation. It is the leaf
    of the fractal.
    """
    def __init__(self, c: int, norm: str = "batchnorm", act: str = "relu"):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=(norm == "none"))
        self.n1 = get_norm(norm, c)
        self.a1 = get_act(act)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=(norm == "none"))
        self.n2 = get_norm(norm, c)
        self.a2 = get_act(act)

    def forward(self, x):
        """Apply conv -> norm -> act twice."""
        h = self.a1(self.n1(self.conv1(x)))
        return self.a2(self.n2(self.conv2(h)))


class WeightedMerge(nn.Module):
    """Softmax-weighted sum of branch outputs.

    The logits live in a fixed-size parameter (``max_branches`` slots) of which only
    ``num_branches`` are active. The parameter object is never replaced during growth or
    pruning, because a replaced parameter would be detached from the optimizer.
    """
    def __init__(self, num_branches: int, max_branches: int = 5):
        super().__init__()
        if num_branches < 1 or num_branches > max_branches:
            raise ValueError(f"num_branches={num_branches} must be in [1, {max_branches}]")
        self.num_branches = num_branches
        self.max_branches = max_branches
        self.logits = nn.Parameter(torch.zeros(max_branches))

    def weights(self) -> torch.Tensor:
        """Softmax of the active logits (they sum to 1)."""
        return F.softmax(self.logits[:self.num_branches], dim=0)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """Merge a list of branch outputs using the current weights."""
        if len(feats) != self.num_branches:
            raise RuntimeError(
                f"WeightedMerge expected {self.num_branches} features, got {len(feats)}")
        if self.num_branches == 1:
            return feats[0]
        w = self.weights()
        out = feats[0] * w[0]
        for i in range(1, self.num_branches):
            out = out + feats[i] * w[i]
        return out

    @torch.no_grad()
    def add_weight(self, initial_logit: float = -30.0):
        """Activate the next slot with a very negative logit, so the new zero-initialised
        branch starts almost invisible and is blended in as it learns. Returns False
        when the merge is full.
        """
        if self.num_branches >= self.max_branches:
            return False
        self.logits[self.num_branches].fill_(float(initial_logit))
        self.num_branches += 1
        return True

    @torch.no_grad()
    def remove_weight(self, idx: int):
        """Remove slot ``idx`` and shift the later slots down by one, exactly like the
        branch list is compacted. Callers must remove indices from highest to lowest.
        """
        if not 0 <= idx < self.num_branches:
            raise IndexError(idx)
        last = self.num_branches - 1
        if idx != last:
            self.logits[idx:last] = self.logits[idx + 1:last + 1].clone()
        self.logits[last].zero_()
        self.num_branches -= 1
