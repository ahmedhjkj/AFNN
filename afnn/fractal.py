"""Recursive fractal block with growth, pruning and branch rejuvenation."""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .layers import ConvBlock, WeightedMerge


class FractalBlock(nn.Module):
    """Recursive fractal block.

    A block of depth ``d`` runs one shallow ConvBlock and ``branches - 1`` blocks of
    depth ``d - 1`` in parallel and merges them with a WeightedMerge. Depth 0 is a
    single ConvBlock. The number of branches can grow (``add_branch``) or shrink
    (``prune_branch``), and dead branches can be re-initialised
    (``reinitialize_dead_branches``).
    """
    def __init__(self, channels: int, depth: int, branches: int = 2,
                 norm: str = "batchnorm", act: str = "relu",
                 use_checkpointing: bool = False,
                 zero_init: bool = False, max_branches: int = 5):
        super().__init__()
        assert depth >= 0 and branches >= 2

        self.channels = channels
        self.depth = depth
        self.branches = branches
        self._initial_branches = branches
        self._norm = norm
        self._act = act
        self.use_checkpointing = use_checkpointing
        self.max_branches = max(max_branches, branches)

        self._branch_std_ema: Optional[List[float]] = None
        self._branch_ema_decay = 0.9
        self._branch_obs_count = 0

        if depth == 0:
            self.leaf = ConvBlock(channels, norm, act)
            self.shallow = None
            self.deep = None
            self.extras = None
            self.merge = None
            return

        self.shallow = ConvBlock(channels, norm, act)
        self.deep = FractalBlock(
            channels, depth - 1, self._initial_branches, norm, act,
            use_checkpointing=use_checkpointing, zero_init=zero_init, max_branches=self.max_branches,
        )
        if branches > 2:
            self.extras = nn.ModuleList([
                FractalBlock(channels, depth - 1, self._initial_branches,
                              norm, act, use_checkpointing=use_checkpointing,
                              zero_init=zero_init, max_branches=self.max_branches)
                for _ in range(branches - 2)
            ])
        else:
            self.extras = None

        self.merge = WeightedMerge(branches, max_branches=self.max_branches)
        if zero_init:
            self._zero_init_last_conv()

    def _zero_init_last_conv(self):
        """Zero the last convolution of every ConvBlock inside, so the block outputs zeros."""
        if self.depth == 0:
            nn.init.zeros_(self.leaf.conv2.weight)
            if self.leaf.conv2.bias is not None:
                nn.init.zeros_(self.leaf.conv2.bias)
            return
        for branch in self._all_branches():
            if isinstance(branch, FractalBlock):
                branch._zero_init_last_conv()
            elif isinstance(branch, ConvBlock):
                nn.init.zeros_(branch.conv2.weight)
                if branch.conv2.bias is not None:
                    nn.init.zeros_(branch.conv2.bias)

    def _all_branches(self) -> List[nn.Module]:
        """Return the parallel branches in merge order: shallow, deep, then the extra
        branches.
        """
        mods = [self.shallow, self.deep]
        if self.extras is not None:
            mods.extend(list(self.extras))
        return mods

    def _forward_impl(self, x):
        """Run all branches and merge them (or run the leaf ConvBlock at depth 0)."""
        if self.depth == 0:
            return self.leaf(x)
        feats = [b(x) for b in self._all_branches()]
        return self.merge(feats)

    def forward(self, x):
        """Forward pass, with gradient checkpointing while training if enabled."""
        if self.use_checkpointing and self.training and self.depth > 0:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, use_reentrant=False,
            )
        return self._forward_impl(x)

    @torch.no_grad()
    def hard_forward(self, x, k: int, mode: str = "dense_inside"):
        """Inference with only the ``k`` heaviest branches.

        ``dense_inside`` selects the top-k at this level only; ``recursive`` selects the
        top-k at every level.
        """
        if mode not in {"dense_inside", "recursive"}:
            raise ValueError(f"Unknown hard_forward mode: {mode}")
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        if self.depth == 0:
            return self.leaf(x)
        if k >= self.branches or k < 1:
            return self.forward(x)

        w = self.merge.weights()
        topk_idx = torch.topk(w, k).indices.tolist()
        branches = self._all_branches()

        feats = []
        for i in topk_idx:
            b = branches[i]
            if mode == "recursive" and isinstance(b, FractalBlock) and b.depth > 0:
                feats.append(b.hard_forward(x, k, mode=mode))
            else:
                feats.append(b(x))

        w_sel = w[topk_idx]
        w_sel = w_sel / w_sel.sum().clamp(min=1e-8)
        out = feats[0] * w_sel[0]
        for i in range(1, len(feats)):
            out = out + feats[i] * w_sel[i]
        return out

    def add_branch(self, max_branches: int) -> bool:
        """Add one branch if the limits allow it and return True on success.

        The new branch is zero-initialised, so the network output barely changes, and it
        is moved to the device and dtype of the existing parameters (a branch built on
        the CPU next to CUDA weights raises "Input type (torch.cuda.FloatTensor) and
        weight type (torch.FloatTensor) should be the same").
        """
        if self.depth == 0 or self.branches >= min(max_branches, self.max_branches):
            return False
        new_branch = FractalBlock(
            self.channels, self.depth - 1, self._initial_branches,
            self._norm, self._act, use_checkpointing=self.use_checkpointing,
            zero_init=False, max_branches=self.max_branches,
        )
        self._apply_zero_init_to_branch(new_branch)
        ref = next(self.parameters(), None)
        if ref is not None:
            new_branch.to(device=ref.device, dtype=ref.dtype)
        if self.extras is None:
            self.extras = nn.ModuleList([new_branch])
        else:
            self.extras.append(new_branch)
        self.branches += 1
        self.merge.add_weight()
        if self._branch_std_ema is not None:
            self._branch_std_ema.append(0.0)
        return True

    def _apply_zero_init_to_branch(self, branch: nn.Module):
        """Zero the terminal convolutions of a new branch so it initially outputs zero."""
        if isinstance(branch, FractalBlock):
            branch._zero_init_last_conv()
        else:
            last_conv = None
            for m in branch.modules():
                if isinstance(m, nn.Conv2d):
                    last_conv = m
            if last_conv is not None:
                nn.init.zeros_(last_conv.weight)

    @torch.no_grad()
    def prune_branch(self, idx: int) -> bool:
        """Remove the extra branch at merge index ``idx`` (indices 0 and 1 are protected).
        Returns True on success.
        """
        if self.depth == 0 or idx in (0, 1):
            return False
        extra_idx = idx - 2
        if self.extras is None or extra_idx >= len(self.extras):
            return False
        keep = [m for i, m in enumerate(self.extras) if i != extra_idx]
        self.extras = nn.ModuleList(keep) if keep else None
        self.branches -= 1
        self.merge.remove_weight(idx)
        # Keep the vitality list aligned with the compacted branch list.
        if self._branch_std_ema is not None and idx < len(self._branch_std_ema):
            del self._branch_std_ema[idx]
        return True

    @torch.no_grad()
    def prune_smallest_branches(self, keep_top: int) -> int:
        """Keep the protected branches plus the heaviest ones until ``keep_top`` remain;
        return how many were removed.
        """
        if self.depth == 0 or self.branches <= max(keep_top, 2):
            return 0
        w = self.merge.weights().clone()
        order = torch.argsort(w, descending=True).tolist()
        keep = {0, 1}
        target = max(keep_top, 2)
        for idx in order:
            if len(keep) >= target:
                break
            keep.add(idx)
        remove_idxs = sorted(
            [i for i in range(self.branches) if i not in keep], reverse=True,
        )
        n_removed = 0
        for idx in remove_idxs:
            if self.prune_branch(idx):
                n_removed += 1
        return n_removed

    @torch.no_grad()
    def update_branch_ema(self, x: torch.Tensor):
        """Measure the output std of every branch, smooth it with an EMA and return the
        merged output. A branch whose std stays near zero is considered dead.
        """
        if self.depth == 0:
            return self.leaf(x)
        branches = self._all_branches()
        feats = [b(x) for b in branches]
        new_stds = [f.std().item() for f in feats]
        if self._branch_std_ema is None:
            self._branch_std_ema = new_stds
        elif len(self._branch_std_ema) != len(new_stds):
            self._branch_std_ema = new_stds
        else:
            d = self._branch_ema_decay
            self._branch_std_ema = [
                d * old + (1 - d) * new
                for old, new in zip(self._branch_std_ema, new_stds)
            ]
        self._branch_obs_count += 1
        return self.merge(feats)

    @torch.no_grad()
    def reinitialize_dead_branches(self, std_threshold: float,
                                     min_observations: int = 20
                                     ) -> Tuple[int, List[nn.Parameter]]:
        """Re-initialise extra branches whose smoothed std fell below ``std_threshold``.
        Returns ``(count, reinitialised_parameters)``.
        """
        if self.depth == 0 or self._branch_std_ema is None:
            return 0, []
        if self._branch_obs_count < min_observations:
            return 0, []
        branches = self._all_branches()
        revived = 0
        reinit_params: List[nn.Parameter] = []
        for i, ema_std in enumerate(self._branch_std_ema):
            if i in (0, 1):
                continue
            if i >= len(branches):
                continue
            if ema_std < std_threshold:
                for p in branches[i].parameters():
                    reinit_params.append(p)
                self._reinit_module(branches[i])
                # Reactivate the parent gate slot to a neutral logit so the
                # revived branch is not permanently suppressed or over-weighted.
                self.merge.logits[i].zero_()
                self._branch_std_ema[i] = 0.0
                revived += 1
        return revived, reinit_params

    def _reinit_module(self, module: nn.Module):
        """Re-initialise the conv, linear and norm weights of ``module`` in place."""
        for m in module.modules():
            if m is module:
                continue
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if self._act == "relu":
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                if isinstance(m, nn.BatchNorm2d):
                    if m.running_mean is not None:
                        m.running_mean.zero_()
                    if m.running_var is not None:
                        m.running_var.fill_(1.0)
            elif isinstance(m, WeightedMerge):
                m.logits.zero_()
