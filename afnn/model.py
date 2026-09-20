"""Full AFNN model plus complexity/latency measurement helpers."""
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AFNNConfig
from .fractal import FractalBlock
from .layers import WeightedMerge, get_act, get_norm


def _convblock_parameter_count(channels: int, normalization: str) -> int:
    """Trainable parameters of a ConvBlock, computed without building it."""
    conv = 2 * (channels * channels * 3 * 3)
    norm = 4 * channels if normalization in {"batchnorm", "groupnorm"} else 0
    bias = 2 * channels if normalization == "none" else 0
    return conv + norm + bias


def _fractal_parameter_count(channels: int, depth: int, branches: int,
                             normalization: str, max_branches: int) -> int:
    """Trainable parameters of a FractalBlock, computed without building it. Used to check
    the growth budget.
    """
    if depth < 0:
        raise ValueError("depth must be >= 0")
    leaf = _convblock_parameter_count(channels, normalization)
    if depth == 0:
        return leaf
    # shallow ConvBlock + `branches` recursive children + fixed-capacity merge.
    merge = max_branches
    return leaf + (branches - 1) * _fractal_parameter_count(
        channels, depth - 1, branches, normalization, max_branches) + merge


class AFNNStage(nn.Module):
    """One stage: an optional 1x1 projection, a FractalBlock and a stride-2 downsampling
    conv (omitted in the last stage).
    """
    def __init__(self, in_c: int, out_c: int, cfg: AFNNConfig, is_last: bool):
        super().__init__()
        if in_c != out_c:
            self.proj = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, bias=(cfg.normalization == "none")),
                get_norm(cfg.normalization, out_c),
            )
        else:
            self.proj = nn.Identity()
        self.fractal = FractalBlock(
            out_c, cfg.initial_depth, cfg.initial_branches,
            cfg.normalization, cfg.activation,
            use_checkpointing=cfg.gradient_checkpointing,
            max_branches=cfg.max_branches_per_stage,
        )
        if not is_last:
            self.down = nn.Sequential(
                nn.Conv2d(out_c, out_c, 3, stride=2, padding=1, bias=(cfg.normalization == "none")),
                get_norm(cfg.normalization, out_c),
                get_act(cfg.activation),
            )
        else:
            self.down = nn.Identity()

    def forward(self, x):
        """projection -> fractal block -> downsample."""
        return self.down(self.fractal(self.proj(x)))

    @torch.no_grad()
    def hard_forward(self, x, k, mode="dense_inside"):
        """Same as ``forward`` but the fractal block uses only its top-k branches."""
        return self.down(self.fractal.hard_forward(self.proj(x), k, mode=mode))


class AFNN(nn.Module):
    """Adaptive Fractal Neural Network classifier.

    stem conv -> fractal stages -> global average pooling -> linear head. An optional
    representation/projection head supports contrastive training.
    """
    def __init__(self, cfg: AFNNConfig):
        super().__init__()
        self.cfg = cfg
        stem_c = cfg.base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.input_channels, stem_c, 3, padding=1, bias=(cfg.normalization == "none")),
            get_norm(cfg.normalization, stem_c),
            get_act(cfg.activation),
        )
        channels = [int(stem_c * m) for m in cfg.channel_multipliers]
        while len(channels) < cfg.stages:
            channels.append(channels[-1])
        channels = channels[:cfg.stages]
        self.stage_channels = channels
        stages = []
        in_c = stem_c
        for i, out_c in enumerate(channels):
            stages.append(AFNNStage(in_c, out_c, cfg,
                                     is_last=(i == cfg.stages - 1)))
            in_c = out_c
        self.stages = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = in_c
        self.head = nn.Linear(in_c, cfg.num_classes)

        # Optional representation head, attached beside the classifier.
        if cfg.use_representation_learning:
            self.representation = nn.Sequential(
                nn.Linear(in_c, cfg.representation_dim),
                nn.LayerNorm(cfg.representation_dim),
                nn.GELU(),
            )
            self.projector = nn.Sequential(
                nn.Linear(cfg.representation_dim, cfg.projection_dim),
                nn.GELU(),
                nn.Linear(cfg.projection_dim, cfg.projection_dim),
            )
        else:
            self.representation = None
            self.projector = None
        self._init_weights()

    def _init_weights(self):
        """Kaiming/Xavier init for convs, ones/zeros for BatchNorm, small normal init for
        linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.cfg.activation in {"relu"}:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _body(self, x):
        """Run the stem and the stages and return the output of every stage."""
        feats = []
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats

    def forward(self, x, return_features: bool = False):
        """Return logits, or a dict with logits and stage features when ``return_features``
        is True.
        """
        feats = self._body(x)
        pooled = self.pool(feats[-1]).flatten(1)
        logits = self.head(pooled)
        if return_features:
            return {"logits": logits, "features": feats}
        return logits

    def forward_features(self, x) -> Dict[str, torch.Tensor]:
        """Return the stage outputs as ``{'stage_0': ..., 'stage_1': ...}``."""
        return {f"stage_{i}": f for i, f in enumerate(self._body(x))}

    def extract_features(self, x) -> List[torch.Tensor]:
        """Return the list of stage outputs."""
        return self._body(x)

    def extract_pool(self, x) -> torch.Tensor:
        """Global-average-pooled features of the last stage."""
        return self.pool(self._body(x)[-1]).flatten(1)

    def extract_representation(self, x) -> torch.Tensor:
        """L2-normalised embedding from the representation head (pooled features if there
        is no head).
        """
        pooled = self.extract_pool(x)
        if self.representation is None:
            return F.normalize(pooled, dim=1)
        z = self.representation(pooled)
        return F.normalize(z, dim=1)

    def project_representation(self, x) -> torch.Tensor:
        """L2-normalised output of the projection head, used by the contrastive loss."""
        pooled = self.extract_pool(x)
        if self.representation is None or self.projector is None:
            return F.normalize(pooled, dim=1)
        h = self.representation(pooled)
        z = self.projector(h)
        return F.normalize(z, dim=1)

    @torch.no_grad()
    def hard_forward(self, x, k: int, mode: str = "dense_inside"):
        """Inference using only the top-``k`` branches of every stage."""
        x = self.stem(x)
        for stage in self.stages:
            x = stage.hard_forward(x, k, mode=mode)
        return self.head(self.pool(x).flatten(1))

    def count_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_branches_per_stage(self) -> List[int]:
        """Root branch count of every stage, e.g. ``[4, 4]``."""
        return [s.fractal.branches for s in self.stages]

    def count_total_branches(self) -> int:
        """Sum of the root branches of all stages."""
        return sum(self.count_branches_per_stage())

    def count_merge_nodes(self) -> int:
        """Number of WeightedMerge modules in the whole model."""
        return sum(1 for m in self.modules() if isinstance(m, WeightedMerge))

    def count_leaf_paths(self) -> int:
        """Number of distinct leaf ConvBlocks reachable in the fractal (all recursion
        levels).
        """
        def leaves(block):
            if not isinstance(block, FractalBlock) or block.depth == 0:
                return 1
            return sum(leaves(b) for b in block._all_branches())
        return sum(leaves(s.fractal) for s in self.stages)

    def grow_one_branch_per_stage(self, max_branches: int,
                                    max_total_params: int) -> int:
        """Try to add one branch to the root block of every stage and return how many were
        added.

        Growth respects ``max_branches`` and only happens if the resulting parameter
        count stays under ``max_total_params``. Only the root of each stage grows;
        nested blocks keep their initial branch count.
        """
        added = 0
        for stage in self.stages:
            if stage.fractal.branches >= max_branches:
                continue
            before = self.count_parameters()
            cost = _fractal_parameter_count(
                stage.fractal.channels, stage.fractal.depth - 1,
                stage.fractal._initial_branches, stage.fractal._norm,
                stage.fractal.max_branches,
            )
            if before + cost > max_total_params:
                continue
            if stage.fractal.add_branch(max_branches):
                added += 1
        return added

    @torch.no_grad()
    def update_all_branch_emas(self, sample_x: torch.Tensor):
        """Measure branch vitality in all stages. Runs in eval mode so BatchNorm statistics
        are untouched.
        """
        was_training = self.training
        self.eval()
        try:
            x = self.stem(sample_x)
            for stage in self.stages:
                x_proj = stage.proj(x)
                merged = stage.fractal.update_branch_ema(x_proj)
                x = stage.down(merged)
        finally:
            self.train(was_training)

    @torch.no_grad()
    def rejuvenate_dead_branches(self, std_threshold: float,
                                    min_observations: int = 20
                                    ) -> Tuple[int, List[nn.Parameter]]:
        """Re-initialise dead branches in every stage. Returns ``(count, parameters)``."""
        total = 0
        all_params: List[nn.Parameter] = []
        for stage in self.stages:
            n, params = stage.fractal.reinitialize_dead_branches(
                std_threshold, min_observations)
            total += n
            all_params.extend(params)
        return total, all_params

    @torch.no_grad()
    def prune_to_top_k(self, keep_top: int) -> int:
        """Prune every stage down to ``keep_top`` branches; returns the number removed."""
        return sum(s.fractal.prune_smallest_branches(keep_top)
                    for s in self.stages)

    def export_onnx(self, path: str,
                     input_shape: Optional[Tuple[int, ...]] = None,
                     opset: int = 14) -> str:
        """Export a static snapshot of the current architecture to ONNX (later growth is
        not reflected).
        """
        if input_shape is None:
            input_shape = (1, self.cfg.input_channels,
                            self.cfg.input_size, self.cfg.input_size)
        dummy = torch.randn(*input_shape)
        was = self.training
        # Export is a static snapshot of the current architecture. Future
        # Growth/Pruning on the Python model does not modify the exported graph.
        self.eval()
        try:
            torch.onnx.export(
                self, dummy, path,
                input_names=["input"], output_names=["logits"],
                opset_version=opset, do_constant_folding=True,
                dynamic_axes={"input": {0: "batch"},
                               "logits": {0: "batch"}},
            )
        finally:
            self.train(was)
        return path

    def export_torchscript(self, path: str,
                            input_shape: Optional[Tuple[int, ...]] = None) -> str:
        """Export a static snapshot of the current architecture with ``torch.jit.trace``."""
        if input_shape is None:
            input_shape = (1, self.cfg.input_channels,
                            self.cfg.input_size, self.cfg.input_size)
        dummy = torch.randn(*input_shape)
        was = self.training
        self.eval()
        try:
            traced = torch.jit.trace(self, dummy)
            traced.save(path)
        finally:
            self.train(was)
        return path


def measure_macs(model: nn.Module, input_shape: Tuple[int, ...],
                  device: str = "cpu") -> int:
    """Multiply-accumulate count of one forward pass (convolutions and linear layers) for a
    single input of ``input_shape``.
    """
    x = torch.zeros(1, *input_shape, device=device)
    total = [0]

    def conv_hook(m, inp, out):
        oh, ow = out.shape[2], out.shape[3]
        k = m.kernel_size[0] * m.kernel_size[1]
        total[0] += oh * ow * m.in_channels * m.out_channels * k // m.groups

    def linear_hook(m, inp, out):
        total[0] += m.in_features * m.out_features

    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
        model.train(was)
    return total[0]


def measure_latency(model: nn.Module, input_shape: Tuple[int, ...],
                     device: str = "cpu", n_warmup: int = 10,
                     n_iters: int = 50) -> Dict[str, float]:
    """Forward latency for one image: mean, p50, p95 and std in milliseconds, after warm-
    up.
    """
    x = torch.randn(1, *input_shape, device=device)
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(n_warmup):
                model(x)
            times = []
            if device.startswith("cuda"):
                for _ in range(n_iters):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record(); model(x); end.record()
                    torch.cuda.synchronize()
                    times.append(start.elapsed_time(end))
            else:
                for _ in range(n_iters):
                    t0 = time.perf_counter(); model(x)
                    times.append((time.perf_counter() - t0) * 1000.0)
            arr = np.asarray(times, dtype=np.float64)
            return {
                "per_forward_ms": float(arr.mean()),
                "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)),
                "std_ms": float(arr.std()),
                "device": device,
            }
    finally:
        model.train(was)
