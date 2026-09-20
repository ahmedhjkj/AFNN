"""Model and training configuration."""
from dataclasses import asdict, dataclass, fields
from typing import Tuple


@dataclass
class AFNNConfig:
    """Hyper-parameters of the network and of its training loop.

    The values are validated in ``__post_init__``. ``to_dict`` / ``from_dict`` are used
    to store the config inside checkpoints.
    """
    # Input
    input_channels: int = 3
    input_size: int = 32
    num_classes: int = 10

    # Width and depth
    base_channels: int = 32
    channel_multipliers: Tuple[float, ...] = (1.0, 2.0)
    stages: int = 2
    initial_depth: int = 2
    initial_branches: int = 2

    # Norm/Act
    normalization: str = "batchnorm"
    activation: str = "relu"

    # Growth
    growth_enabled: bool = True
    growth_patience: int = 3
    growth_min_delta: float = 0.005
    max_branches_per_stage: int = 5
    max_total_params: int = 10_000_000
    new_branch_lr_multiplier: float = 5.0
    new_branch_warmup_epochs: int = 2

    # Rejuvenation
    rejuvenation_enabled: bool = True
    dead_branch_std_threshold: float = 1e-3
    rejuvenation_check_every: int = 5
    rejuvenation_min_observations: int = 20
    # Branch vitality is measured with an extra forward pass; sample it every N batches.
    rejuvenation_ema_stride: int = 16

    # Regularization
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    use_cutmix: bool = True
    cutmix_alpha: float = 1.0
    label_smoothing: float = 0.1
    use_load_balance: bool = True
    load_balance_weight: float = 0.01

    # Contrastive representation learning (optional, slower)
    use_representation_learning: bool = False
    representation_dim: int = 128
    projection_dim: int = 128
    contrastive_weight: float = 0.20
    invariance_weight: float = 0.05
    contrastive_temperature: float = 0.20
    representation_warmup_epochs: int = 3

    # Training
    use_amp: bool = True
    use_ema: bool = True
    ema_decay: float = 0.999
    warmup_epochs: int = 2
    cosine_t_max: int = 20
    gradient_checkpointing: bool = False

    architecture_version: str = "1.1.0"

    def __post_init__(self):
        """Reject invalid values early instead of failing deep inside training."""
        assert self.input_channels >= 1
        assert self.stages >= 1
        assert self.initial_branches >= 2
        assert self.max_branches_per_stage >= self.initial_branches
        assert self.normalization in {"batchnorm", "groupnorm", "none"}
        assert self.activation in {"relu", "gelu", "silu", "mish"}
        assert 0.0 <= self.label_smoothing < 1.0
        assert self.max_total_params > 0
        assert self.max_branches_per_stage >= 2
        assert self.new_branch_lr_multiplier > 0
        assert self.new_branch_warmup_epochs >= 0
        assert self.warmup_epochs >= 0
        assert self.cosine_t_max >= 1
        if self.use_mixup:
            assert self.mixup_alpha > 0, "mixup_alpha must be > 0 when Mixup is enabled"
        if self.use_cutmix:
            assert self.cutmix_alpha > 0, "cutmix_alpha must be > 0 when CutMix is enabled"
        assert self.representation_dim >= 1
        assert self.projection_dim >= 1
        assert self.contrastive_weight >= 0.0
        assert self.invariance_weight >= 0.0
        assert self.contrastive_temperature > 0.0
        assert self.representation_warmup_epochs >= 0
        assert self.rejuvenation_ema_stride >= 1

    def to_dict(self):
        """Return the config as a plain dict (tuples become lists, so it is JSON friendly)."""
        d = asdict(self)
        d["channel_multipliers"] = list(self.channel_multipliers)
        return d

    @classmethod
    def from_dict(cls, d):
        """Rebuild a config from the output of ``to_dict``. Unknown keys (from checkpoints
        written by older versions) are ignored.
        """
        known = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in known}      # ignore fields removed since
        if "channel_multipliers" in d:
            d["channel_multipliers"] = tuple(d["channel_multipliers"])
        return cls(**d)
