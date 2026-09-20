# Configuration reference

Every field of `AFNNConfig` (`afnn/config.py`) with its default. `build_trainer()` and the GUI/CLI set the most common ones for you.

| Field | Default | Meaning |
|---|---|---|
| `input_channels` | `3` | Channels of the input image (3 = RGB). |
| `input_size` | `32` | Side of the square input in pixels; images are resized to it. |
| `num_classes` | `10` | Number of output classes. |
| `base_channels` | `32` | Channels of the stem and of stage 1; later stages multiply it. |
| `channel_multipliers` | `(1.0, 2.0)` | Channel multiplier of each stage relative to ``base_channels``. |
| `stages` | `2` | Number of stages (each is a fractal block plus a downsampling conv). |
| `initial_depth` | `2` | Recursion depth of the fractal block in every stage. |
| `initial_branches` | `2` | Branches per block at the start; nested blocks keep this count. |
| `normalization` | `batchnorm` | ``batchnorm``, ``groupnorm`` or ``none``. |
| `activation` | `relu` | ``relu``, ``gelu``, ``silu`` or ``mish``. |
| `growth_enabled` | `True` | Allow the network to add branches during training. |
| `growth_patience` | `3` | Epochs without validation-loss improvement before it grows. |
| `growth_min_delta` | `0.005` | Smallest validation-loss decrease that counts as an improvement. |
| `max_branches_per_stage` | `5` | Upper limit of root branches per stage. |
| `max_total_params` | `10000000` | Growth stops when the parameter count would exceed this. |
| `new_branch_lr_multiplier` | `5.0` | LR multiplier applied to the parameters of a new branch. |
| `new_branch_warmup_epochs` | `2` | Epochs over which a new branch's LR is ramped up. |
| `rejuvenation_enabled` | `True` | Re-initialise branches whose output has collapsed. |
| `dead_branch_std_threshold` | `0.001` | A branch with smoothed output std below this is dead. |
| `rejuvenation_check_every` | `5` | Check for dead branches every N epochs. |
| `rejuvenation_min_observations` | `20` | Vitality samples required before a branch can be judged. |
| `rejuvenation_ema_stride` | `16` | Measure branch vitality every N batches (it costs a forward pass). |
| `use_mixup` | `True` | Enable Mixup. |
| `mixup_alpha` | `0.2` | Beta(alpha, alpha) parameter of Mixup. |
| `use_cutmix` | `True` | Enable CutMix. |
| `cutmix_alpha` | `1.0` | Beta(alpha, alpha) parameter of CutMix. |
| `label_smoothing` | `0.1` | Label smoothing of the cross-entropy loss. |
| `use_load_balance` | `True` | Add the load-balancing loss on the merge weights. |
| `load_balance_weight` | `0.01` | Weight of the load-balancing loss. |
| `use_representation_learning` | `False` | Add the contrastive representation loss (two extra passes per batch). |
| `representation_dim` | `128` | Size of the representation embedding. |
| `projection_dim` | `128` | Size of the projection head output. |
| `contrastive_weight` | `0.2` | Weight of the contrastive loss. |
| `invariance_weight` | `0.05` | Weight of the view-invariance loss. |
| `contrastive_temperature` | `0.2` | Temperature of the contrastive loss. |
| `representation_warmup_epochs` | `3` | Epochs before the representation loss is switched on. |
| `use_amp` | `True` | Mixed precision. Switched off automatically on GPUs older than Volta. |
| `use_ema` | `True` | Keep an exponential moving average of the weights. |
| `ema_decay` | `0.999` | EMA decay. |
| `warmup_epochs` | `2` | Linear LR warm-up epochs. |
| `cosine_t_max` | `20` | Length of the cosine schedule in epochs (the trainer overrides it). |
| `gradient_checkpointing` | `False` | Trade compute for memory inside fractal blocks. |
| `architecture_version` | `1.1.0` | Version tag stored in checkpoints. |
