# API reference

Generated from the docstrings by `tools/gen_api_docs.py`; edit the docstrings, not this file.

| File | Contents |
|---|---|
| [`afnn/config.py`](#afnnconfigpy) | Configuration dataclass. |
| [`afnn/layers.py`](#afnnlayerspy) | Building blocks. |
| [`afnn/fractal.py`](#afnnfractalpy) | The recursive fractal block. |
| [`afnn/model.py`](#afnnmodelpy) | Stages, the full network and measurement helpers. |
| [`afnn/augment.py`](#afnnaugmentpy) | Batch augmentation. |
| [`afnn/data.py`](#afnndatapy) | Image folders, cache and loaders. |
| [`afnn/training.py`](#afnntrainingpy) | EMA, growth, LR schedule, Trainer, fit. |
| [`afnn/inference.py`](#afnninferencepy) | Checkpoint loading and prediction. |
| [`gui/app.py`](#guiapppy) | Tkinter application. |
| [`scripts/train.py`](#scriptstrainpy) | Command-line training. |
| [`scripts/predict.py`](#scriptspredictpy) | Command-line prediction. |

## `afnn/config.py`

Model and training configuration.

### class `AFNNConfig()`

Hyper-parameters of the network and of its training loop.

The values are validated in ``__post_init__``. ``to_dict`` / ``from_dict`` are used
to store the config inside checkpoints.

#### `__post_init__(self)`

Reject invalid values early instead of failing deep inside training.

#### `to_dict(self)`

Return the config as a plain dict (tuples become lists, so it is JSON friendly).

#### `from_dict(cls, d)`

Rebuild a config from the output of ``to_dict``. Unknown keys (from checkpoints
written by older versions) are ignored.

## `afnn/layers.py`

Basic building blocks: normalization/activation factories, ConvBlock, WeightedMerge.

### `_select_groups(c: int) -> int` *(internal)*

Pick a GroupNorm group count that divides ``c`` (lookup table first, then the
largest sensible divisor).

### `get_norm(name: str, c: int) -> nn.Module`

Return BatchNorm2d, GroupNorm or Identity for ``c`` channels.

### `get_act(name: str) -> nn.Module`

Return the activation module called ``name`` (relu, gelu, silu or mish).

### class `ConvBlock(c: int, norm: str='batchnorm', act: str='relu')`

Two 3x3 convolutions, each followed by normalization and activation. It is the leaf
of the fractal.

#### `forward(self, x)`

Apply conv -> norm -> act twice.

### class `WeightedMerge(num_branches: int, max_branches: int=5)`

Softmax-weighted sum of branch outputs.

The logits live in a fixed-size parameter (``max_branches`` slots) of which only
``num_branches`` are active. The parameter object is never replaced during growth or
pruning, because a replaced parameter would be detached from the optimizer.

#### `weights(self) -> torch.Tensor`

Softmax of the active logits (they sum to 1).

#### `forward(self, feats: List[torch.Tensor]) -> torch.Tensor`

Merge a list of branch outputs using the current weights.

#### `add_weight(self, initial_logit: float=-30.0)`

Activate the next slot with a very negative logit, so the new zero-initialised
branch starts almost invisible and is blended in as it learns. Returns False
when the merge is full.

#### `remove_weight(self, idx: int)`

Remove slot ``idx`` and shift the later slots down by one, exactly like the
branch list is compacted. Callers must remove indices from highest to lowest.

## `afnn/fractal.py`

Recursive fractal block with growth, pruning and branch rejuvenation.

### class `FractalBlock(channels: int, depth: int, branches: int=2, norm: str='batchnorm', act: str='relu', use_checkpointing: bool=False, zero_init: bool=False, max_branches: int=5)`

Recursive fractal block.

A block of depth ``d`` runs one shallow ConvBlock and ``branches - 1`` blocks of
depth ``d - 1`` in parallel and merges them with a WeightedMerge. Depth 0 is a
single ConvBlock. The number of branches can grow (``add_branch``) or shrink
(``prune_branch``), and dead branches can be re-initialised
(``reinitialize_dead_branches``).

#### `_zero_init_last_conv(self)` *(internal)*

Zero the last convolution of every ConvBlock inside, so the block outputs zeros.

#### `_all_branches(self) -> List[nn.Module]` *(internal)*

Return the parallel branches in merge order: shallow, deep, then the extra
branches.

#### `_forward_impl(self, x)` *(internal)*

Run all branches and merge them (or run the leaf ConvBlock at depth 0).

#### `forward(self, x)`

Forward pass, with gradient checkpointing while training if enabled.

#### `hard_forward(self, x, k: int, mode: str='dense_inside')`

Inference with only the ``k`` heaviest branches.

``dense_inside`` selects the top-k at this level only; ``recursive`` selects the
top-k at every level.

#### `add_branch(self, max_branches: int) -> bool`

Add one branch if the limits allow it and return True on success.

The new branch is zero-initialised, so the network output barely changes, and it
is moved to the device and dtype of the existing parameters (a branch built on
the CPU next to CUDA weights raises "Input type (torch.cuda.FloatTensor) and
weight type (torch.FloatTensor) should be the same").

#### `_apply_zero_init_to_branch(self, branch: nn.Module)` *(internal)*

Zero the terminal convolutions of a new branch so it initially outputs zero.

#### `prune_branch(self, idx: int) -> bool`

Remove the extra branch at merge index ``idx`` (indices 0 and 1 are protected).
Returns True on success.

#### `prune_smallest_branches(self, keep_top: int) -> int`

Keep the protected branches plus the heaviest ones until ``keep_top`` remain;
return how many were removed.

#### `update_branch_ema(self, x: torch.Tensor)`

Measure the output std of every branch, smooth it with an EMA and return the
merged output. A branch whose std stays near zero is considered dead.

#### `reinitialize_dead_branches(self, std_threshold: float, min_observations: int=20) -> Tuple[int, List[nn.Parameter]]`

Re-initialise extra branches whose smoothed std fell below ``std_threshold``.
Returns ``(count, reinitialised_parameters)``.

#### `_reinit_module(self, module: nn.Module)` *(internal)*

Re-initialise the conv, linear and norm weights of ``module`` in place.

## `afnn/model.py`

Full AFNN model plus complexity/latency measurement helpers.

### `_convblock_parameter_count(channels: int, normalization: str) -> int` *(internal)*

Trainable parameters of a ConvBlock, computed without building it.

### `_fractal_parameter_count(channels: int, depth: int, branches: int, normalization: str, max_branches: int) -> int` *(internal)*

Trainable parameters of a FractalBlock, computed without building it. Used to check
the growth budget.

### class `AFNNStage(in_c: int, out_c: int, cfg: AFNNConfig, is_last: bool)`

One stage: an optional 1x1 projection, a FractalBlock and a stride-2 downsampling
conv (omitted in the last stage).

#### `forward(self, x)`

projection -> fractal block -> downsample.

#### `hard_forward(self, x, k, mode='dense_inside')`

Same as ``forward`` but the fractal block uses only its top-k branches.

### class `AFNN(cfg: AFNNConfig)`

Adaptive Fractal Neural Network classifier.

stem conv -> fractal stages -> global average pooling -> linear head. An optional
representation/projection head supports contrastive training.

#### `_init_weights(self)` *(internal)*

Kaiming/Xavier init for convs, ones/zeros for BatchNorm, small normal init for
linear layers.

#### `_body(self, x)` *(internal)*

Run the stem and the stages and return the output of every stage.

#### `forward(self, x, return_features: bool=False)`

Return logits, or a dict with logits and stage features when ``return_features``
is True.

#### `forward_features(self, x) -> Dict[str, torch.Tensor]`

Return the stage outputs as ``{'stage_0': ..., 'stage_1': ...}``.

#### `extract_features(self, x) -> List[torch.Tensor]`

Return the list of stage outputs.

#### `extract_pool(self, x) -> torch.Tensor`

Global-average-pooled features of the last stage.

#### `extract_representation(self, x) -> torch.Tensor`

L2-normalised embedding from the representation head (pooled features if there
is no head).

#### `project_representation(self, x) -> torch.Tensor`

L2-normalised output of the projection head, used by the contrastive loss.

#### `hard_forward(self, x, k: int, mode: str='dense_inside')`

Inference using only the top-``k`` branches of every stage.

#### `count_parameters(self) -> int`

Number of trainable parameters.

#### `count_branches_per_stage(self) -> List[int]`

Root branch count of every stage, e.g. ``[4, 4]``.

#### `count_total_branches(self) -> int`

Sum of the root branches of all stages.

#### `count_merge_nodes(self) -> int`

Number of WeightedMerge modules in the whole model.

#### `count_leaf_paths(self) -> int`

Number of distinct leaf ConvBlocks reachable in the fractal (all recursion
levels).

#### `grow_one_branch_per_stage(self, max_branches: int, max_total_params: int) -> int`

Try to add one branch to the root block of every stage and return how many were
added.

Growth respects ``max_branches`` and only happens if the resulting parameter
count stays under ``max_total_params``. Only the root of each stage grows;
nested blocks keep their initial branch count.

#### `update_all_branch_emas(self, sample_x: torch.Tensor)`

Measure branch vitality in all stages. Runs in eval mode so BatchNorm statistics
are untouched.

#### `rejuvenate_dead_branches(self, std_threshold: float, min_observations: int=20) -> Tuple[int, List[nn.Parameter]]`

Re-initialise dead branches in every stage. Returns ``(count, parameters)``.

#### `prune_to_top_k(self, keep_top: int) -> int`

Prune every stage down to ``keep_top`` branches; returns the number removed.

#### `export_onnx(self, path: str, input_shape: Optional[Tuple[int, ...]]=None, opset: int=14) -> str`

Export a static snapshot of the current architecture to ONNX (later growth is
not reflected).

#### `export_torchscript(self, path: str, input_shape: Optional[Tuple[int, ...]]=None) -> str`

Export a static snapshot of the current architecture with ``torch.jit.trace``.

### `measure_macs(model: nn.Module, input_shape: Tuple[int, ...], device: str='cpu') -> int`

Multiply-accumulate count of one forward pass (convolutions and linear layers) for a
single input of ``input_shape``.

### `measure_latency(model: nn.Module, input_shape: Tuple[int, ...], device: str='cpu', n_warmup: int=10, n_iters: int=50) -> Dict[str, float]`

Forward latency for one image: mean, p50, p95 and std in milliseconds, after warm-
up.

## `afnn/augment.py`

Batch-level augmentation: Mixup / CutMix and a fast GPU crop+flip.

### class `MixupCutMix(mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5, enable_mixup=True, enable_cutmix=True)`

Randomly applies Mixup or CutMix to a batch.

Usage: ``x, y_a, y_b, lam = mixup(x, y)``. When nothing is applied, ``y_b`` and
``lam`` are None.

#### `__call__(self, x, y)`

Mix the batch with probability ``prob``; pick Mixup or CutMix at random among
the enabled ones.

#### `_torch_beta(alpha: float, device) -> float` *(internal)*

Sample from Beta(alpha, alpha) using torch's RNG (so ``torch.manual_seed`` makes
runs reproducible).

#### `_mixup(self, x, y)` *(internal)*

Blend each image with a shuffled partner: ``lam * x + (1 - lam) * x_shuffled``.

#### `_cutmix(self, x, y)` *(internal)*

Paste a random box from a shuffled partner and correct ``lam`` to the real
pasted area.

### `gpu_augment(x: torch.Tensor) -> torch.Tensor`

Random horizontal flip and random shifted crop for a whole batch on the GPU.

## `afnn/data.py`

Image-folder loading with a one-off uint8 cache that lives on the GPU.

Every image is resized once to ``size x size`` and stored in a single ``.npy`` array.
Training then never touches the disk or the JPEG decoder again: batches are sliced
straight out of a GPU tensor.

### `scan_image_folder(root) -> Tuple[List[Sample], List[str]]`

List ``(path, class_index)`` pairs for a ``root/<class>/<image>`` tree.

Classes are the sub-directory names in sorted order, so the index of a
class is stable between runs.

### `open_rgb(path) -> Image.Image`

Open any image as RGB.

Pillow is tried first. If it cannot decode the file (typically AVIF/HEIC
with a Pillow build that lacks the codec: "No codec available") the
fallbacks are pillow-heif, OpenCV and finally the ``ffmpeg`` binary.

### `load_image_array(path, size: int) -> np.ndarray`

Load one image as a ``uint8`` array of shape ``[size, size, 3]``.

### `_cache_key(samples: List[Sample], size: int) -> str` *(internal)*

Hash of the sample list (paths, sizes, mtimes, labels) and target size.

### `build_cache(samples: List[Sample], size: int, cache_dir=DEFAULT_CACHE_DIR, progress: Optional[Callable[[int, int, bool], None]]=None, cancel=None) -> Tuple[np.ndarray, np.ndarray, List[str]]`

Convert every image to a small array once and store it on disk.

Returns ``(X, y, failed_paths)`` with ``X`` of shape ``[N, size, size, 3]``
(uint8) and ``y`` of shape ``[N]`` (int64). A second call with the same
images and size loads the arrays from disk instead of decoding again.

``progress(done, total, from_cache)`` is called while working and
``cancel`` may be a ``threading.Event`` used to abort.

### class `CachedImageStore(x_u8: np.ndarray, y: np.ndarray, device: str)`

All cached images in one place: on the GPU if they fit, else in RAM.

### class `CachedImageLoader(store: CachedImageStore, indices: torch.Tensor, batch_size: int, shuffle: bool, augment: bool)`

Minimal DataLoader replacement: ``len()`` and iteration yield ``(x, y)``.

Images are returned as float tensors in ``[0, 1]`` with shape ``[B, 3, S, S]``.

#### `__len__(self) -> int`

Number of batches per epoch.

#### `__iter__(self)`

Yield ``(images, labels)`` batches, shuffled and augmented when configured.

### `split_indices(n: int, val_fraction: float, seed: int=42)`

Deterministic train/validation split. Returns ``(train_idx, val_idx)``.

### `save_arrays(X: np.ndarray, y: np.ndarray, classes: List[str], path) -> None`

Write the resized images to one compressed ``.npz`` file.

The file is small (about ``N * size * size * 3`` bytes before compression) and can be
copied to another machine, e.g. a cloud GPU, instead of the original images.

### `load_arrays(path) -> Tuple[np.ndarray, np.ndarray, List[str]]`

Read a file written by ``save_arrays``; returns ``(X, y, class_names)``.

### `pack_dataset(root, size: int, out_path, cache_dir=DEFAULT_CACHE_DIR, progress=None)`

Resize every image of ``root`` once and store the result as a single ``.npz``.

Returns the list of unreadable files that were skipped.

### `make_loaders(root, size: int, batch_size: int, val_fraction: float, device: str, cache_dir=DEFAULT_CACHE_DIR, progress=None, cancel=None)`

Build the train/validation loaders from an image folder or a packed ``.npz``.

For a folder the images are scanned and cached; for an ``.npz`` (see ``pack_dataset``)
the arrays are loaded directly and ``size`` is taken from the file.
Returns ``(train_loader, val_loader, class_names, failed_paths)``.

## `afnn/training.py`

Training utilities: EMA, growth policy, LR schedule and the Trainer.

### class `TrainingStopped()`

Raised inside ``Trainer.step_epoch`` when the stop event is set.

### class `EMA(model: nn.Module, decay: float=0.999)`

Exponential moving average of the model state dict.

The shadow copy follows growth and pruning because its keys are synchronised on
every update.

#### `sync_model(self, model: nn.Module, initialize: bool=False)`

Add shadow entries for new parameters/buffers and drop the ones that were
pruned.

#### `update(self, model: nn.Module, steps: int=1)`

Blend the current weights into the shadow copy. ``steps > 1`` applies ``decay **
steps`` in one go, to update every few batches.

#### `apply_to(self, model: nn.Module)`

Load the EMA weights into ``model``.

#### `reset_parameters(self, model: nn.Module, params: List[nn.Parameter])`

Overwrite the shadow entries of ``params`` with the model's current values (used
after re-initialisation).

#### `state_dict(self)`

Return ``{'decay', 'shadow'}`` for checkpointing.

#### `load_state_dict(self, state)`

Restore decay and shadow weights from a checkpoint.

### class `GrowthManager(model: AFNN, cfg: AFNNConfig)`

Decides when the network grows: after ``growth_patience`` epochs without a
validation-loss improvement of at least ``growth_min_delta``.

#### `observe(self, val_loss: float) -> bool`

Record a validation loss and return True when the model should grow now.

#### `grow(self, epoch: int) -> int`

Grow the model, log the event and return the number of branches added.

### class `DynamicLRScheduler(optimizer, warmup_epochs: int, total_epochs: int, base_lr: float)`

Warmup + cosine schedule that also handles parameter groups added later (new
branches get their own short warmup and a higher LR).

#### `factor(self, epoch: int) -> float`

Schedule multiplier for ``epoch``: linear warmup, then cosine decay.

#### `step(self, epoch: Optional[int]=None)`

Set the LR of every param group for ``epoch`` from its frozen base LR.

#### `state_dict(self)`

Scheduler state for checkpointing.

#### `load_state_dict(self, state)`

Restore the scheduler state.

### class `Trainer(model: AFNN, optimizer, scheduler, cfg: AFNNConfig, device: str='cpu', grad_clip: float=1.0)`

Training loop and state management for AFNN.

Handles AMP (switched off automatically on GPUs older than Volta), gradient
clipping, Mixup/CutMix, EMA, the load-balancing loss, the optional contrastive
representation loss, growth, rejuvenation and checkpoints.

#### `_ensure_device(self)` *(internal)*

Make sure the model, optimizer state and EMA all live on ``self.device``. Cheap;
called once per epoch.

#### `_sync_optimizer(self, epoch: int) -> int` *(internal)*

Drop pruned parameters from the optimizer and put new ones in a fresh param
group with a higher LR. Returns the number of new parameters.

#### `synchronize_dynamic_state(self, epoch: int=0) -> Dict[str, int]`

Sync optimizer and EMA after growth or pruning done outside ``step_epoch``.

#### `_clear_optimizer_state(self, params: List[nn.Parameter]) -> int` *(internal)*

Delete the optimizer state (momentum) of the given parameters; returns how many
were cleared.

#### `_augment_view(x: torch.Tensor) -> torch.Tensor` *(internal)*

Random flip, crop-and-resize and brightness/contrast jitter for one contrastive
view.

#### `_nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor` *(internal)*

SimCLR-style symmetric NT-Xent loss between two views.

#### `_supervised_contrastive(z: torch.Tensor, y: torch.Tensor, temperature: float) -> torch.Tensor` *(internal)*

Pull same-class embeddings together and push different classes apart.

#### `_representation_loss(self, x: torch.Tensor, y: torch.Tensor, epoch: int) -> torch.Tensor` *(internal)*

Contrastive + invariance loss on two augmented views of the un-mixed batch (zero
during the warm-up epochs).

#### `_compute_loss(self, logits, y_a, y_b, lam)` *(internal)*

Cross-entropy with label smoothing; handles the two-target case of Mixup/CutMix.

#### `_load_balance_loss(self)` *(internal)*

Penalise merge weights that drift away from a uniform split, which prevents one
branch from taking over (routing collapse).

#### `train_epoch(self, loader, epoch: int=0, progress_callback=None, stop_event=None)`

Train for one epoch and return ``{'loss', 'acc', 'mixed_acc'}``. ``acc`` is
measured on un-mixed batches only.

#### `evaluate(self, loader, progress_callback=None, stop_event=None)`

Return ``{'loss', 'acc'}`` on ``loader`` without gradients.

#### `step_epoch(self, train_loader, val_loader, epoch: int, progress_callback=None, stop_event=None) -> Dict[str, Any]`

Run one full epoch: LR schedule, training, validation, rejuvenation of dead
branches and growth when validation stalls. Returns a dict describing the epoch.

#### `save_checkpoint(self, path: str, **extra)`

Write model, optimizer, scheduler, EMA, growth state and config to ``path``.
Extra keyword arguments (for example ``class_names``) are stored as-is.

#### `load_training_state(self, ckpt: Dict[str, Any])`

Restore optimizer, scheduler, growth manager, EMA and grad scaler from a
checkpoint.

#### `load_checkpoint(path: str, device: str='cpu')`

Rebuild the model (including grown branches) from a checkpoint. Returns
``(model, cfg, ckpt)``.

### `build_trainer(num_classes: int, size: int, *, channels: int=28, depth: int=2, branches: int=3, max_branches: int=5, norm: str='batchnorm', act: str='relu', load_balance: bool=True, representation: bool=False, lr: float=0.002, epochs: int=20, device: Optional[str]=None) -> Trainer`

Create a model, optimizer, scheduler and Trainer from the usual settings.

Fixed choices: 2 stages, channel multipliers 1x/2x, Mixup + CutMix, label smoothing
0.05, EMA, AdamW with weight decay 1e-4, growth up to 10 M parameters. The default
sizes are a small starting point; raise ``channels`` and ``depth`` for larger
networks. ``device`` defaults to CUDA when available.

### `fit(trainer: Trainer, train_loader, val_loader, *, start_epoch: int=0, num_epochs: int=20, history: Optional[List[Dict[str, Any]]]=None, class_names: Optional[List[str]]=None, checkpoint_path=None, progress=None, on_epoch=None, stop_event=None) -> List[Dict[str, Any]]`

Train from ``start_epoch`` up to (excluding) ``num_epochs``.

After every epoch the metrics are appended to ``history`` and, if
``checkpoint_path`` is given, a checkpoint is written atomically (temporary
file, then rename), so a power cut never leaves a corrupt file. The
checkpoint also stores ``history``, ``next_epoch`` and ``class_names`` so a
later run can resume.

``progress(kind, done, total, elapsed)`` is forwarded to the Trainer and
``on_epoch(info, history)`` is called after every epoch. Setting
``stop_event`` (a ``threading.Event``) stops training between batches.
Returns the history list.

### `resume_trainer(path, device: Optional[str]=None, lr: float=0.002)`

Rebuild a Trainer from a checkpoint so training can continue.

Returns ``(trainer, ckpt)``. ``ckpt["next_epoch"]``, ``ckpt["history"]`` and
``ckpt["class_names"]`` (when present) are what ``fit`` needs to carry on.

### `normalize_history(ckpt: Dict[str, Any]) -> Dict[str, Any]`

Make sure ``ckpt["history"]`` (list of per-epoch dicts) and ``ckpt["next_epoch"]``
exist.

    Checkpoints written by the first version of the Studio kept the curves as a dict of
    lists under ``studio_state``; they are converted here. Modifies and returns ``ckpt``.

## `afnn/inference.py`

Load a trained checkpoint and classify single images.

### `load_model(path, device: Optional[str]=None)`

Load a checkpoint written by ``Trainer.save_checkpoint``.

Returns ``(model, cfg, class_names)``; ``class_names`` is empty if the
checkpoint does not contain them.

### `preprocess(image: Image.Image, size: int) -> torch.Tensor`

Resize to ``size x size`` and convert to a ``[1, 3, size, size]`` float tensor.

### `predict(model, cfg, image_path, class_names: Optional[List[str]]=None, top_k: int=5) -> Tuple[Image.Image, List[Tuple[str, float]]]`

Classify one image file.

Returns the opened image and the ``top_k`` predictions as ``(label, probability)``.

## `gui/app.py`

Tkinter front end for AFNN: train, watch the curves, test images.

Run from the project root:  python -m gui.app

### `format_seconds(seconds: float) -> str`

Format a duration as ``1h 05m``, ``12m 30s`` or ``45s``.

### class `App()`

Main window. Training runs in a worker thread and reports through a queue.

#### `_build_settings_panel(self)` *(internal)*

Left column: dataset picker, hyper-parameters and the control buttons.

#### `_build_tabs(self)` *(internal)*

Right side: training, image test and model tabs.

#### `_load_settings(self)` *(internal)*

Restore the last used hyper-parameters (and dataset folder).

#### `_save_settings(self)` *(internal)*

Persist the current hyper-parameters (best effort).

#### `_read_settings(self) -> dict` *(internal)*

Parse and validate the form; raises ValueError with a readable message.

#### `choose_dataset(self)`

Ask for the dataset root and show how many classes/images it holds.

#### `_set_dataset(self, path: Path)` *(internal)*

Remember the dataset folder and describe it in the side panel.

#### `start_training(self, resume: bool)`

Validate the form and start the worker thread.

#### `_train_worker(self, s: dict, resume: bool)` *(internal)*

Worker thread: build loaders and the trainer, then run ``fit``.

#### `stop_training(self)`

Ask the worker to stop after the current batch.

#### `_set_running(self, running: bool)` *(internal)*

Enable/disable the buttons depending on whether training is active.

#### `_poll_events(self)` *(internal)*

Apply queued worker messages to the widgets (runs on the Tk thread).

#### `_on_log(self, text)` *(internal)*

Event handler: append a message to the log.

#### `_on_cache(self, payload)` *(internal)*

Event handler: show image-preparation progress.

#### `_on_progress(self, payload)` *(internal)*

Event handler: show batch progress, speed and ETA of the running epoch.

#### `_on_epoch(self, payload)` *(internal)*

Event handler: log the finished epoch, update labels and redraw the charts.

#### `_on_model(self, _payload)` *(internal)*

Event handler: refresh the Model tab.

#### `_on_done(self, _payload)` *(internal)*

Event handler: training finished or was stopped.

#### `_on_error(self, payload)` *(internal)*

Event handler: show a worker exception.

#### `_append_log(self, text: str)` *(internal)*

Add text to the read-only log box.

#### `_finite(v)` *(internal)*

True for real numbers (not NaN or infinity).

#### `_plot(self, canvas, values, color, lo, hi, pad=10)` *(internal)*

Draw one series (skipping NaN/inf) scaled to ``[lo, hi]``.

#### `_redraw(self)` *(internal)*

Redraw both charts and the memorisation verdict from ``self.history``.

#### `_refresh_model_tab(self)` *(internal)*

Show architecture and progress information in the Model tab.

#### `save_checkpoint(self)`

Save the current model and training state to a chosen file.

#### `load_checkpoint(self)`

Load a checkpoint and restore model, history and settings.

#### `test_image(self)`

Classify a user-chosen image with the current model.

#### `_on_close(self, tries: int=0)` *(internal)*

Stop the worker (if any) and close the window.

### `main()`

Open the application window.

## `scripts/train.py`

Train AFNN on a folder of images (one sub-folder per class), without the GUI.

Example:
    python scripts/train.py --data ~/datasets/animals --out runs/animals.pt
    python scripts/train.py --data ~/datasets/animals --out runs/animals.pt --resume runs/animals.pt --epochs 30

### `parse_args()`

Parse the command-line options.

### `main()`

Load the data, build or resume the trainer and run ``fit``.

## `scripts/predict.py`

Classify images with a trained checkpoint.

Example:
    python scripts/predict.py runs/animals.pt photo1.jpg photo2.png --top 3

### `main()`

Print the top predictions for every image given on the command line.
