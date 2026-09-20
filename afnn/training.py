"""Training utilities: EMA, growth policy, LR schedule and the Trainer."""
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .augment import MixupCutMix
from .config import AFNNConfig
from .layers import WeightedMerge
from .model import AFNN


class TrainingStopped(RuntimeError):
    """Raised inside ``Trainer.step_epoch`` when the stop event is set."""


class EMA:
    """Exponential moving average of the model state dict.

            The shadow copy follows growth and pruning because its keys are synchronised on
            every update.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.sync_model(model, initialize=True)

    @torch.no_grad()
    def sync_model(self, model: nn.Module, initialize: bool = False):
        """Add shadow entries for new parameters/buffers and drop the ones that were
                        pruned.
        """
        current = model.state_dict()
        current_keys = set(current)
        for k in list(self.shadow):
            if k not in current_keys:
                del self.shadow[k]
        for k, v in current.items():
            if initialize or k not in self.shadow:
                self.shadow[k] = v.detach().clone()
            elif not self.shadow[k].is_floating_point() or not v.is_floating_point():
                self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module, steps: int = 1):
        """Blend the current weights into the shadow copy. ``steps > 1`` applies ``decay **
                        steps`` in one go, to update every few batches.
        """
        self.sync_model(model)
        d = self.decay ** max(1, int(steps))
        sh_f, m_f = [], []
        for k, v in model.state_dict().items():
            s_t = self.shadow[k]
            if s_t.dtype.is_floating_point and v.dtype.is_floating_point:
                sh_f.append(s_t)
                m_f.append(v.detach())
            else:
                self.shadow[k] = v.detach().clone()
        if not sh_f:
            return
        try:
            torch._foreach_mul_(sh_f, d)
            torch._foreach_add_(sh_f, m_f, alpha=1 - d)
        except Exception:
            for s_t, m_t in zip(sh_f, m_f):
                s_t.mul_(d).add_(m_t, alpha=1 - d)

    def apply_to(self, model: nn.Module):
        """Load the EMA weights into ``model``."""
        self.sync_model(model)
        model.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def reset_parameters(self, model: nn.Module, params: List[nn.Parameter]):
        """Overwrite the shadow entries of ``params`` with the model's current values (used
                        after re-initialisation).
        """
        ids = {id(p) for p in params}
        for name, p in model.named_parameters():
            if id(p) in ids:
                self.shadow[name] = p.detach().clone()

    def state_dict(self):
        """Return ``{'decay', 'shadow'}`` for checkpointing."""
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        """Restore decay and shadow weights from a checkpoint."""
        self.decay = state["decay"]
        self.shadow = state["shadow"]


class GrowthManager:
    """Decides when the network grows: after ``growth_patience`` epochs without a
            validation-loss improvement of at least ``growth_min_delta``.
    """
    def __init__(self, model: AFNN, cfg: AFNNConfig):
        self.model = model
        self.cfg = cfg
        self.best_loss = float("inf")
        self.waiting = 0
        self.growth_events: List[Dict[str, Any]] = []

    def observe(self, val_loss: float) -> bool:
        """Record a validation loss and return True when the model should grow now."""
        if not self.cfg.growth_enabled:
            return False
        if val_loss < self.best_loss - self.cfg.growth_min_delta:
            self.best_loss = val_loss
            self.waiting = 0
            return False
        self.waiting += 1
        return self.waiting >= self.cfg.growth_patience

    def grow(self, epoch: int) -> int:
        """Grow the model, log the event and return the number of branches added."""
        p_before = self.model.count_parameters()
        b_before = self.model.count_total_branches()
        added = self.model.grow_one_branch_per_stage(
            self.cfg.max_branches_per_stage, self.cfg.max_total_params,
        )
        if added == 0:
            return 0
        p_after = self.model.count_parameters()
        b_after = self.model.count_total_branches()
        self.growth_events.append({
            "epoch": epoch,
            "branches_before": b_before, "branches_after": b_after,
            "branches_added": added,
            "params_before": p_before, "params_after": p_after,
            "params_added": p_after - p_before,
        })
        self.waiting = 0
        self.best_loss = float("inf")
        return added


class DynamicLRScheduler:
    """Warmup + cosine schedule that also handles parameter groups added later (new
            branches get their own short warmup and a higher LR).
    """
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, base_lr: float):
        self.optimizer = optimizer
        self.warmup_epochs = max(0, warmup_epochs)
        self.total_epochs = max(total_epochs, 1)
        self.base_lr = base_lr
        self.last_epoch = -1

    def factor(self, epoch: int) -> float:
        """Schedule multiplier for ``epoch``: linear warmup, then cosine decay."""
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            return (epoch + 1) / self.warmup_epochs
        p = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
        p = min(max(p, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    def step(self, epoch: Optional[int] = None):
        """Set the LR of every param group for ``epoch`` from its frozen base LR."""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        f = self.factor(epoch)
        warm = self.optimizer.defaults.get("_new_branch_warmup_epochs", 0)
        for g in self.optimizer.param_groups:
            # Freeze each group's base LR once; deriving it from the scaled LR would compound.
            if "_base_lr" not in g:
                g["_base_lr"] = float(g["lr"])
            base = float(g["_base_lr"])
            local = f
            if g.get("_new", False) and warm > 0:
                age = max(0, epoch - g.get("_new_epoch", epoch))
                if age < warm:
                    local *= (age + 1) / warm
            g["lr"] = base * local

    def state_dict(self):
        """Scheduler state for checkpointing."""
        return {"warmup_epochs": self.warmup_epochs, "total_epochs": self.total_epochs,
                "base_lr": self.base_lr, "last_epoch": self.last_epoch}

    def load_state_dict(self, state):
        """Restore the scheduler state."""
        self.warmup_epochs = state["warmup_epochs"]
        self.total_epochs = state["total_epochs"]
        self.base_lr = state["base_lr"]
        self.last_epoch = state.get("last_epoch", -1)


class Trainer:
    """Training loop and state management for AFNN.

            Handles AMP (switched off automatically on GPUs older than Volta), gradient
            clipping, Mixup/CutMix, EMA, the load-balancing loss, the optional contrastive
            representation loss, growth, rejuvenation and checkpoints.
    """
    def __init__(self, model: AFNN, optimizer, scheduler,
                 cfg: AFNNConfig, device: str = "cpu", grad_clip: float = 1.0):
        torch.backends.cudnn.benchmark = True     # input sizes are fixed
        self.model = model.to(device)
        self.opt = optimizer
        self.sched = scheduler
        self.cfg = cfg
        self.device = device
        self.grad_clip = grad_clip
        self.growth_manager = GrowthManager(model, cfg)
        self.mixup = MixupCutMix(
            cfg.mixup_alpha, cfg.cutmix_alpha,
            enable_mixup=cfg.use_mixup, enable_cutmix=cfg.use_cutmix,
        ) if (cfg.use_mixup or cfg.use_cutmix) else None
        self.ema = EMA(model, cfg.ema_decay) if cfg.use_ema else None
        self.amp_enabled = cfg.use_amp and device.startswith("cuda")
        if self.amp_enabled:
            try:
                if torch.cuda.get_device_capability(0)[0] < 7:
                    # Maxwell/Pascal (e.g. Quadro M1200) have no tensor cores and no
                    # fast FP16, so AMP only adds casts and syncs.
                    self.amp_enabled = False
                    print("AFNN: GPU older than Volta detected, AMP disabled.", file=sys.stderr)
            except Exception:
                pass
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.n_classes = cfg.num_classes

    def _ensure_device(self):
        """Make sure the model, optimizer state and EMA all live on ``self.device``. Cheap;
                        called once per epoch.
        """
        dev = torch.device(self.device)
        self.model.to(dev)
        for p, st in self.opt.state.items():
            for k, v in list(st.items()):
                if torch.is_tensor(v) and v.device.type != p.device.type:
                    st[k] = v.to(p.device)
        if self.ema is not None:
            for k, v in list(self.ema.shadow.items()):
                if torch.is_tensor(v) and v.device.type != dev.type:
                    self.ema.shadow[k] = v.to(dev)

    def _sync_optimizer(self, epoch: int) -> int:
        """Drop pruned parameters from the optimizer and put new ones in a fresh param
                        group with a higher LR. Returns the number of new parameters.
        """
        current = list(self.model.parameters())
        current_ids = {id(p) for p in current}

        # Remove physically pruned parameters and their optimizer state first.
        for g in self.opt.param_groups:
            stale = [p for p in g["params"] if id(p) not in current_ids]
            if stale:
                g["params"] = [p for p in g["params"] if id(p) in current_ids]
                for p in stale:
                    self.opt.state.pop(p, None)

        # The optimizer is the source of truth for which parameters it already holds.
        optimizer_ids = {id(p) for g in self.opt.param_groups for p in g["params"]}
        new_params = [p for p in current if id(p) not in optimizer_ids]
        if not new_params:
            return 0

        base_lr = self.opt.param_groups[0].get("_base_lr", self.opt.param_groups[0]["lr"])
        wd = self.opt.param_groups[0].get("weight_decay", 0.0)
        new_base = base_lr * self.cfg.new_branch_lr_multiplier
        self.opt.add_param_group({
            "params": new_params,
            "lr": new_base,
            "_base_lr": new_base,
            "weight_decay": wd,
            "_new": True,
            "_new_epoch": epoch,
        })
        return len(new_params)

    def synchronize_dynamic_state(self, epoch: int = 0) -> Dict[str, int]:
        """Sync optimizer and EMA after growth or pruning done outside ``step_epoch``."""
        n_new = self._sync_optimizer(epoch)
        if self.ema is not None:
            self.ema.sync_model(self.model)
        return {"new_optimizer_params": n_new,
                "optimizer_param_count": sum(len(g["params"]) for g in self.opt.param_groups),
                "ema_keys": len(self.ema.shadow) if self.ema is not None else 0}

    def _clear_optimizer_state(self, params: List[nn.Parameter]) -> int:
        """Delete the optimizer state (momentum) of the given parameters; returns how many
                        were cleared.
        """
        cleared = 0
        for p in params:
            if p in self.opt.state:
                del self.opt.state[p]
                cleared += 1
        return cleared

    @staticmethod
    def _augment_view(x: torch.Tensor) -> torch.Tensor:
        """Random flip, crop-and-resize and brightness/contrast jitter for one contrastive
                        view.
        """
        b, c, h, w = x.shape
        v = x

        # Batch-wise horizontal flip.
        mask = torch.rand(b, device=x.device) < 0.5
        if mask.any():
            v = torch.where(mask[:, None, None, None], torch.flip(v, dims=[3]), v)

        # Random resized crop implemented with one batched grid_sample call.
        # Each sample gets an independent scale and center, but no Python loop.
        if h >= 8 and w >= 8:
            scale = torch.empty(b, device=x.device).uniform_(0.80, 1.0)
            max_cx = 1.0 - scale
            max_cy = 1.0 - scale
            cx = (torch.rand(b, device=x.device) * 2.0 - 1.0) * max_cx
            cy = (torch.rand(b, device=x.device) * 2.0 - 1.0) * max_cy
            theta = torch.zeros(b, 2, 3, device=x.device, dtype=v.dtype)
            theta[:, 0, 0] = 1.0 / scale
            theta[:, 1, 1] = 1.0 / scale
            theta[:, 0, 2] = cx
            theta[:, 1, 2] = cy
            grid = F.affine_grid(theta, size=v.shape, align_corners=False)
            v = F.grid_sample(v, grid, mode="bilinear", padding_mode="border",
                              align_corners=False)

        contrast = torch.empty(b, 1, 1, 1, device=x.device, dtype=v.dtype).uniform_(0.85, 1.15)
        brightness = torch.empty(b, 1, 1, 1, device=x.device, dtype=v.dtype).uniform_(-0.10, 0.10)
        v = v * contrast + brightness
        v = v + 0.02 * torch.randn_like(v)
        return v

    @staticmethod
    def _nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
        """SimCLR-style symmetric NT-Xent loss between two views."""
        n = z1.size(0)
        if n < 2:
            return torch.zeros((), device=z1.device, dtype=z1.dtype)
        z = torch.cat([F.normalize(z1, dim=1), F.normalize(z2, dim=1)], dim=0)
        logits = torch.mm(z, z.t()) / temperature
        eye = torch.eye(2 * n, device=z.device, dtype=torch.bool)
        logits = logits.masked_fill(eye, float("-inf"))
        targets = torch.arange(2 * n, device=z.device)
        targets = (targets + n) % (2 * n)
        return F.cross_entropy(logits, targets)

    @staticmethod
    def _supervised_contrastive(z: torch.Tensor, y: torch.Tensor, temperature: float) -> torch.Tensor:
        """Pull same-class embeddings together and push different classes apart."""
        n = z.size(0)
        if n < 2:
            return torch.zeros((), device=z.device, dtype=z.dtype)
        z = F.normalize(z, dim=1)
        sim = torch.mm(z, z.t()) / temperature
        eye = torch.eye(n, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(eye, float("-inf"))
        positive = y[:, None].eq(y[None, :]) & (~eye)
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        denom = positive.sum(dim=1)
        valid = denom > 0
        if not valid.any():
            return torch.zeros((), device=z.device, dtype=z.dtype)
        pos_log_prob = torch.where(positive, log_prob, torch.zeros_like(log_prob))
        mean_log_prob_pos = pos_log_prob.sum(dim=1) / denom.clamp_min(1)
        return -mean_log_prob_pos[valid].mean()

    def _representation_loss(self, x: torch.Tensor, y: torch.Tensor, epoch: int) -> torch.Tensor:
        """Contrastive + invariance loss on two augmented views of the un-mixed batch (zero
                        during the warm-up epochs).
        """
        if not self.cfg.use_representation_learning:
            return torch.zeros((), device=self.device)
        if epoch < self.cfg.representation_warmup_epochs:
            # Warm the classifier/backbone before adding the extra objective.
            return torch.zeros((), device=self.device)
        x1 = self._augment_view(x)
        x2 = self._augment_view(x)
        z1 = self.model.project_representation(x1)
        z2 = self.model.project_representation(x2)
        # Supervised contrastive term always; the instance-level SimCLR term only
        # when the batch has several classes (it would push same-class images apart).
        sup = 0.5 * (
            self._supervised_contrastive(z1, y, self.cfg.contrastive_temperature) +
            self._supervised_contrastive(z2, y, self.cfg.contrastive_temperature)
        )
        if torch.unique(y).numel() > 1:
            contrast = self._nt_xent(z1, z2, self.cfg.contrastive_temperature)
            concept_loss = 0.70 * sup + 0.30 * contrast
        else:
            concept_loss = sup
        invariance = 1.0 - F.cosine_similarity(z1, z2, dim=1).mean()
        return (self.cfg.contrastive_weight * concept_loss +
                self.cfg.invariance_weight * invariance)

    def _compute_loss(self, logits, y_a, y_b, lam):
        """Cross-entropy with label smoothing; handles the two-target case of Mixup/CutMix."""
        if y_b is not None:
            target_probs = (
                lam * F.one_hot(y_a, self.n_classes).float() +
                (1 - lam) * F.one_hot(y_b, self.n_classes).float()
            )
            return F.cross_entropy(
                logits, target_probs,
                label_smoothing=self.cfg.label_smoothing,
            )
        return F.cross_entropy(
            logits, y_a,
            label_smoothing=self.cfg.label_smoothing,
        )

    def _load_balance_loss(self):
        """Penalise merge weights that drift away from a uniform split, which prevents one
                        branch from taking over (routing collapse).
        """
        if not self.cfg.use_load_balance:
            return torch.tensor(0.0, device=self.device)
        total_loss = torch.tensor(0.0, device=self.device)
        n_modules = 0
        for stage in self.model.stages:
            for module in stage.fractal.modules():
                if isinstance(module, WeightedMerge):
                    w = module.weights()  # (N,)
                    uniform = torch.ones_like(w) / module.num_branches
                    total_loss = total_loss + F.mse_loss(w, uniform)
                    n_modules += 1
        if n_modules > 0:
            total_loss = total_loss / n_modules
        return total_loss * self.cfg.load_balance_weight

    def train_epoch(self, loader, epoch: int = 0, progress_callback=None, stop_event=None):
        """Train for one epoch and return ``{'loss', 'acc', 'mixed_acc'}``. ``acc`` is
                        measured on un-mixed batches only.
        """
        self._ensure_device()
        self.model.train()
        tot_loss = tot = corr = clean_tot = clean_corr = mixed_tot = 0
        started = time.time()
        total_batches = len(loader)
        for batch_idx, (x, y) in enumerate(loader):
            if stop_event is not None and stop_event.is_set():
                break
            x, y = x.to(self.device), y.to(self.device)
            if self.cfg.rejuvenation_enabled and (
                    batch_idx % self.cfg.rejuvenation_ema_stride == 0):
                self.model.update_all_branch_emas(x)
            x_orig, y_orig = x, y
            if self.mixup is not None:
                x, y_a, y_b, lam = self.mixup(x, y)
            else:
                y_a, y_b, lam = y, None, 1.0
            self.opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                logits = self.model(x)
                loss = self._compute_loss(logits, y_a, y_b, lam)
                loss = loss + self._load_balance_loss()
                # Representation learning uses the original, un-mixed batch.
                if self.cfg.use_representation_learning:
                    loss = loss + self._representation_loss(x_orig, y_orig, epoch)
            if self.amp_enabled:
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.opt)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip)
                self.opt.step()
            if self.ema is not None and (batch_idx % 4 == 3):
                self.ema.update(self.model, steps=4)
            # Accumulate on the GPU; every .item() would force a CPU<->GPU sync.
            tot_loss = tot_loss + loss.detach() * x.size(0)
            if y_b is None:
                clean_corr = clean_corr + (logits.argmax(1) == y_orig).sum()
                clean_tot += x_orig.size(0)
            else:
                # Accuracy on mixed batches is tracked separately from clean accuracy.
                pred = logits.argmax(1)
                corr = corr + (lam * (pred == y_a).float() + (1 - lam) * (pred == y_b).float()).sum()
                mixed_tot += x.size(0)
            tot += x.size(0)
            if progress_callback is not None:
                progress_callback("train", batch_idx + 1, total_batches, time.time() - started)
        if tot == 0:
            raise ValueError("train_epoch received an empty loader")
        tot_loss = float(tot_loss)
        clean_corr = float(clean_corr)
        corr = float(corr)
        acc = (clean_corr / clean_tot) if clean_tot else float("nan")
        mixed_acc = (corr / mixed_tot) if mixed_tot else None
        return {"loss": tot_loss / tot, "acc": acc, "mixed_acc": mixed_acc}

    @torch.no_grad()
    def evaluate(self, loader, progress_callback=None, stop_event=None):
        """Return ``{'loss', 'acc'}`` on ``loader`` without gradients."""
        self._ensure_device()
        self.model.eval()
        tot_loss = tot = corr = 0
        started = time.time()
        total_batches = len(loader)
        for batch_idx, (x, y) in enumerate(loader):
            if stop_event is not None and stop_event.is_set():
                break
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = F.cross_entropy(
                logits, y, label_smoothing=self.cfg.label_smoothing)
            tot_loss += loss.item() * x.size(0)
            corr += (logits.argmax(1) == y).sum().item()
            tot += x.size(0)
            if progress_callback is not None:
                progress_callback("val", batch_idx + 1, total_batches, time.time() - started)
        if tot == 0:
            raise ValueError("evaluate received an empty loader")
        return {"loss": tot_loss / tot, "acc": corr / tot}

    def step_epoch(self, train_loader, val_loader, epoch: int, progress_callback=None, stop_event=None) -> Dict[str, Any]:
        """Run one full epoch: LR schedule, training, validation, rejuvenation of dead
                        branches and growth when validation stalls. Returns a dict describing the epoch.
        """
        if self.sched is not None:
            self.sched.step(epoch)
        tr = self.train_epoch(train_loader, epoch, progress_callback=progress_callback, stop_event=stop_event)
        if stop_event is not None and stop_event.is_set():
            raise TrainingStopped("Training was stopped by the user.")
        vl = self.evaluate(val_loader, progress_callback=progress_callback, stop_event=stop_event)
        info = {
            "epoch": epoch,
            "train_loss": tr["loss"], "train_acc": tr["acc"],
            "val_loss": vl["loss"], "val_acc": vl["acc"],
            "params": self.model.count_parameters(),
            "branches_per_stage": self.model.count_branches_per_stage(),
            "total_branches": self.model.count_total_branches(),
            "grew": False, "rejuvenated": 0, "opt_state_cleared": 0,
        }
        if (self.cfg.rejuvenation_enabled and
                (epoch + 1) % self.cfg.rejuvenation_check_every == 0):
            n_revived, reinit_params = self.model.rejuvenate_dead_branches(
                self.cfg.dead_branch_std_threshold,
                self.cfg.rejuvenation_min_observations,
            )
            if n_revived > 0:
                cleared = self._clear_optimizer_state(reinit_params)
                if self.ema is not None:
                    self.ema.reset_parameters(self.model, reinit_params)
                info["rejuvenated"] = n_revived
                info["opt_state_cleared"] = cleared
        if self.growth_manager.observe(vl["loss"]):
            added = self.growth_manager.grow(epoch)
            if added > 0:
                self.model.to(self.device)    
                n_new = self._sync_optimizer(epoch)
                if self.ema is not None:
                    self.ema.sync_model(self.model)
                info["grew"] = True
                info["branches_added"] = added
                info["params_after_growth"] = self.model.count_parameters()
                info["new_params_in_optimizer"] = n_new
        if self.sched is not None:
            self.sched.step(epoch)  # apply current factor to newly-added groups too
        return info

    def save_checkpoint(self, path: str, **extra):
        """Write model, optimizer, scheduler, EMA, growth state and config to ``path``.
                        Extra keyword arguments (for example ``class_names``) are stored as-is.
        """
        state = {
            "arch_version": self.cfg.architecture_version,
            "config": self.cfg.to_dict(),
            "architecture": {"root_branches": self.model.count_branches_per_stage()},
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "optimizer_param_names": [
                [name for name, p in self.model.named_parameters()
                 if any(id(p) == id(q) for q in group["params"])]
                for group in self.opt.param_groups
            ],
            "scheduler": self.sched.state_dict() if self.sched else None,
            "growth_manager": {
                "best_loss": self.growth_manager.best_loss,
                "waiting": self.growth_manager.waiting,
                "growth_events": self.growth_manager.growth_events,
            },
            "trainer": {},
            **extra,
        }
        if self.ema is not None:
            state["ema"] = self.ema.state_dict()
        if self.amp_enabled:
            state["scaler"] = self.scaler.state_dict()
        torch.save(state, path)

    def load_training_state(self, ckpt: Dict[str, Any]):
        """Restore optimizer, scheduler, growth manager, EMA and grad scaler from a
                        checkpoint.
        """
        saved_opt = ckpt["optimizer"]
        name_to_param = dict(self.model.named_parameters())
        saved_names = ckpt.get("optimizer_param_names")

        self.opt.param_groups.clear()
        self.opt.state.clear()
        if saved_names is not None:
            if len(saved_names) != len(saved_opt["param_groups"]):
                raise RuntimeError("Checkpoint optimizer groups/names mismatch")
            for saved_group, names in zip(saved_opt["param_groups"], saved_names):
                if len(names) != len(saved_group["params"]):
                    raise RuntimeError("Checkpoint optimizer parameter-name count mismatch")
                try:
                    params = [name_to_param[n] for n in names]
                except KeyError as e:
                    raise RuntimeError(f"Missing model parameter in checkpoint: {e.args[0]}") from e
                group = {k: v for k, v in saved_group.items() if k != "params"}
                group["params"] = params
                self.opt.add_param_group(group)
        else:
            # Older checkpoints store parameter indices instead of names.
            params = list(self.model.parameters())
            for saved_group in saved_opt["param_groups"]:
                ids = saved_group["params"]
                if any(i >= len(params) for i in ids):
                    raise RuntimeError("Legacy checkpoint parameter indices do not match model")
                group = {k: v for k, v in saved_group.items() if k != "params"}
                group["params"] = [params[i] for i in ids]
                self.opt.add_param_group(group)
        self.opt.load_state_dict(saved_opt)
        self.opt.defaults["_new_branch_warmup_epochs"] = self.cfg.new_branch_warmup_epochs
        if self.sched is not None and ckpt.get("scheduler") is not None:
            self.sched.load_state_dict(ckpt["scheduler"])
        gm = ckpt.get("growth_manager", {})
        self.growth_manager.best_loss = gm.get("best_loss", float("inf"))
        self.growth_manager.waiting = gm.get("waiting", 0)
        self.growth_manager.growth_events = gm.get("growth_events", [])
        if self.ema is not None and ckpt.get("ema") is not None:
            self.ema.load_state_dict(ckpt["ema"])
            self.ema.sync_model(self.model)
        if self.amp_enabled and ckpt.get("scaler") is not None:
            self.scaler.load_state_dict(ckpt["scaler"])

    @staticmethod
    def load_checkpoint(path: str, device: str = "cpu"):
        """Rebuild the model (including grown branches) from a checkpoint. Returns
                        ``(model, cfg, ckpt)``.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = AFNNConfig.from_dict(ckpt["config"])
        model = AFNN(cfg).to(device)
        target = ckpt.get("architecture", {}).get("root_branches", [cfg.initial_branches] * cfg.stages)
        for stage, target_branches in zip(model.stages, target):
            while stage.fractal.branches < target_branches:
                if not stage.fractal.add_branch(cfg.max_branches_per_stage):
                    raise RuntimeError("Could not reconstruct checkpoint architecture")
        model.load_state_dict(ckpt["model"])
        return model, cfg, ckpt


def build_trainer(num_classes: int, size: int, *, channels: int = 28, depth: int = 2,
                  branches: int = 3, max_branches: int = 5, norm: str = "batchnorm",
                  act: str = "relu", load_balance: bool = True,
                  representation: bool = False, lr: float = 2e-3, epochs: int = 20,
                  device: Optional[str] = None) -> Trainer:
    """Create a model, optimizer, scheduler and Trainer from the usual settings.

    Fixed choices: 2 stages, channel multipliers 1x/2x, Mixup + CutMix, label smoothing
    0.05, EMA, AdamW with weight decay 1e-4, growth up to 10 M parameters. The default
    sizes are a small starting point; raise ``channels`` and ``depth`` for larger
    networks. ``device`` defaults to CUDA when available.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = AFNNConfig(
        input_channels=3, input_size=size, num_classes=num_classes,
        base_channels=channels, channel_multipliers=(1.0, 2.0), stages=2,
        initial_depth=depth, initial_branches=branches,
        max_branches_per_stage=max(max_branches, branches),
        max_total_params=10_000_000, normalization=norm, activation=act,
        use_mixup=True, use_cutmix=True, label_smoothing=0.05,
        use_amp=True, use_ema=True, use_load_balance=load_balance,
        use_representation_learning=representation,
    )
    model = AFNN(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    opt.defaults["_new_branch_warmup_epochs"] = cfg.new_branch_warmup_epochs
    for group in opt.param_groups:
        group["_base_lr"] = group["lr"]
    sched = DynamicLRScheduler(opt, cfg.warmup_epochs, max(int(epochs), 1), lr)
    return Trainer(model, opt, sched, cfg, device=device, grad_clip=1.0)


def fit(trainer: Trainer, train_loader, val_loader, *, start_epoch: int = 0,
        num_epochs: int = 20, history: Optional[List[Dict[str, Any]]] = None,
        class_names: Optional[List[str]] = None, checkpoint_path=None,
        progress=None, on_epoch=None, stop_event=None) -> List[Dict[str, Any]]:
    """Train from ``start_epoch`` up to (excluding) ``num_epochs``.

            After every epoch the metrics are appended to ``history`` and, if
            ``checkpoint_path`` is given, a checkpoint is written atomically (temporary
            file, then rename), so a power cut never leaves a corrupt file. The
            checkpoint also stores ``history``, ``next_epoch`` and ``class_names`` so a
            later run can resume.

            ``progress(kind, done, total, elapsed)`` is forwarded to the Trainer and
            ``on_epoch(info, history)`` is called after every epoch. Setting
            ``stop_event`` (a ``threading.Event``) stops training between batches.
            Returns the history list.
    """
    history = history if history is not None else []
    if trainer.sched is not None:
        trainer.sched.total_epochs = max(int(num_epochs), 1)
    for epoch in range(start_epoch, num_epochs):
        if stop_event is not None and stop_event.is_set():
            break
        started = time.time()
        try:
            info = trainer.step_epoch(train_loader, val_loader, epoch,
                                      progress_callback=progress, stop_event=stop_event)
        except TrainingStopped:
            break
        info["epoch_seconds"] = time.time() - started
        best = max([h["val_acc"] for h in history] + [info["val_acc"]])
        history.append({
            "epoch": epoch + 1, "train_loss": info["train_loss"],
            "train_acc": info["train_acc"], "val_loss": info["val_loss"],
            "val_acc": info["val_acc"], "params": info["params"],
            "branches": info["branches_per_stage"],
            "seconds": info["epoch_seconds"],
        })
        if checkpoint_path is not None:
            tmp = f"{checkpoint_path}.tmp"
            trainer.save_checkpoint(tmp, history=history, next_epoch=epoch + 1,
                                    best_val_acc=best, class_names=class_names)
            os.replace(tmp, checkpoint_path)
            info["checkpoint"] = str(checkpoint_path)
        if on_epoch is not None:
            on_epoch(info, history)
    return history


def resume_trainer(path, device: Optional[str] = None, lr: float = 2e-3):
    """Rebuild a Trainer from a checkpoint so training can continue.

        Returns ``(trainer, ckpt)``. ``ckpt["next_epoch"]``, ``ckpt["history"]`` and
        ``ckpt["class_names"]`` (when present) are what ``fit`` needs to carry on.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, ckpt = Trainer.load_checkpoint(str(path), device=device)
    normalize_history(ckpt)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    opt.defaults["_new_branch_warmup_epochs"] = cfg.new_branch_warmup_epochs
    for group in opt.param_groups:
        group["_base_lr"] = group["lr"]
    sched = DynamicLRScheduler(opt, cfg.warmup_epochs, cfg.cosine_t_max, lr)
    trainer = Trainer(model, opt, sched, cfg, device=device)
    trainer.load_training_state(ckpt)
    return trainer, ckpt


def normalize_history(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    """Make sure ``ckpt["history"]`` (list of per-epoch dicts) and ``ckpt["next_epoch"]``
    exist.

        Checkpoints written by the first version of the Studio kept the curves as a dict of
        lists under ``studio_state``; they are converted here. Modifies and returns ``ckpt``.
    """
    if isinstance(ckpt.get("history"), list):
        ckpt.setdefault("next_epoch", len(ckpt["history"]))
        return ckpt
    legacy = ckpt.get("studio_state") or {}
    old = legacy.get("history") or {}
    epochs = list(old.get("epochs", []))
    nan = float("nan")

    def column(name):
        values = list(old.get(name, []))[:len(epochs)]
        return values + [nan] * (len(epochs) - len(values))

    cols = {k: column(k) for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    ckpt["history"] = [
        {"epoch": int(e), **{k: v[i] for k, v in cols.items()},
         "params": None, "branches": None, "seconds": 0.0}
        for i, e in enumerate(epochs)
    ]
    ckpt["next_epoch"] = int(legacy.get("next_epoch", len(epochs)))
    return ckpt
