"""Train AFNN on a folder of images (one sub-folder per class), without the GUI.

Example:
    python scripts/train.py --data ~/datasets/animals --out runs/animals.pt
    python scripts/train.py --data ~/datasets/animals --out runs/animals.pt --resume runs/animals.pt --epochs 30
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from afnn import build_trainer, fit, resume_trainer  # noqa: E402
from afnn.data import DEFAULT_CACHE_DIR, make_loaders  # noqa: E402


def parse_args():
    """Parse the command-line options."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True,
                   help="dataset root <root>/<class>/<image>, or a .npz from pack_dataset.py")
    p.add_argument("--out", default="runs/model.pt", help="checkpoint path (written every epoch)")
    p.add_argument("--resume", help="continue from this checkpoint")
    p.add_argument("--size", type=int, default=48, help="input resolution (default 48)")
    p.add_argument("--epochs", type=int, default=20, help="total epochs to reach")
    p.add_argument("--batch", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--val", type=float, default=0.15, help="validation fraction")
    p.add_argument("--channels", type=int, default=28, help="base channels of the stem")
    p.add_argument("--depth", type=int, default=2, help="initial fractal depth")
    p.add_argument("--branches", type=int, default=3, help="initial branches per stage")
    p.add_argument("--max-branches", type=int, default=5)
    p.add_argument("--norm", default="batchnorm", choices=["batchnorm", "groupnorm", "none"])
    p.add_argument("--act", default="relu", choices=["relu", "gelu", "silu", "mish"])
    p.add_argument("--no-load-balance", action="store_true")
    p.add_argument("--representation", action="store_true",
                   help="add the contrastive representation loss (slower)")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    return p.parse_args()


def main():
    """Load the data, build or resume the trainer and run ``fit``."""
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    def cache_progress(done, total, cached):
        if cached:
            print("loading images from cache")
        elif done == total or done % 5000 < 128:
            print(f"preparing images: {done:,}/{total:,}")

    size = args.size
    if args.resume:
        trainer, ckpt = resume_trainer(args.resume, device=device, lr=args.lr)
        size = int(trainer.cfg.input_size)
        history = ckpt.get("history", [])
        start = int(ckpt.get("next_epoch", len(history)))
    else:
        trainer, ckpt, history, start = None, None, [], 0

    train_loader, val_loader, classes, failed = make_loaders(
        args.data, size, args.batch, args.val, device, args.cache_dir,
        progress=cache_progress)
    size = int(train_loader.store.size)                 # authoritative when --data is a .npz
    if failed:
        print(f"skipped {len(failed):,} unreadable images")
    print(f"{len(classes)} classes | train batches {len(train_loader):,} | "
          f"val batches {len(val_loader):,}")

    if trainer is None:
        trainer = build_trainer(
            len(classes), size, channels=args.channels, depth=args.depth,
            branches=args.branches, max_branches=args.max_branches, norm=args.norm,
            act=args.act, load_balance=not args.no_load_balance,
            representation=args.representation, lr=args.lr, epochs=args.epochs,
            device=device)
    print(f"parameters: {trainer.model.count_parameters():,} | "
          f"branches per stage: {trainer.model.count_branches_per_stage()}")

    started = time.time()

    def on_epoch(info, hist):
        h = hist[-1]
        grew = f" | grew +{info['branches_added']}" if info.get("grew") else ""
        print(f"epoch {h['epoch']:>3}/{args.epochs} | train_loss {h['train_loss']:.4f} | "
              f"val_loss {h['val_loss']:.4f} | train_acc {h['train_acc']:.3f} | "
              f"val_acc {h['val_acc']:.3f} | {h['seconds']:.0f}s | "
              f"params {h['params']:,}{grew}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fit(trainer, train_loader, val_loader, start_epoch=start, num_epochs=args.epochs,
        history=history, class_names=classes, checkpoint_path=args.out, on_epoch=on_epoch)
    print(f"done in {(time.time() - started) / 60:.1f} min | checkpoint: {args.out}")


if __name__ == "__main__":
    main()
