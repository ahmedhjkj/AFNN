"""Shrink an image folder into one small .npz file (for uploading to a cloud GPU).

Example:
    python scripts/pack_dataset.py ~/datasets/animals animals_48.npz --size 48
Then train with:  python scripts/train.py --data animals_48.npz --out runs/animals.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afnn.data import DEFAULT_CACHE_DIR, pack_dataset  # noqa: E402


def main():
    """Pack the dataset given on the command line."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", help="dataset root: <root>/<class>/<image>")
    p.add_argument("out", help="output .npz file")
    p.add_argument("--size", type=int, default=48)
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    args = p.parse_args()

    def progress(done, total, cached):
        if done == total or done % 5000 < 128:
            print(f"{done:,}/{total:,}", flush=True)

    failed = pack_dataset(args.data, args.size, args.out, args.cache_dir, progress)
    print(f"saved {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB); "
          f"skipped {len(failed)} unreadable images")


if __name__ == "__main__":
    main()
