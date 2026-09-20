"""Classify images with a trained checkpoint.

Example:
    python scripts/predict.py runs/animals.pt photo1.jpg photo2.png --top 3
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afnn import load_model, predict  # noqa: E402


def main():
    """Print the top predictions for every image given on the command line."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("images", nargs="+")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    model, cfg, names = load_model(args.checkpoint, args.device)
    for path in args.images:
        _, top = predict(model, cfg, path, names, args.top)
        print(path)
        for label, prob in top:
            print(f"  {prob:6.1%}  {label}")


if __name__ == "__main__":
    main()
