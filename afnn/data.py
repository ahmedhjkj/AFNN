"""Image-folder loading with a one-off uint8 cache that lives on the GPU.

Every image is resized once to ``size x size`` and stored in a single ``.npy`` array.
Training then never touches the disk or the JPEG decoder again: batches are sliced
straight out of a GPU tensor.
"""
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile

from .augment import gpu_augment

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff",
    ".avif", ".heic",
}
DEFAULT_CACHE_DIR = Path.home() / ".afnn" / "cache"

Sample = Tuple[str, int]


def scan_image_folder(root) -> Tuple[List[Sample], List[str]]:
    """List ``(path, class_index)`` pairs for a ``root/<class>/<image>`` tree.

            Classes are the sub-directory names in sorted order, so the index of a
            class is stable between runs.
    """
    root = Path(root)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    samples: List[Sample] = []
    for idx, name in enumerate(classes):
        for path in sorted((root / name).rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((str(path), idx))
    return samples, classes


def open_rgb(path) -> Image.Image:
    """Open any image as RGB.

            Pillow is tried first. If it cannot decode the file (typically AVIF/HEIC
            with a Pillow build that lacks the codec: "No codec available") the
            fallbacks are pillow-heif, OpenCV and finally the ``ffmpeg`` binary.
    """
    first_error = None
    try:
        with Image.open(path) as im:
            im.load()
            return im.convert("RGB")
    except Exception as exc:
        first_error = exc
    try:
        import pillow_heif
        return pillow_heif.open_heif(str(path)).to_pillow().convert("RGB")
    except Exception:
        pass
    try:
        import cv2
        arr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is not None:
            return Image.fromarray(arr[:, :, ::-1])
    except Exception:
        pass
    try:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            proc = subprocess.run(
                [ffmpeg, "-v", "error", "-i", str(path), "-frames:v", "1",
                 "-f", "image2pipe", "-vcodec", "png", "-"],
                capture_output=True, timeout=30)
            if proc.returncode == 0 and proc.stdout:
                with Image.open(io.BytesIO(proc.stdout)) as im:
                    return im.convert("RGB")
    except Exception:
        pass
    raise OSError(
        f"Cannot decode {path}. If it is an AVIF/HEIC file run "
        f"`pip install pillow-heif` or convert it to JPG/PNG. ({first_error})")


def load_image_array(path, size: int) -> np.ndarray:
    """Load one image as a ``uint8`` array of shape ``[size, size, 3]``."""
    im = open_rgb(path).resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def _cache_key(samples: List[Sample], size: int) -> str:
    """Hash of the sample list (paths, sizes, mtimes, labels) and target size."""
    h = hashlib.md5(f"afnn-cache-v1|{size}|{len(samples)}".encode())
    for path, label in samples:
        try:
            st = os.stat(path)
            h.update(f"{path}|{st.st_size}|{st.st_mtime_ns}|{label}\n".encode("utf-8", "ignore"))
        except OSError:
            h.update(f"{path}|missing|{label}\n".encode("utf-8", "ignore"))
    return h.hexdigest()[:20]


def build_cache(samples: List[Sample], size: int, cache_dir=DEFAULT_CACHE_DIR,
                progress: Optional[Callable[[int, int, bool], None]] = None,
                cancel=None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Convert every image to a small array once and store it on disk.

            Returns ``(X, y, failed_paths)`` with ``X`` of shape ``[N, size, size, 3]``
            (uint8) and ``y`` of shape ``[N]`` (int64). A second call with the same
            images and size loads the arrays from disk instead of decoding again.

            ``progress(done, total, from_cache)`` is called while working and
            ``cancel`` may be a ``threading.Event`` used to abort.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(samples, size)
    fx, fy, ff = (cache_dir / f"{key}_x.npy", cache_dir / f"{key}_y.npy",
                  cache_dir / f"{key}_failed.json")
    if fx.exists() and fy.exists() and ff.exists():
        try:
            X, y = np.load(fx), np.load(fy)
            failed = json.loads(ff.read_text(encoding="utf-8"))
            if X.ndim == 4 and X.shape[0] == y.shape[0] and X.shape[1] == size:
                if progress:
                    progress(len(samples), len(samples), True)
                return X, y, failed
        except Exception:
            pass                                    # corrupt cache: rebuild

    n = len(samples)
    X = np.empty((n, size, size, 3), dtype=np.uint8)
    ok = np.zeros(n, dtype=bool)

    def work(a: int, b: int) -> int:
        for i in range(a, b):
            try:
                X[i] = load_image_array(samples[i][0], size)
                ok[i] = True
            except Exception:
                ok[i] = False
        return b - a

    chunk = 128
    workers = max(2, min(8, os.cpu_count() or 2))
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, a, min(a + chunk, n)) for a in range(0, n, chunk)]
        try:
            for fut in as_completed(futures):
                done += fut.result()
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("Image preparation was cancelled.")
                if progress:
                    progress(done, n, False)
        except BaseException:
            for fut in futures:
                fut.cancel()
            raise

    labels = np.asarray([label for _, label in samples], dtype=np.int64)
    failed = [samples[i][0] for i in range(n) if not ok[i]]
    if not ok.all():
        X, labels = X[ok], labels[ok]
    for path, arr in ((fx, X), (fy, labels)):
        tmp = path.with_name(path.name + ".tmp.npy")
        np.save(tmp, arr)
        os.replace(tmp, path)
    tmp = ff.with_name(ff.name + ".tmp")
    tmp.write_text(json.dumps(failed), encoding="utf-8")
    os.replace(tmp, ff)
    return X, labels, failed


class CachedImageStore:
    """All cached images in one place: on the GPU if they fit, else in RAM."""

    MAX_GPU_ELEMENTS = 600_000_000        # about 600 MB of uint8

    def __init__(self, x_u8: np.ndarray, y: np.ndarray, device: str):
        x = torch.from_numpy(np.ascontiguousarray(x_u8))              # N,S,S,3
        labels = torch.from_numpy(np.ascontiguousarray(y.astype(np.int64)))
        self.device = device
        self.on_gpu = str(device).startswith("cuda") and x.numel() <= self.MAX_GPU_ELEMENTS
        self.x = x.to(device) if self.on_gpu else x
        self.y = labels.to(device) if self.on_gpu else labels
        self.size = int(x_u8.shape[1])


class CachedImageLoader:
    """Minimal DataLoader replacement: ``len()`` and iteration yield ``(x, y)``.

    Images are returned as float tensors in ``[0, 1]`` with shape ``[B, 3, S, S]``.
    """

    def __init__(self, store: CachedImageStore, indices: torch.Tensor,
                 batch_size: int, shuffle: bool, augment: bool):
        self.store = store
        self.indices = indices.to(store.x.device)
        self.batch_size = max(1, int(batch_size))
        self.shuffle = shuffle
        self.augment = augment
        n = int(self.indices.numel())
        batches = max(1, math.ceil(n / self.batch_size))
        # A last batch with a single image breaks BatchNorm in training.
        if shuffle and batches > 1 and n - (batches - 1) * self.batch_size < 2:
            batches -= 1
        self._batches = batches

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return self._batches

    def __iter__(self):
        """Yield ``(images, labels)`` batches, shuffled and augmented when configured."""
        store = self.store
        n = int(self.indices.numel())
        dev = self.indices.device
        order = torch.randperm(n, device=dev) if self.shuffle else torch.arange(n, device=dev)
        bs = self.batch_size
        for b in range(self._batches):
            sel = self.indices[order[b * bs:(b + 1) * bs]]
            xb, yb = store.x[sel], store.y[sel]
            if not store.on_gpu:
                xb = xb.to(store.device, non_blocking=True)
                yb = yb.to(store.device, non_blocking=True)
            xb = xb.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
            if self.augment:
                xb = gpu_augment(xb)
            yield xb, yb


def split_indices(n: int, val_fraction: float, seed: int = 42):
    """Deterministic train/validation split. Returns ``(train_idx, val_idx)``."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    val_n = max(1, int(n * val_fraction))
    return perm[val_n:], perm[:val_n]


def save_arrays(X: np.ndarray, y: np.ndarray, classes: List[str], path) -> None:
    """Write the resized images to one compressed ``.npz`` file.

        The file is small (about ``N * size * size * 3`` bytes before compression) and can be
        copied to another machine, e.g. a cloud GPU, instead of the original images.
    """
    np.savez_compressed(path, x=X, y=y, classes=np.array(classes, dtype=object))


def load_arrays(path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Read a file written by ``save_arrays``; returns ``(X, y, class_names)``."""
    with np.load(path, allow_pickle=True) as f:
        return f["x"], f["y"], [str(c) for c in f["classes"]]


def pack_dataset(root, size: int, out_path, cache_dir=DEFAULT_CACHE_DIR, progress=None):
    """Resize every image of ``root`` once and store the result as a single ``.npz``.

    Returns the list of unreadable files that were skipped.
    """
    samples, classes = scan_image_folder(root)
    X, y, failed = build_cache(samples, size, cache_dir, progress)
    save_arrays(X, y, classes, out_path)
    return failed


def make_loaders(root, size: int, batch_size: int, val_fraction: float, device: str,
                 cache_dir=DEFAULT_CACHE_DIR, progress=None, cancel=None):
    """Build the train/validation loaders from an image folder or a packed ``.npz``.

        For a folder the images are scanned and cached; for an ``.npz`` (see ``pack_dataset``)
        the arrays are loaded directly and ``size`` is taken from the file.
        Returns ``(train_loader, val_loader, class_names, failed_paths)``.
    """
    if Path(root).is_file():
        X, y, classes = load_arrays(root)
        failed = []
    else:
        samples, classes = scan_image_folder(root)
        if len(classes) < 2:
            raise ValueError("The dataset needs at least two class folders.")
        X, y, failed = build_cache(samples, size, cache_dir, progress, cancel)
    n = int(len(y))
    if n < 2 or len(np.unique(y)) < 2:
        raise ValueError("Too few readable images.")
    train_idx, val_idx = split_indices(n, val_fraction)
    if len(train_idx) < 1:
        raise ValueError("Too few images left after the validation split.")
    store = CachedImageStore(X, y, device)
    train_loader = CachedImageLoader(store, train_idx, batch_size, shuffle=True, augment=True)
    val_loader = CachedImageLoader(store, val_idx, batch_size, shuffle=False, augment=False)
    return train_loader, val_loader, classes, failed
