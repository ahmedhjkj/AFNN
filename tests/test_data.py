import numpy as np
import pytest
import torch
from PIL import Image

from afnn.data import (build_cache, make_loaders, open_rgb, scan_image_folder,
                       split_indices)


@pytest.fixture()
def dataset(tmp_path):
    rng = np.random.default_rng(0)
    for cls in ("cat", "dog"):
        (tmp_path / cls).mkdir()
        for i in range(12):
            arr = rng.integers(0, 255, (20, 30, 3), dtype=np.uint8)
            Image.fromarray(arr).save(tmp_path / cls / f"{i}.png")
    (tmp_path / "cat" / "broken.jpg").write_bytes(b"not an image")
    return tmp_path


def test_scan_finds_sorted_classes(dataset):
    samples, classes = scan_image_folder(dataset)
    assert classes == ["cat", "dog"]
    assert len(samples) == 25


def test_open_rgb_reads_png(dataset):
    assert open_rgb(dataset / "cat" / "0.png").mode == "RGB"


def test_open_rgb_reports_undecodable_file(dataset):
    with pytest.raises(OSError):
        open_rgb(dataset / "cat" / "broken.jpg")


def test_cache_skips_broken_files_and_is_reused(dataset, tmp_path):
    samples, _ = scan_image_folder(dataset)
    cache = tmp_path / "cache"
    X, y, failed = build_cache(samples, 16, cache)
    assert X.shape == (24, 16, 16, 3) and len(failed) == 1
    calls = []
    build_cache(samples, 16, cache, progress=lambda d, t, c: calls.append(c))
    assert calls and all(calls)                        # second run came from disk


def test_loaders_yield_normalised_batches(dataset, tmp_path):
    train, val, classes, failed = make_loaders(
        dataset, 16, batch_size=5, val_fraction=0.25, device="cpu",
        cache_dir=tmp_path / "cache")
    xb, yb = next(iter(train))
    assert xb.shape[1:] == (3, 16, 16) and xb.dtype == torch.float32
    assert 0.0 <= xb.min() and xb.max() <= 1.0
    assert sum(len(b[1]) for b in val) == len(val.indices)
    assert classes == ["cat", "dog"] and len(failed) == 1


def test_split_is_deterministic():
    a, b = split_indices(100, 0.2), split_indices(100, 0.2)
    assert torch.equal(a[0], b[0]) and len(a[1]) == 20


def test_pack_dataset_roundtrip(dataset, tmp_path):
    from afnn.data import load_arrays, pack_dataset
    out = tmp_path / "packed.npz"
    failed = pack_dataset(dataset, 16, out, cache_dir=tmp_path / "cache")
    X, y, classes = load_arrays(out)
    assert X.shape == (24, 16, 16, 3) and classes == ["cat", "dog"] and len(failed) == 1
    train, val, names, _ = make_loaders(out, 999, 5, 0.25, "cpu")
    assert names == ["cat", "dog"] and train.store.size == 16
