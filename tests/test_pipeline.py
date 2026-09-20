"""End to end: images on disk -> cache -> training -> checkpoint -> prediction."""
import numpy as np
import torch
from PIL import Image

from afnn import build_trainer, fit, load_model, predict
from afnn.data import make_loaders


def make_dataset(root):
    rng = np.random.default_rng(1)
    for cls, base in (("dark", 40), ("light", 210)):
        (root / cls).mkdir()
        for i in range(20):
            arr = np.clip(rng.normal(base, 15, (24, 24, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(root / cls / f"{i}.png")


def test_train_save_load_predict(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    make_dataset(data)
    train, val, classes, _ = make_loaders(data, 16, batch_size=8, val_fraction=0.25,
                                          device="cpu", cache_dir=tmp_path / "cache")
    torch.manual_seed(0)
    trainer = build_trainer(len(classes), 16, channels=8, depth=1, branches=2,
                            lr=3e-3, epochs=4, device="cpu")
    ckpt = tmp_path / "model.pt"
    seen = []
    history = fit(trainer, train, val, num_epochs=4, class_names=classes,
                  checkpoint_path=ckpt, on_epoch=lambda info, h: seen.append(len(h)))
    assert seen == [1, 2, 3, 4] and len(history) == 4
    assert history[-1]["val_acc"] > 0.8          # two trivially separable classes

    model, cfg, names = load_model(ckpt, device="cpu")
    assert names == ["dark", "light"]
    _, top = predict(model, cfg, data / "dark" / "0.png", names, top_k=2)
    assert top[0][0] == "dark"


def test_resume_continues_from_saved_epoch(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    make_dataset(data)
    train, val, classes, _ = make_loaders(data, 16, 8, 0.25, "cpu", tmp_path / "cache")
    trainer = build_trainer(2, 16, channels=8, depth=1, branches=2, epochs=2, device="cpu")
    ckpt = tmp_path / "m.pt"
    fit(trainer, train, val, num_epochs=2, class_names=classes, checkpoint_path=ckpt)

    from afnn import resume_trainer
    resumed, state = resume_trainer(ckpt, device="cpu")
    history = fit(resumed, train, val, start_epoch=state["next_epoch"], num_epochs=4,
                  history=state["history"], class_names=classes)
    assert [h["epoch"] for h in history] == [1, 2, 3, 4]


def test_stop_event_ends_training_cleanly(tmp_path):
    import threading
    data = tmp_path / "data"
    data.mkdir()
    make_dataset(data)
    train, val, classes, _ = make_loaders(data, 16, 8, 0.25, "cpu", tmp_path / "cache")
    trainer = build_trainer(2, 16, channels=8, depth=1, branches=2, epochs=5, device="cpu")
    stop = threading.Event()
    history = fit(trainer, train, val, num_epochs=5, stop_event=stop,
                  on_epoch=lambda info, h: stop.set())
    assert len(history) == 1


def test_legacy_checkpoint_history_is_converted():
    from afnn.training import normalize_history
    ckpt = {"studio_state": {"history": {"epochs": [1, 2], "train_loss": [1.0, 0.8],
                                         "val_loss": [1.1, 0.9], "val_acc": [0.3, 0.4],
                                         "train_acc": [0.35]},
                             "next_epoch": 2}}
    normalize_history(ckpt)
    assert [h["epoch"] for h in ckpt["history"]] == [1, 2] and ckpt["next_epoch"] == 2
    assert ckpt["history"][1]["val_acc"] == 0.4 and ckpt["history"][1]["train_acc"] != ckpt["history"][1]["train_acc"]


def test_config_from_dict_ignores_removed_fields():
    from afnn import AFNNConfig
    d = AFNNConfig().to_dict()
    d.update(use_compile=False, hard_forward_mode="dense_inside", inference_top_k=0)
    assert AFNNConfig.from_dict(d).to_dict() == AFNNConfig().to_dict()
