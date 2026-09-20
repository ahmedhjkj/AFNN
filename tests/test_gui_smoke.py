"""Drives the GUI through a short training run. Skipped without a display/tkinter."""
import os
import time

import numpy as np
import pytest
from PIL import Image

tk = pytest.importorskip("tkinter")


def make_dataset(root):
    rng = np.random.default_rng(3)
    for cls, base in (("dark", 40), ("light", 210)):
        (root / cls).mkdir(parents=True)
        for i in range(20):
            arr = np.clip(rng.normal(base, 15, (24, 24, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(root / cls / f"{i}.png")


def wait(app, timeout=120):
    """Pump the Tk loop until the worker has finished and its events were applied."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        time.sleep(0.03)
        if not app.worker.is_alive() and app.events.empty():
            break
    for _ in range(5):
        app.update()
        time.sleep(0.03)


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="needs a display (use xvfb-run)")
def test_gui_trains_and_reports(tmp_path, monkeypatch):
    import gui.app as app_mod
    monkeypatch.setattr(app_mod, "APP_DIR", tmp_path / "state")
    monkeypatch.setattr(app_mod, "AUTOSAVE", tmp_path / "state" / "autosave.pt")
    monkeypatch.setattr(app_mod, "SETTINGS_FILE", tmp_path / "state" / "settings.json")
    make_dataset(tmp_path / "data")

    app = app_mod.App()
    try:
        app._set_dataset(tmp_path / "data")
        for key, value in dict(epochs="2", batch="8", size="32", depth="1", branches="2",
                               max_branches="3", channels="8", lr="0.003").items():
            app.vars[key].set(value)
        app.start_training(resume=False)
        wait(app)
        assert len(app.history) == 2
        assert (tmp_path / "state" / "autosave.pt").exists()
        assert app.verdict_label.cget("text")
        app.start_training(resume=True)             # +2 epochs from the same state
        wait(app)
        assert [h["epoch"] for h in app.history] == [1, 2, 3, 4]
    finally:
        app.destroy()
