"""Tkinter front end for AFNN: train, watch the curves, test images.

Run from the project root:  python -m gui.app
"""
import json
import queue
import threading
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import torch
from PIL import ImageTk

from afnn import build_trainer, fit, predict, resume_trainer
from afnn.data import IMAGE_EXTENSIONS, make_loaders, scan_image_folder

APP_DIR = Path.home() / ".afnn"
AUTOSAVE = APP_DIR / "autosave.pt"
SETTINGS_FILE = APP_DIR / "settings.json"

BLUE, RED, GREEN, ORANGE = "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"


def format_seconds(seconds: float) -> str:
    """Format a duration as ``1h 05m``, ``12m 30s`` or ``45s``."""
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s" if m else f"{s}s"


class App(tk.Tk):
    """Main window. Training runs in a worker thread and reports through a queue."""

    def __init__(self):
        super().__init__()
        self.title("AFNN - Adaptive Fractal Neural Network")
        self.geometry("1100x760")
        ttk.Style(self).theme_use("clam")

        self.trainer = None
        self.class_names = []
        self.history = []
        self.next_epoch = 0
        self.data_dir = None
        self.worker = None
        self.stop_event = threading.Event()
        self.events = queue.Queue()

        self._build_settings_panel()
        self._build_tabs()
        self._load_settings()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_events)

    # ------------------------------------------------------------------ layout
    def _build_settings_panel(self):
        """Left column: dataset picker, hyper-parameters and the control buttons."""
        panel = ttk.Frame(self, padding=10)
        panel.pack(side="left", fill="y")

        ttk.Button(panel, text="Choose dataset folder...", command=self.choose_dataset).pack(fill="x")
        self.data_label = ttk.Label(panel, text="No dataset selected", wraplength=250,
                                    foreground="#666")
        self.data_label.pack(fill="x", pady=(4, 10))

        self.vars = {
            "epochs": tk.StringVar(value="20"), "batch": tk.StringVar(value="40"),
            "lr": tk.StringVar(value="0.002"), "val": tk.StringVar(value="0.15"),
            "size": tk.StringVar(value="48"), "depth": tk.StringVar(value="2"),
            "branches": tk.StringVar(value="3"), "max_branches": tk.StringVar(value="5"),
            "channels": tk.StringVar(value="28"), "norm": tk.StringVar(value="batchnorm"),
            "act": tk.StringVar(value="relu"),
        }
        self.load_balance = tk.BooleanVar(value=True)
        self.representation = tk.BooleanVar(value=False)

        grid = ttk.Frame(panel)
        grid.pack(fill="x")
        rows = [("Epochs", "epochs", None), ("Batch size", "batch", None),
                ("Learning rate", "lr", None), ("Validation split", "val", None),
                ("Image size", "size", ("32", "48", "64")),
                ("Initial depth", "depth", None), ("Initial branches", "branches", None),
                ("Max branches", "max_branches", None), ("Base channels", "channels", None),
                ("Normalization", "norm", ("batchnorm", "groupnorm", "none")),
                ("Activation", "act", ("relu", "gelu", "silu", "mish"))]
        for r, (label, key, choices) in enumerate(rows):
            ttk.Label(grid, text=label).grid(row=r, column=0, sticky="w", pady=2)
            if choices:
                w = ttk.Combobox(grid, textvariable=self.vars[key], values=choices,
                                 state="readonly", width=10)
            else:
                w = ttk.Entry(grid, textvariable=self.vars[key], width=12)
            w.grid(row=r, column=1, sticky="e", padx=(8, 0), pady=2)
        ttk.Checkbutton(panel, text="Load-balancing loss", variable=self.load_balance).pack(anchor="w", pady=(8, 0))
        ttk.Checkbutton(panel, text="Representation learning (slower)",
                        variable=self.representation).pack(anchor="w")

        self.start_button = ttk.Button(panel, text="Start new training",
                                       command=lambda: self.start_training(resume=False))
        self.start_button.pack(fill="x", pady=(14, 2))
        self.resume_button = ttk.Button(panel, text="Continue training (+ Epochs)",
                                        command=lambda: self.start_training(resume=True),
                                        state="disabled")
        self.resume_button.pack(fill="x", pady=2)
        self.stop_button = ttk.Button(panel, text="Stop", command=self.stop_training,
                                      state="disabled")
        self.stop_button.pack(fill="x", pady=2)
        ttk.Separator(panel).pack(fill="x", pady=8)
        ttk.Button(panel, text="Load checkpoint...", command=self.load_checkpoint).pack(fill="x", pady=2)
        ttk.Button(panel, text="Save checkpoint...", command=self.save_checkpoint).pack(fill="x", pady=2)

    def _build_tabs(self):
        """Right side: training, image test and model tabs."""
        tabs = ttk.Notebook(self)
        tabs.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)

        train_tab = ttk.Frame(tabs, padding=8)
        tabs.add(train_tab, text="Training")
        self.status_label = ttk.Label(train_tab, text="Idle", font=("TkDefaultFont", 10, "bold"))
        self.status_label.pack(anchor="w")
        self.detail_label = ttk.Label(train_tab, text="", foreground="#555")
        self.detail_label.pack(anchor="w")
        self.bar = ttk.Progressbar(train_tab, maximum=100)
        self.bar.pack(fill="x", pady=4)
        self.event_label = ttk.Label(train_tab, text="", foreground=GREEN)
        self.event_label.pack(anchor="w")

        ttk.Label(train_tab, text="Learning curves").pack(anchor="w", pady=(6, 0))
        self.curve_canvas = tk.Canvas(train_tab, height=190, bg="white", highlightthickness=1,
                                      highlightbackground="#ccc")
        self.curve_canvas.pack(fill="x")
        legend = ttk.Frame(train_tab)
        legend.pack(anchor="w")
        for color, text in ((BLUE, "train_loss"), (RED, "val_loss"), (GREEN, "val_acc")):
            ttk.Label(legend, text="\u25CF " + text, foreground=color).pack(side="left", padx=6)

        ttk.Label(train_tab, text="Understanding vs. memorising").pack(anchor="w", pady=(6, 0))
        self.gap_canvas = tk.Canvas(train_tab, height=150, bg="white", highlightthickness=1,
                                    highlightbackground="#ccc")
        self.gap_canvas.pack(fill="x")
        legend = ttk.Frame(train_tab)
        legend.pack(anchor="w")
        for color, text in ((BLUE, "train_acc"), (GREEN, "val_acc"), (ORANGE, "gap")):
            ttk.Label(legend, text="\u25CF " + text, foreground=color).pack(side="left", padx=6)
        self.verdict_label = ttk.Label(train_tab, text="", font=("TkDefaultFont", 10, "bold"))
        self.verdict_label.pack(anchor="w")

        self.log = tk.Text(train_tab, height=7, state="disabled", wrap="word")
        self.log.pack(fill="both", expand=True, pady=(6, 0))
        for canvas in (self.curve_canvas, self.gap_canvas):
            canvas.bind("<Configure>", lambda _e: self._redraw())

        test_tab = ttk.Frame(tabs, padding=8)
        tabs.add(test_tab, text="Test image")
        ttk.Button(test_tab, text="Choose image...", command=self.test_image).pack(anchor="w")
        self.preview = ttk.Label(test_tab)
        self.preview.pack(pady=8)
        self.prediction_label = ttk.Label(test_tab, text="", justify="left",
                                          font=("TkFixedFont", 11))
        self.prediction_label.pack(anchor="w")

        model_tab = ttk.Frame(tabs, padding=8)
        tabs.add(model_tab, text="Model")
        self.model_text = tk.Text(model_tab, height=18, state="disabled", wrap="word")
        self.model_text.pack(fill="both", expand=True)
        self._refresh_model_tab()

    # ---------------------------------------------------------------- settings
    def _load_settings(self):
        """Restore the last used hyper-parameters (and dataset folder)."""
        try:
            saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return
        for key, var in self.vars.items():
            if key in saved:
                var.set(str(saved[key]))
        self.load_balance.set(bool(saved.get("load_balance", True)))
        self.representation.set(bool(saved.get("representation", False)))
        if saved.get("data") and Path(saved["data"]).is_dir():
            self._set_dataset(Path(saved["data"]))

    def _save_settings(self):
        """Persist the current hyper-parameters (best effort)."""
        data = {k: v.get() for k, v in self.vars.items()}
        data.update(load_balance=self.load_balance.get(),
                    representation=self.representation.get(),
                    data=str(self.data_dir) if self.data_dir else None)
        try:
            APP_DIR.mkdir(parents=True, exist_ok=True)
            SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _read_settings(self) -> dict:
        """Parse and validate the form; raises ValueError with a readable message."""
        try:
            s = {
                "epochs": int(self.vars["epochs"].get()), "batch": int(self.vars["batch"].get()),
                "lr": float(self.vars["lr"].get()), "val": float(self.vars["val"].get()),
                "size": int(self.vars["size"].get()), "depth": int(self.vars["depth"].get()),
                "branches": int(self.vars["branches"].get()),
                "max_branches": int(self.vars["max_branches"].get()),
                "channels": int(self.vars["channels"].get()),
                "norm": self.vars["norm"].get(), "act": self.vars["act"].get(),
                "load_balance": self.load_balance.get(),
                "representation": self.representation.get(),
            }
        except ValueError:
            raise ValueError("Every numeric field must contain a valid number.")
        if s["epochs"] < 1 or s["batch"] < 2 or s["lr"] <= 0:
            raise ValueError("Epochs must be >= 1, batch size >= 2 and learning rate > 0.")
        if not 0.02 <= s["val"] <= 0.5:
            raise ValueError("Validation split must be between 0.02 and 0.5.")
        if s["branches"] < 2 or s["depth"] < 1 or s["channels"] < 4:
            raise ValueError("Branches >= 2, depth >= 1 and channels >= 4 are required.")
        s["max_branches"] = max(s["max_branches"], s["branches"])
        return s

    # ------------------------------------------------------------- dataset
    def choose_dataset(self):
        """Ask for the dataset root and show how many classes/images it holds."""
        path = filedialog.askdirectory(title="Dataset folder (one sub-folder per class)")
        if path:
            self._set_dataset(Path(path))

    def _set_dataset(self, path: Path):
        """Remember the dataset folder and describe it in the side panel."""
        self.data_dir = path
        try:
            samples, classes = scan_image_folder(path)
            text = f"{path}\n{len(classes)} classes | {len(samples):,} images"
            if len(classes) < 2:
                text += "\nNeeds at least two class sub-folders."
        except OSError as exc:
            text = f"{path}\n{exc}"
        self.data_label.config(text=text)

    # ------------------------------------------------------------ training
    def start_training(self, resume: bool):
        """Validate the form and start the worker thread."""
        if self.worker is not None and self.worker.is_alive():
            return
        if self.data_dir is None:
            messagebox.showwarning("Dataset", "Choose a dataset folder first.")
            return
        if resume and self.trainer is None:
            messagebox.showwarning("Resume", "Train or load a model first.")
            return
        try:
            settings = self._read_settings()
        except ValueError as exc:
            messagebox.showerror("Settings", str(exc))
            return
        self._save_settings()
        self.stop_event.clear()
        self._set_running(True)
        self.status_label.config(text="Preparing images...")
        self.worker = threading.Thread(target=self._train_worker, args=(settings, resume),
                                       daemon=True)
        self.worker.start()

    def _train_worker(self, s: dict, resume: bool):
        """Worker thread: build loaders and the trainer, then run ``fit``."""
        put = self.events.put
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            size = int(self.trainer.cfg.input_size) if resume else s["size"]
            train_loader, val_loader, classes, failed = make_loaders(
                self.data_dir, size, s["batch"], s["val"], device,
                progress=lambda d, t, c: put(("cache", (d, t, c))), cancel=self.stop_event)
            if failed:
                put(("log", f"Skipped {len(failed):,} unreadable images.\n"))
            if resume:
                if len(classes) != self.trainer.cfg.num_classes:
                    raise ValueError("The dataset has a different number of classes than the model.")
                history, start = list(self.history), self.next_epoch
            else:
                self.trainer = build_trainer(
                    len(classes), size, channels=s["channels"], depth=s["depth"],
                    branches=s["branches"], max_branches=s["max_branches"], norm=s["norm"],
                    act=s["act"], load_balance=s["load_balance"],
                    representation=s["representation"], lr=s["lr"],
                    epochs=s["epochs"], device=device)
                history, start = [], 0
            self.class_names = classes
            self.history, self.next_epoch = history, start
            total = start + s["epochs"]
            put(("log", f"Device: {device} | parameters: "
                        f"{self.trainer.model.count_parameters():,} | epochs {start + 1}..{total}\n"))
            put(("model", None))

            def progress(kind, done, n, elapsed):
                if done == n or done % 10 == 0:
                    put(("progress", (kind, done, n, elapsed)))

            def on_epoch(info, hist):
                self.history, self.next_epoch = hist, len(hist)
                put(("epoch", (info, total)))

            APP_DIR.mkdir(parents=True, exist_ok=True)
            fit(self.trainer, train_loader, val_loader, start_epoch=start, num_epochs=total,
                history=history, class_names=classes, checkpoint_path=AUTOSAVE,
                progress=progress, on_epoch=on_epoch, stop_event=self.stop_event)
            put(("done", None))
        except Exception as exc:
            put(("error", (str(exc), traceback.format_exc())))

    def stop_training(self):
        """Ask the worker to stop after the current batch."""
        if self.worker is not None and self.worker.is_alive():
            self.stop_event.set()
            self._append_log("Stop requested...\n")

    def _set_running(self, running: bool):
        """Enable/disable the buttons depending on whether training is active."""
        self.start_button.config(state="disabled" if running else "normal")
        self.resume_button.config(
            state="disabled" if running or self.trainer is None else "normal")
        self.stop_button.config(state="normal" if running else "disabled")

    # -------------------------------------------------------------- events
    def _poll_events(self):
        """Apply queued worker messages to the widgets (runs on the Tk thread)."""
        try:
            while True:
                kind, payload = self.events.get_nowait()
                getattr(self, f"_on_{kind}")(payload)
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _on_log(self, text):
        """Event handler: append a message to the log."""
        self._append_log(text)

    def _on_cache(self, payload):
        """Event handler: show image-preparation progress."""
        done, total, cached = payload
        self.status_label.config(text="Loading images from cache..." if cached
                                 else f"Preparing images: {done:,}/{total:,}")
        self.bar["value"] = 100 * done / max(total, 1)

    def _on_progress(self, payload):
        """Event handler: show batch progress, speed and ETA of the running epoch."""
        kind, done, total, elapsed = payload
        label = "Training" if kind == "train" else "Validating"
        eta = elapsed * (total - done) / max(done, 1)
        self.status_label.config(
            text=f"Epoch {self.next_epoch + 1}: {label} {done}/{total} "
                 f"({done / max(elapsed, 1e-6):.2f} batch/s, {format_seconds(eta)} left)")
        self.bar["value"] = 100 * done / max(total, 1)

    def _on_epoch(self, payload):
        """Event handler: log the finished epoch, update labels and redraw the charts."""
        info, total = payload
        h = self.history[-1]
        self._append_log(
            f"epoch {h['epoch']}/{total} | train_loss {h['train_loss']:.4f} | "
            f"val_loss {h['val_loss']:.4f} | train_acc {h['train_acc']:.3f} | "
            f"val_acc {h['val_acc']:.3f} | {format_seconds(h['seconds'])}\n")
        remaining = h["seconds"] * (total - h["epoch"])
        self.detail_label.config(
            text=f"Epoch {h['epoch']}/{total} done | {format_seconds(h['seconds'])} per epoch | "
                 f"about {format_seconds(remaining)} left")
        if info.get("grew"):
            self.event_label.config(
                text=f"Model grew: +{info['branches_added']} branches "
                     f"({info['params']:,} parameters)")
        elif info.get("rejuvenated"):
            self.event_label.config(text=f"Re-initialised {info['rejuvenated']} dead branches")
        self._redraw()
        self._refresh_model_tab()

    def _on_model(self, _payload):
        """Event handler: refresh the Model tab."""
        self._refresh_model_tab()

    def _on_done(self, _payload):
        """Event handler: training finished or was stopped."""
        best = max((h["val_acc"] for h in self.history), default=0.0)
        stopped = self.stop_event.is_set()
        self.status_label.config(text="Stopped" if stopped else "Finished")
        self._append_log(f"{'Stopped' if stopped else 'Finished'} | best val_acc {best:.2%} | "
                         f"checkpoint: {AUTOSAVE}\n")
        self._set_running(False)
        self._refresh_model_tab()

    def _on_error(self, payload):
        """Event handler: show a worker exception."""
        message, tb = payload
        self.status_label.config(text="Error")
        self._append_log(tb + "\n")
        self._set_running(False)
        messagebox.showerror("Training error", message)

    def _append_log(self, text: str):
        """Add text to the read-only log box."""
        self.log.config(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.config(state="disabled")

    # ------------------------------------------------------------- charts
    @staticmethod
    def _finite(v):
        """True for real numbers (not NaN or infinity)."""
        return v == v and v not in (float("inf"), float("-inf"))

    def _plot(self, canvas, values, color, lo, hi, pad=10):
        """Draw one series (skipping NaN/inf) scaled to ``[lo, hi]``."""
        w, h = max(canvas.winfo_width(), 200), max(canvas.winfo_height(), 80)
        n, span, pts = len(values), max(hi - lo, 1e-8), []
        for i, v in enumerate(values):
            if not self._finite(v):
                continue
            x = pad + (w - 2 * pad) * i / max(n - 1, 1) if n > 1 else w / 2
            y = h - pad - (h - 2 * pad) * (v - lo) / span
            pts.extend([x, y])
        if len(pts) >= 4:
            canvas.create_line(*pts, fill=color, width=2)
        elif len(pts) == 2:
            canvas.create_oval(pts[0] - 4, pts[1] - 4, pts[0] + 4, pts[1] + 4,
                               fill=color, outline=color)
        return (pts[-2], pts[-1]) if pts else None

    def _redraw(self):
        """Redraw both charts and the memorisation verdict from ``self.history``."""
        hist = self.history
        for canvas in (self.curve_canvas, self.gap_canvas):
            canvas.delete("all")
        if not hist:
            self.verdict_label.config(text="", foreground="#555")
            return
        tl = [h["train_loss"] for h in hist]
        vl = [h["val_loss"] for h in hist]
        va = [h["val_acc"] for h in hist]
        ta = [h["train_acc"] for h in hist]
        losses = [v for v in tl + vl if self._finite(v)] or [0.0, 1.0]
        lo, hi = min(losses), max(losses)
        self._plot(self.curve_canvas, tl, BLUE, lo, hi)
        self._plot(self.curve_canvas, vl, RED, lo, hi)
        end = self._plot(self.curve_canvas, va, GREEN, 0.0, 1.0)
        if end:
            self.curve_canvas.create_text(
                self.curve_canvas.winfo_width() - 10, max(12, end[1] - 10), anchor="e",
                fill=GREEN, text=f"val_acc {va[-1]:.1%}", font=("TkDefaultFont", 10, "bold"))

        gap = [max(t - v, 0.0) if self._finite(t) and self._finite(v) else float("nan")
               for t, v in zip(ta, va)]
        top = min(1.0, max([0.05] + [v for v in ta + va + gap if self._finite(v)]) * 1.15)
        self._plot(self.gap_canvas, ta, BLUE, 0.0, top)
        self._plot(self.gap_canvas, va, GREEN, 0.0, top)
        self._plot(self.gap_canvas, gap, ORANGE, 0.0, top)

        t_last, v_last = ta[-1], va[-1]
        g_last = t_last - v_last
        recent = va[-4:]
        improving = len(recent) >= 2 and recent[-1] >= recent[0] - 0.002
        nums = f"train {t_last:.0%} | validation {v_last:.0%} | gap {max(g_last, 0):.0%}"
        if len(hist) < 2:
            text, color = "One epoch only - the verdict starts at epoch 2", "#555"
        elif g_last <= 0.05 and improving:
            text, color = "Learning well (not memorising)", "#1a7d1a"
        elif g_last <= 0.05:
            text, color = "Not memorising, but no longer improving", "#a86a00"
        elif g_last <= 0.15:
            text, color = "Balanced, slight memorisation starting", "#a86a00"
        else:
            text, color = "Memorising: large train/validation gap", "#b00020"
        self.verdict_label.config(text=f"{text}\n{nums}", foreground=color)

    # ------------------------------------------------------------- model
    def _refresh_model_tab(self):
        """Show architecture and progress information in the Model tab."""
        lines = []
        if self.trainer is None:
            lines.append("No model yet. Start a training or load a checkpoint.")
        else:
            m, cfg = self.trainer.model, self.trainer.cfg
            best = max((h["val_acc"] for h in self.history), default=float("nan"))
            lines += [
                f"Classes:            {cfg.num_classes}",
                f"Input size:         {cfg.input_size} x {cfg.input_size}",
                f"Parameters:         {m.count_parameters():,}",
                f"Branches per stage: {m.count_branches_per_stage()}",
                f"Leaf paths:         {m.count_leaf_paths()}",
                f"Base channels:      {cfg.base_channels}",
                f"Normalization:      {cfg.normalization} | activation: {cfg.activation}",
                f"Epochs completed:   {self.next_epoch}",
                f"Best val accuracy:  {best:.2%}" if best == best else "Best val accuracy:  -",
                f"Autosave:           {AUTOSAVE}",
            ]
        self.model_text.config(state="normal")
        self.model_text.delete("1.0", "end")
        self.model_text.insert("end", "\n".join(lines))
        self.model_text.config(state="disabled")

    def save_checkpoint(self):
        """Save the current model and training state to a chosen file."""
        if self.trainer is None or (self.worker and self.worker.is_alive()):
            messagebox.showinfo("Save", "Nothing to save (or training is still running).")
            return
        path = filedialog.asksaveasfilename(defaultextension=".pt",
                                            filetypes=[("Checkpoint", "*.pt")])
        if path:
            self.trainer.save_checkpoint(path, history=self.history, next_epoch=self.next_epoch,
                                         class_names=self.class_names)

    def load_checkpoint(self):
        """Load a checkpoint and restore model, history and settings."""
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Load", "Stop training first.")
            return
        path = filedialog.askopenfilename(filetypes=[("Checkpoint", "*.pt"), ("All files", "*")])
        if not path:
            return
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            trainer, ckpt = resume_trainer(path, device=device)
        except Exception as exc:
            messagebox.showerror("Load", f"Could not load the checkpoint:\n{exc}")
            return
        self.trainer = trainer
        self.class_names = list(ckpt.get("class_names") or [])
        self.history = list(ckpt.get("history") or [])
        self.next_epoch = int(ckpt.get("next_epoch", len(self.history)))
        cfg = trainer.cfg
        for key, value in (("size", cfg.input_size), ("depth", cfg.initial_depth),
                           ("branches", cfg.initial_branches),
                           ("max_branches", cfg.max_branches_per_stage),
                           ("channels", cfg.base_channels), ("norm", cfg.normalization),
                           ("act", cfg.activation)):
            self.vars[key].set(str(value))
        self._set_running(False)
        self._redraw()
        self._refresh_model_tab()
        self._append_log(f"Loaded {path}\n")

    # -------------------------------------------------------------- test
    def test_image(self):
        """Classify a user-chosen image with the current model."""
        if self.trainer is None:
            messagebox.showinfo("Test", "Train or load a model first.")
            return
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Test", "Stop training before testing an image.")
            return
        patterns = " ".join(f"*{e}" for e in sorted(IMAGE_EXTENSIONS))
        path = filedialog.askopenfilename(filetypes=[("Images", patterns), ("All files", "*")])
        if not path:
            return
        try:
            image, top = predict(self.trainer.model, self.trainer.cfg, path, self.class_names, 5)
        except Exception as exc:
            messagebox.showerror("Test image", str(exc))
            return
        thumb = image.copy()
        thumb.thumbnail((280, 280))
        self._photo = ImageTk.PhotoImage(thumb)
        self.preview.config(image=self._photo)
        self.prediction_label.config(
            text="\n".join(f"{p:6.1%}  {name}" for name, p in top))

    # ------------------------------------------------------------- close
    def _on_close(self, tries: int = 0):
        """Stop the worker (if any) and close the window."""
        if self.worker is not None and self.worker.is_alive() and tries < 50:
            self.stop_event.set()
            self.after(200, lambda: self._on_close(tries + 1))
            return
        self.destroy()


def main():
    """Open the application window."""
    App().mainloop()


if __name__ == "__main__":
    main()
