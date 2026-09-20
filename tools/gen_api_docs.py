"""Generate docs/API.md and docs/CONFIGURATION.md from the source code.

    python tools/gen_api_docs.py
"""
import ast
import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODULES = [
    ("afnn/config.py", "Configuration dataclass."),
    ("afnn/layers.py", "Building blocks."),
    ("afnn/fractal.py", "The recursive fractal block."),
    ("afnn/model.py", "Stages, the full network and measurement helpers."),
    ("afnn/augment.py", "Batch augmentation."),
    ("afnn/data.py", "Image folders, cache and loaders."),
    ("afnn/training.py", "EMA, growth, LR schedule, Trainer, fit."),
    ("afnn/inference.py", "Checkpoint loading and prediction."),
    ("gui/app.py", "Tkinter application."),
    ("scripts/train.py", "Command-line training."),
    ("scripts/predict.py", "Command-line prediction."),
]

FIELD_DOCS = {
    "input_channels": "Channels of the input image (3 = RGB).",
    "input_size": "Side of the square input in pixels; images are resized to it.",
    "num_classes": "Number of output classes.",
    "base_channels": "Channels of the stem and of stage 1; later stages multiply it.",
    "channel_multipliers": "Channel multiplier of each stage relative to ``base_channels``.",
    "stages": "Number of stages (each is a fractal block plus a downsampling conv).",
    "initial_depth": "Recursion depth of the fractal block in every stage.",
    "initial_branches": "Branches per block at the start; nested blocks keep this count.",
    "normalization": "``batchnorm``, ``groupnorm`` or ``none``.",
    "activation": "``relu``, ``gelu``, ``silu`` or ``mish``.",
    "growth_enabled": "Allow the network to add branches during training.",
    "growth_patience": "Epochs without validation-loss improvement before it grows.",
    "growth_min_delta": "Smallest validation-loss decrease that counts as an improvement.",
    "max_branches_per_stage": "Upper limit of root branches per stage.",
    "max_total_params": "Growth stops when the parameter count would exceed this.",
    "new_branch_lr_multiplier": "LR multiplier applied to the parameters of a new branch.",
    "new_branch_warmup_epochs": "Epochs over which a new branch's LR is ramped up.",
    "rejuvenation_enabled": "Re-initialise branches whose output has collapsed.",
    "dead_branch_std_threshold": "A branch with smoothed output std below this is dead.",
    "rejuvenation_check_every": "Check for dead branches every N epochs.",
    "rejuvenation_min_observations": "Vitality samples required before a branch can be judged.",
    "rejuvenation_ema_stride": "Measure branch vitality every N batches (it costs a forward pass).",
    "use_mixup": "Enable Mixup.",
    "mixup_alpha": "Beta(alpha, alpha) parameter of Mixup.",
    "use_cutmix": "Enable CutMix.",
    "cutmix_alpha": "Beta(alpha, alpha) parameter of CutMix.",
    "label_smoothing": "Label smoothing of the cross-entropy loss.",
    "use_load_balance": "Add the load-balancing loss on the merge weights.",
    "load_balance_weight": "Weight of the load-balancing loss.",
    "use_representation_learning": "Add the contrastive representation loss (two extra passes per batch).",
    "representation_dim": "Size of the representation embedding.",
    "projection_dim": "Size of the projection head output.",
    "contrastive_weight": "Weight of the contrastive loss.",
    "invariance_weight": "Weight of the view-invariance loss.",
    "contrastive_temperature": "Temperature of the contrastive loss.",
    "representation_warmup_epochs": "Epochs before the representation loss is switched on.",
    "use_amp": "Mixed precision. Switched off automatically on GPUs older than Volta.",
    "use_ema": "Keep an exponential moving average of the weights.",
    "ema_decay": "EMA decay.",
    "warmup_epochs": "Linear LR warm-up epochs.",
    "cosine_t_max": "Length of the cosine schedule in epochs (the trainer overrides it).",
    "gradient_checkpointing": "Trade compute for memory inside fractal blocks.",
    "architecture_version": "Version tag stored in checkpoints.",
}


def signature(fn: ast.FunctionDef) -> str:
    args = ast.unparse(fn.args)
    ret = f" -> {ast.unparse(fn.returns)}" if fn.returns else ""
    return f"{fn.name}({args}){ret}"


def render_function(fn, level: str) -> list:
    tag = " *(internal)*" if fn.name.startswith("_") and not fn.name.startswith("__") else ""
    out = [f"{level} `{signature(fn)}`{tag}", ""]
    doc = ast.get_docstring(fn)
    out += [doc or "*No description.*", ""]
    return out


def render_module(path: str, blurb: str) -> list:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    out = [f"## `{path}`", "", ast.get_docstring(tree) or blurb, ""]
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            init = next((n for n in node.body
                         if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
            args = ast.unparse(init.args) if init else ""
            args = args.replace("self, ", "").replace("self", "")
            out += [f"### class `{node.name}({args})`", "", ast.get_docstring(node) or "", ""]
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name != "__init__":
                    out += render_function(item, "####")
        elif isinstance(node, ast.FunctionDef):
            out += render_function(node, "###")
    return out


def build_api() -> str:
    lines = ["# API reference", "",
             "Generated from the docstrings by `tools/gen_api_docs.py`; edit the docstrings, "
             "not this file.", "", "| File | Contents |", "|---|---|"]
    for path, blurb in MODULES:
        lines.append(f"| [`{path}`](#{path.replace('/', '').replace('.', '')}) | {blurb} |")
    lines.append("")
    for path, blurb in MODULES:
        lines += render_module(path, blurb)
    return "\n".join(lines).rstrip() + "\n"


def build_config() -> str:
    from afnn.config import AFNNConfig
    lines = ["# Configuration reference", "",
             "Every field of `AFNNConfig` (`afnn/config.py`) with its default. "
             "`build_trainer()` and the GUI/CLI set the most common ones for you.", "",
             "| Field | Default | Meaning |", "|---|---|---|"]
    for f in dataclasses.fields(AFNNConfig):
        default = f.default if f.default is not dataclasses.MISSING else "-"
        lines.append(f"| `{f.name}` | `{default}` | {FIELD_DOCS.get(f.name, '')} |")
    missing = [f.name for f in dataclasses.fields(AFNNConfig) if f.name not in FIELD_DOCS]
    assert not missing, f"undocumented config fields: {missing}"
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "API.md").write_text(build_api(), encoding="utf-8")
    (ROOT / "docs" / "CONFIGURATION.md").write_text(build_config(), encoding="utf-8")
    print("docs written")
