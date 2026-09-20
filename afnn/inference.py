"""Load a trained checkpoint and classify single images."""
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import open_rgb
from .training import Trainer


def load_model(path, device: Optional[str] = None):
    """Load a checkpoint written by ``Trainer.save_checkpoint``.

        Returns ``(model, cfg, class_names)``; ``class_names`` is empty if the
        checkpoint does not contain them.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, ckpt = Trainer.load_checkpoint(str(path), device=device)
    return model.eval(), cfg, list(ckpt.get("class_names") or [])


def preprocess(image: Image.Image, size: int) -> torch.Tensor:
    """Resize to ``size x size`` and convert to a ``[1, 3, size, size]`` float tensor."""
    arr = np.asarray(image.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).float().div(255.0).unsqueeze(0)


@torch.no_grad()
def predict(model, cfg, image_path, class_names: Optional[List[str]] = None,
            top_k: int = 5) -> Tuple[Image.Image, List[Tuple[str, float]]]:
    """Classify one image file.

    Returns the opened image and the ``top_k`` predictions as ``(label, probability)``.
    """
    device = next(model.parameters()).device
    image = open_rgb(image_path)
    x = preprocess(image, int(cfg.input_size)).to(device)
    was_training = model.training
    model.eval()
    try:
        probs = F.softmax(model(x), dim=1)[0]
    finally:
        model.train(was_training)
    values, indices = probs.topk(min(top_k, probs.numel()))
    names = class_names or []
    result = [(names[i] if i < len(names) else f"class {i}", float(v))
              for v, i in zip(values.tolist(), indices.tolist())]
    return image, result
