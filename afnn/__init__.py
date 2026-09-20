"""AFNN - Adaptive Fractal Neural Network."""
import os

# Must be set before CUDA is initialised. Ignored on Windows (unsupported there).
if os.name == "posix":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .config import AFNNConfig
from .model import AFNN, measure_latency, measure_macs
from .inference import load_model, predict
from .training import (DynamicLRScheduler, EMA, GrowthManager, Trainer,
                       TrainingStopped, build_trainer, fit, resume_trainer)

__all__ = [
    "AFNNConfig", "AFNN", "Trainer", "EMA", "GrowthManager",
    "DynamicLRScheduler", "TrainingStopped", "build_trainer", "fit", "resume_trainer", "load_model", "predict",
    "measure_macs", "measure_latency",
]
__version__ = "1.0.0"
