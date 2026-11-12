"""Training and prediction with CaDRiN."""

__version__ = "1.0.0"

from .bundle import FrozenCaDRiNPredictor
from .predict import predict_dataset
from .train import train_from_config

__all__ = ["FrozenCaDRiNPredictor", "predict_dataset", "train_from_config"]
