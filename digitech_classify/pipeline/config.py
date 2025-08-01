
from pathlib import Path  # noqa: I001
import os

from dataclasses import dataclass, field
from typing import List, Tuple
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths from .env
DATA_DIR = Path(os.getenv("DATA_DIR")) # type: ignore
MODELS_DIR = Path(os.getenv("MODELS_DIR"))# type: ignore
REPORTS_DIR = Path(os.getenv("REPORTS_DIR"))# type: ignore

# Subdirectories for data
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Figures directory inside reports
FIGURES_DIR = REPORTS_DIR / "figures"


@dataclass
class ModelConfig:
    """MLP model configuration"""
    hidden_layer_sizes_options: List[Tuple] = field(default_factory=lambda: [
        (128,), (256,), (384,), (512,),
        (128, 64), (256, 128), (384, 128),
        (256, 128, 64), (384, 256, 128)
    ])
    activation_options: List[str] = field(default_factory=lambda: ['relu', 'tanh', 'logistic'])
    alpha_range: Tuple[float, float] = (1e-6, 1e-2)
    learning_rate_range: Tuple[float, float] = (1e-4, 1e-2)
    max_iter_options: List[int] = field(default_factory=lambda: [400, 500, 600, 700, 1000])
    solver_options: List[str] = field(default_factory=lambda: ['adam', 'sgd'])
    random_state: int = 42
    random_state: int = 42









# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm
    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
