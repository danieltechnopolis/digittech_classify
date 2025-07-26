
from pathlib import Path  # noqa: I001
import os

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


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm
    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
