import os
from pathlib import Path

PROJECT_ROOT = Path(os.getcwd()).parent
DATA_PATH = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
SAMPLING_RATE = 20000  # in Hz