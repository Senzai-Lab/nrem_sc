import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(os.getcwd()).parent

DATA_PATH = Path(os.environ.get("DATA_PATH"))
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
INTERIM_DATA_PATH = DATA_PATH / "interim"
EXTERNAL_DATA_PATH = DATA_PATH / "external"
SAMPLING_RATE = 20000  # in Hz