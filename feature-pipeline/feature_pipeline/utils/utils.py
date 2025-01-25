import logging
from pathlib import Path
import json

from feature_pipeline.utils import settings


def get_logger(name):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name)

    return logger


def save_json(data, file_name, save_dir=settings.OUTPUT_DIR):

    data_path = Path(save_dir) / file_name
    with open(data_path, "w") as f:
        json.dump(data, f)

def load_json(file_name, save_dir= settings.OUTPUT_DIR):
    
    data_path = Path(save_dir) / file_name
    if not data_path.exists():
        raise FileNotFoundError(f"Cached JSON from {data_path} does not exist.")

    with open(data_path, "r") as f:
        return json.load(f)