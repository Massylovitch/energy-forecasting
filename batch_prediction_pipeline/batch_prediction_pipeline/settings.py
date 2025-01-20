from pathlib import Path
import os
from dotenv import load_dotenv

def get_root_dir(default_value="."):
    return Path(os.getenv("ML_PIPELINE_ROOT_DIR", default_value))

def load_env_vars(root_dir):
    if isinstance(root_dir, str):
        root_dir = Path(root_dir)

    load_dotenv(dotenv_path=root_dir / ".env")

    return dict(os.environ)

ML_PIPELINE_ROOT_DIR = get_root_dir()
OUTPUT_DIR = ML_PIPELINE_ROOT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS= load_env_vars(
    root_dir=ML_PIPELINE_ROOT_DIR
)