from pydantic import BaseSettings
import enum
from functools import lru_cache
from typing import Optional


class LogLevel(str, enum.Enum):

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"

class Settings(BaseSettings):

    HOST: str = "0.0.0.0"
    PORT: int = 8001
    LOG_LEVEL: LogLevel = LogLevel.INFO
    VERSION: str = "v1"
    WORKERS_COUNT: int = 1
    RELOAD: bool = False

    PROJECT_NAME: str = "Energy Consumption API"

    # Google Cloud Platform credentials
    GCP_PROJECT: Optional[str] = None
    GCP_BUCKET: Optional[str] = None
    GCP_SERVICE_ACCOUNT_JSON_PATH: Optional[str] = None

    class Config:
        env_file = ".env"
        env_prefix = "APP_API_"
        case_sensitive = False
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings():
    return Settings()
