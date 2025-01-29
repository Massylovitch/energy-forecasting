from pydantic import BaseSettings


class Settings(BaseSettings):

    HOST="0.0.0.0"
    PORT=8001
    LOG_LEVEL=LogLevel.INFO

    VERSION="v1"
    WORKERS_COUNT=1
    RELOAD=False
    PROJECT_NAME="Energy Consumption API"

    GCP_PROJECT=None
    GCP_BUCKET=None
    GCP_SERVICE_ACCOUNT_JSON_PATH=None

    class Config:
        env_file=".env"
        env_prefix="APP_API_"
        case_sensitive=False
        env_file_encoding="utf-8"


@lru_cache()
def get_settings():
    return Settings()