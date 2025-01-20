import json
from pathlib import Path
import joblib
import pandas as pd
from batch_prediction_pipeline import settings
from google.cloud import storage

def load_json(file_name, save_dir=settings.OUTPUT_DIR):
    data_path = Path(save_dir) / file_name
    with open(data_path, "r") as f:
        return json.load(f)


def load_model(model_path):
    return joblib.load(model_path)

def get_bucket(
    bucket_name: str = settings.SETTINGS["GOOGLE_CLOUD_BUCKET_NAME"],
    bucket_project: str = settings.SETTINGS["GOOGLE_CLOUD_PROJECT"],
    json_credentials_path: str = settings.SETTINGS[
        "GOOGLE_CLOUD_SERVICE_ACCOUNT_JSON_PATH"
    ],
) :

    storage_client = storage.Client.from_service_account_json(
        json_credentials_path=json_credentials_path,
        project=bucket_project,
    )
    bucket = storage_client.bucket(bucket_name=bucket_name)

    return bucket

def write_blob_to(bucket, blob_name, data):

    blob = bucket.blob(blob_name=blob_name)
    with blob.open("wb") as f:
        data.to_parquet(f)

def read_blob_from(bucket, blob_name):

    blob = bucket.blob(blob_name=blob_name)
    if not blob.exists():
        return None

    with blob.open("rb") as f:
        return pd.read_parquet(f)