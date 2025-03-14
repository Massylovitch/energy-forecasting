import json
from pathlib import Path
import pandas as pd
import wandb
import joblib

from training_pipeline import settings


def load_json(file_name, save_dir=settings.OUTPUT_DIR):
    data_path = Path(save_dir) / file_name
    with open(data_path, "r") as f:
        return json.load(f)


def init_wandb_run(
    name,
    group=None,
    job_type=None,
    add_timestamp_to_name=False,
    run_id=None,
    resume=None,
    reinit=False,
    project=settings.SETTINGS["WANDB_PROJECT"],
    entity=settings.SETTINGS["WANDB_ENTITY"],
):
    if add_timestamp_to_name:
        name = f"{name}_{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=group,
        job_type=job_type,
        id=run_id,
        reinit=reinit,
        resume=resume,
    )

    return run


def save_json(data, file_name, save_dir=settings.OUTPUT_DIR):

    data_path = Path(save_dir) / file_name
    with open(data_path, "w") as f:
        json.dump(data, f)


def save_model(model, model_path):
    joblib.dump(model, model_path)


def check_if_artifact_exists(
    artifact_name,
    project=settings.SETTINGS["WANDB_PROJECT"],
    entity= settings.SETTINGS["WANDB_ENTITY"],
):
    try:
        get_artifact(artifact_name, project, entity)

        return True
    except wandb.errors.CommError:
        return False


def get_artifact(
    artifact_name,
    project=settings.SETTINGS["WANDB_PROJECT"],
    entity=settings.SETTINGS["WANDB_ENTITY"],
):
    api = wandb.Api()
    artifact = api.artifact(f"{entity}/{project}/{artifact_name}:latest")

    return artifact
