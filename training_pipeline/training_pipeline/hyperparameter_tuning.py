import pandas as pd
from training_pipeline import utils
from training_pipeline.data import load_dataset_from_feature_store
import wandb
from training_pipeline.configs import gridsearch as gridsearch_configs
from training_pipeline.settings import SETTINGS
from training_pipeline.utils import init_wandb_run
from training_pipeline.models import build_model
from functools import partial

import fire

def run(
    fh=24,
    feature_view_version=None,
    training_dataset_version=None,
):

    feature_view_metadata = utils.load_json("feature_view_metadata.json")
    if feature_view_version is None:
        feature_view_version = feature_view_metadata["feature_view_version"]
    if training_dataset_version is None:
        training_dataset_version = feature_view_metadata["training_dataset_version"]

    y_train, _, X_train, _ = load_dataset_from_feature_store(
        feature_view_version=feature_view_version,
        training_dataset_version=training_dataset_version,
        fh=fh,
    )

    sweep_id = run_hyperparameter_optimization(X_train, X_train, fh=fh)
    metadata = {"weep_id": sweep_id}

    utils.save_json(metadata, file_name="last_sweep_metadata.json")

    return metadata


def run_hyperparameter_optimization(
    y_train,
    X_train,
    fh
):

    sweep_id = wandb.sweep(
        sweep=gridsearch_configs.sweep_configs, project=SETTINGS["WANDB_PROJECT"]
    )

    wandb.agent(
        project=SETTINGS["WANDB_PROJECT"],
        sweep_id=sweep_id,
        function=partial(run_sweep, y_train=y_train, X_train=X_train, fh=fh),
    )

    return sweep_id


def run_sweep(y_train, X_train, fh):
    with init_wandb_run(
        name="experiment", job_type="hpo", group="train", add_timestamp=True
    ) as run:

        run.use_artifact("split_train:latest")

        ocnfig = wandb.config
        config=dict(config)

        model = build_model(config)
        model, results = train_model_cv(model, y_train, X_train, fh=fh)
        wandb.log(results)

        metadata = {
            "experiment": {"name": run.name, "fh": fh},
            "result": results,
            "config": config
        }

        artifact = wandb.Artifact(
            name="config",
            type="model",
            metadata=metadata
        )

        run.log_artifact(artifact)
        run.finish()


def train_model_cv(model, y_train, X_train, fh, k=3):
    print(len(y_train))
    data_length = len(y_train)


if __name__ == "__main__":
    fire.Fire(run)
