import pandas as pd
from training_pipeline import utils
from training_pipeline.data import load_dataset_from_feature_store
import wandb
from training_pipeline.configs import gridsearch as gridsearch_configs
from training_pipeline.settings import SETTINGS, OUTPUT_DIR
from training_pipeline.utils import init_wandb_run
from training_pipeline.models import build_model
from functools import partial
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.forecasting.model_evaluation import evaluate as cv_evaluate
import numpy as np
import fire
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from sktime.utils.plotting import plot_windows
from matplotlib import pyplot as plt


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

    sweep_id = run_hyperparameter_optimization(y_train, X_train, fh=fh)
    metadata = {"sweep_id": sweep_id}

    utils.save_json(metadata, file_name="last_sweep_metadata.json")

    return metadata


def run_hyperparameter_optimization(y_train, X_train, fh):

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
        name="experiment", job_type="hpo", group="train", add_timestamp_to_name=True
    ) as run:

        run.use_artifact("split_train:latest")

        config = wandb.config
        config = dict(config)

        model = build_model(config)
        model, results = train_model_cv(model, y_train, X_train, fh=fh)
        wandb.log(results)

        metadata = {
            "experiment": {"name": run.name, "fh": fh},
            "result": results,
            "config": config,
        }

        artifact = wandb.Artifact(name="config", type="model", metadata=metadata)

        run.log_artifact(artifact)
        run.finish()


def train_model_cv(model, y_train, X_train, fh, k=3):

    data_length = len(y_train.index.get_level_values(-1).unique())
    assert data_length >= fh * 10, "Not enough data to perform a 3 fold CV."

    cv_step_length = data_length // k
    initial_window = max(fh * 3, cv_step_length - fh)
    cv = ExpandingWindowSplitter(
        step_length=cv_step_length, fh=np.arange(fh) + 1, initial_window=initial_window
    )
    render_cv_scheme(cv, y_train)

    results = cv_evaluate(
        forecaster=model,
        y=y_train,
        X=X_train,
        cv=cv,
        strategy="refit",
        scoring=MeanAbsolutePercentageError(symmetric=False),
        error_score="raise",
        return_data=False,
    )

    results = results.rename(
        columns={
            "test_MeanAbsolutePercentageError": "MAPE",
            "fit_time": "fit_time",
            "pred_time": "prediction_time",
        }
    )
    mean_results = results[["MAPE", "fit_time", "prediction_time"]].mean(axis=0)
    mean_results = mean_results.to_dict()
    results = {"validation": mean_results}

    return model, results


def render_cv_scheme(cv, y_train):

    random_time_series = (
        y_train.groupby(level=[0, 1])
        .get_group((1, 111))
        .reset_index(level=[0, 1], drop=True)
    )
    plot_windows(cv, random_time_series)

    save_path = str(OUTPUT_DIR / "cv_scheme.png")
    plt.savefig(save_path)
    wandb.log({"cv_scheme": wandb.Image(save_path)})

    return save_path


if __name__ == "__main__":
    fire.Fire(run)
