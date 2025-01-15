from training_pipeline import utils
from training_pipeline.data import load_dataset_from_feature_store
from pathlib import Path
import json
from training_pipeline.models import build_baseline_model, build_model
import numpy as np
import fire
from sktime.performance_metrics.forecasting import (
    mean_squared_percentage_error,
    mean_absolute_percentage_error,
)
import pandas as pd
import wandb
from collections import OrderedDict
import os
import matplotlib.pyplot as plt
from training_pipeline.settings import SETTINGS, OUTPUT_DIR
from sktime.utils.plotting import plot_series
import hopsworks


def from_best_config(fh=24, feature_view_version=None, feature_dataset_version=None):

    feature_view_metadata = utils.load_json("feature_view_metadata.json")
    if feature_view_version is None:
        feature_view_version = feature_view_metadata["feature_view_version"]
    if feature_dataset_version is None:
        training_dataset_version = feature_view_metadata["training_dataset_version"]

    y_train, y_test, X_train, X_test = load_dataset_from_feature_store(
        feature_view_version=feature_view_version,
        training_dataset_version=training_dataset_version,
        fh=fh,
    )

    training_start_datetime = y_train.index.get_level_values("datetime_utc").min()
    training_end_datetime = y_train.index.get_level_values("datetime_utc").max()
    testing_start_datetime = y_test.index.get_level_values("datetime_utc").min()
    testing_end_datetime = y_test.index.get_level_values("datetime_utc").max()

    with utils.init_wandb_run(
        name="best_model",
        job_type="train_best_model",
        group="trian",
        reinit=True,
        add_timestamp_to_name=True,
    ) as run:
        run.use_artifact("split_train:latest")
        run.use_artifact("split_test:latest")

        best_config_artifact = run.use_artifact("best_config:latest", type="model")
        dowload_dir = best_config_artifact.download()
        config_path = Path(dowload_dir) / "best_config.json"

        with open(config_path, "r") as f:
            config = json.load(f)

        run.config.update(config)

        baseline_forecaster = build_baseline_model(seasonal_periodicity=fh)
        baseline_forecaster = train_model(baseline_forecaster, y_train, X_train, fh=fh)
        _, metrics_baseline = evaluate(baseline_forecaster, y_test, X_test)

        slices = metrics_baseline.pop("slices")
        wandb.log({"test": {"baseline": metrics_baseline}})
        wandb.log({"test.baseline.slices": wandb.Table(dataframe=slices)})

        # Build & train best model.
        best_model = build_model(config)
        best_forecaster = train_model(best_model, y_train, X_train, fh=fh)

        # Evaluate best model
        y_pred, metrics = evaluate(best_forecaster, y_test, X_test)
        slices = metrics.pop("slices")
        wandb.log({"test": {"model": metrics}})
        wandb.log({"test.model.slices": wandb.Table(dataframe=slices)})

        results = OrderedDict({"y_train": y_train, "y_test": y_test, "y_pred": y_pred})
        render(results, prefix="images_test")


        best_forecaster = train_model(
            model=best_forecaster,
            y_train=pd.concat([y_train, y_test]).sort_index(),
            X_train=pd.concat([X_train, X_test]).sort_index(),
            fh=fh,
        )
        X_forecast = compute_forecast_exogenous_variables(X_test, fh)
        y_forecast = forecast(best_forecaster, X_forecast)
        
        results = OrderedDict(
            {
                "y_train": y_train,
                "y_test": y_test,
                "y_forecast": y_forecast,
            }
        )

        render(results, prefix="images_forecast")

        save_model_path = OUTPUT_DIR / "best_model.pkl"
        utils.save_model(best_forecaster, save_model_path)
        metadata = {
            "experiment": {
                "fh": fh,
                "feature_view_version": feature_view_version,
                "training_dataset_version": training_dataset_version,
                "training_start_datetime": training_start_datetime.to_timestamp().isoformat(),
                "training_end_datetime": training_end_datetime.to_timestamp().isoformat(),
                "testing_start_datetime": testing_start_datetime.to_timestamp().isoformat(),
                "testing_end_datetime": testing_end_datetime.to_timestamp().isoformat(),
            },
            "results": {"test": metrics},
        }

        artifact = wandb.Artifact(name="best_model", type="model", metadata=metadata)
        artifact.add_file(str(save_model_path))
        run.log_artifact(artifact)

        run.finish()
        artifact.wait()

        model_version = add_best_model_to_model_registry(artifact)

    metadata = {"model_version": model_version}
    utils.save_json(metadata, file_name="train_metadata.json")

    return metadata

def train_model(model, y_train, X_train, fh):

    fh = np.arange(fh) + 1
    model.fit(y_train, X_train, fh=fh)

    return model


def evaluate(model, y_test, X_test):

    y_pred = model.predict(X=X_test)

    results = dict()
    rmspe = mean_squared_percentage_error(y_test, y_pred, squared=False)
    results["RMSPE"] = rmspe
    mape = mean_absolute_percentage_error(y_test, y_pred, symmetric=False)
    results["MAPE"] = mape

    y_test_slices = y_test.groupby(["area", "consumer_type"])
    y_pred_slices = y_pred.groupby(["area", "consumer_type"])
    slices = pd.DataFrame(columns=["area", "consumer_type", "RMSPE", "MAPE"])
    for y_test_slice, y_pred_slice in zip(y_test_slices, y_pred_slices):
        (area_y_test, consumer_type_y_test), y_test_slice_data = y_test_slice
        (area_y_pred, consumer_type_y_pred), y_pred_slice_data = y_pred_slice

        assert (
            area_y_test == area_y_pred and consumer_type_y_test == consumer_type_y_pred
        ), "Slices are not aligned."

        rmspe_slice = mean_squared_percentage_error(
            y_test_slice_data, y_pred_slice_data, squared=False
        )
        mape_slice = mean_absolute_percentage_error(
            y_test_slice_data, y_pred_slice_data, symmetric=False
        )

        slice_results = pd.DataFrame(
            {
                "area": [area_y_test],
                "consumer_type": [consumer_type_y_test],
                "RMSPE": [rmspe_slice],
                "MAPE": [mape_slice],
            }
        )
        slices = pd.concat([slices, slice_results], ignore_index=True)

    results["slices"] = slices

    return y_pred, results


def render(
    timeseries,
    prefix=None,
    delete_from_disk=True,
):

    grouped_timeseries = OrderedDict()
    for split, df in timeseries.items():
        df = df.reset_index(level=[0, 1])
        groups = df.groupby(["area", "consumer_type"])
        for group_name, split_group_values in groups:
            group_values = grouped_timeseries.get(group_name, {})

            grouped_timeseries[group_name] = {
                f"{split}": split_group_values["energy_consumption"],
                **group_values,
            }

    output_dir = OUTPUT_DIR / prefix if prefix else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for group_name, group_values_dict in grouped_timeseries.items():
        fig, ax = plot_series(
            *group_values_dict.values(), labels=group_values_dict.keys()
        )
        fig.suptitle(f"Area: {group_name[0]} - Consumer type: {group_name[1]}")

        # save matplotlib image
        image_save_path = str(output_dir / f"{group_name[0]}_{group_name[1]}.png")
        plt.savefig(image_save_path)
        plt.close(fig)

        if prefix:
            wandb.log({prefix: wandb.Image(image_save_path)})
        else:
            wandb.log(wandb.Image(image_save_path))

        if delete_from_disk:
            os.remove(image_save_path)


def compute_forecast_exogenous_variables(X_test, fh):

    X_forecast = X_test.copy()
    X_forecast.index.set_levels(
        X_forecast.index.levels[-1] + fh, level=-1, inplace=True
    )

    return X_forecast

def forecast(forecaster, X_forecast):

    return forecaster.predict(X=X_forecast)


def add_best_model_to_model_registry(best_model_artifact):

    project = hopsworks.login(
        api_key_value=SETTINGS["FS_API_KEY"], project=SETTINGS["FS_PROJECT_NAME"]
    )

    best_model_dir = best_model_artifact.download()
    best_model_path = best_model_dir + "/" + "best_model.pkl"
    best_model_metrics = best_model_artifact.metadata["results"]["test"]

    mr = project.get_model_registry()
    py_model = mr.python.create_model("best_model", metrics=best_model_metrics)
    py_model.save(best_model_path)

    return py_model.version



if __name__ == "__main__":
    fire.Fire(from_best_config)
