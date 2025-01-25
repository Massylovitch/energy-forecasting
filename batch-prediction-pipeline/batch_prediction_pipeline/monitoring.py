from batch_prediction_pipeline import utils
import pandas as pd
import hopsworks
from batch_prediction_pipeline.settings import SETTINGS
from batch_prediction_pipeline import data
import numpy as np
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error


def compute(feature_view_version=None):
    
    if feature_view_version is None:
        feature_view_metadata = utils.load_json("feature_view_metadata.json")
        feature_view_version = feature_view_metadata["feature_view_version"]

    bucket = utils.get_bucket()
    predictions = utils.read_blob_from(
        bucket=bucket, blob_name="predictions_monitoring.parquet"
    )

    if predictions is None or len(predictions) == 0:
        return
    
    # predictions.index = predictions.index.set_levels(
    #     pd.to_datetime(predictions.index.levels[2], unit="h").to_period("H"), level=2
    # )
    
    project = hopsworks.login(
        api_key_value=SETTINGS["FS_API_KEY"], project=SETTINGS["FS_PROJECT_NAME"]
    )

    fs = project.get_feature_store()
    predictions_min_datetime_utc = (
        predictions.index.get_level_values("datetime_utc").min().to_timestamp()
    )
    predictions_max_datetime_utc = (
        predictions.index.get_level_values("datetime_utc").max().to_timestamp()
    )
    
    _, latest_observations = data.load_data_from_feature_store(
        fs,
        feature_view_version,
        start_datetime=predictions_min_datetime_utc,
        end_datetime=predictions_max_datetime_utc,
    )

    if len(latest_observations) == 0:
        return

    predictions = predictions.rename(
        columns={"energy_consumption": "energy_consumption_predicitons"}
    )

    latest_observations = latest_observations.rename(
        columns={"energy_consumption": "energy_consumption_observations"}
    )

    predictions["energy_consumption_observations"] = np.nan
    predictions.update(latest_observations)

    predictions = predictions.dropna(subset=["energy_consumption_observations"])
    if len(predictions) == 0:
        return

    mape_metrics = predictions.groupby("datetime_utc").apply(
        lambda x: mean_absolute_percentage_error(
            x["energy_consumption_observations"],
            x["energy_consumption_predicitons"],
            symmetric=False,
        )
    )

    mape_metrics = mape_metrics.rename("MAPE")
    metrics = mape_metrics.to_frame()

    utils.write_blob_to(
        bucket=bucket,
        blob_name="metrics_monitoring.parquet",
        data=metrics,
    )

    latest_observations = latest_observations.raname(
        columns={"energy_comsumption_observations": "energy_consumption"}
    )
    utils.write_blob_to(
        bucket=bucket,
        blob_name="y_monitoring.parquet",
        data=latest_observations[["energy_consumption"]],
    )


if __name__ == "__main__":
    compute()
