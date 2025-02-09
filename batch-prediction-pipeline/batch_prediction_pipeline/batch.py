from batch_prediction_pipeline import utils
from datetime import datetime
import hopsworks
from batch_prediction_pipeline import settings
from batch_prediction_pipeline import data
from pathlib import Path
import pandas as pd


def predict(
    fh=24,
    feature_view_version=None,
    model_version=None,
    start_datetime=None,
    end_datetime=None,
):

    if feature_view_version is None:
        feature_view_metadata = utils.load_json("feature_view_metadata.json")
        feature_view_version = feature_view_metadata["feature_view_version"]

    if model_version is None:
        model_metadata = utils.load_json("train_metadata.json")
        model_version = model_metadata["model_version"]

    if start_datetime is None:
        feature_pipeline_metadata = utils.load_json("feature_pipeline_metadata.json")
        start_datetime = datetime.strptime(
            feature_pipeline_metadata["export_datetime_utc_start"],
            feature_pipeline_metadata["datetime_format"],
        )
    if end_datetime is None:
        feature_pipeline_metadata = utils.load_json("feature_pipeline_metadata.json")
        end_datetime = datetime.strptime(
            feature_pipeline_metadata["export_datetime_utc_end"],
            feature_pipeline_metadata["datetime_format"],
        )

    project = hopsworks.login(
        api_key_value=settings.SETTINGS["FS_API_KEY"],
        project=settings.SETTINGS["FS_PROJECT_NAME"],
    )
    fs = project.get_feature_store()

    X, y = data.load_data_from_feature_store(
        fs,
        feature_view_version,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )

    model = load_model_from_model_registry(project, model_version)

    predictions = forecast(model, X, fh=fh)
    predictions_start_datetime = predictions.index.get_level_values(
        level="datetime_utc"
    ).min()
    predictions_end_datetime = predictions.index.get_level_values(
        level="datetime_utc"
    ).max()

    save(X, y, predictions)
    save_for_monitoring(predictions, start_datetime)


def load_model_from_model_registry(project, model_version):

    mr = project.get_model_registry()
    model_registry_reference = mr.get_model(name="best_model", version=model_version)
    model_dir = model_registry_reference.download()
    model_path = Path(model_dir) / "best_model.pkl"

    model = utils.load_model(model_path)

    return model


def forecast(model, X, fh=24):

    all_areas = X.index.get_level_values(level=0).unique()
    all_consumer_types = X.index.get_level_values(level=1).unique()
    latest_datetime = X.index.get_level_values(level=2).max()

    start = latest_datetime + 1
    end = start + fh - 1
    fh_range = pd.date_range(
        start=start.to_timestamp(), end=end.to_timestamp(), freq="H"
    )
    fh_range = pd.PeriodIndex(fh_range, freq="H")

    index = pd.MultiIndex.from_product(
        [all_areas, all_consumer_types, fh_range],
        names=["area", "consumer_type", "datetime_utc"],
    )
    X_forecast = pd.DataFrame(index=index)
    X_forecast["area_exog"] = X_forecast.index.get_level_values(0)
    X_forecast["consumer_type_exog"] = X_forecast.index.get_level_values(1)

    predictions = model.predict(X=X_forecast)

    return predictions


def save(X, y, predictions):

    bucket = utils.get_bucket()

    for df, blob_name in zip(
        [X, y, predictions], ["X.parquet", "y.parquet", "predictions.parquet"]
    ):
        utils.write_blob_to(
            bucket=bucket,
            blob_name=blob_name,
            data=df,
        )


def save_for_monitoring(predictions, start_datetime):

    bucket = utils.get_bucket()

    cached_predictions = utils.read_blob_from(
        bucket=bucket, blob_name=f"predictions_monitoring.parquet"
    )
    has_cached_predictions = cached_predictions is not None
    if has_cached_predictions is True:
        
        # cached_predictions.index = cached_predictions.index.set_levels(
        #     pd.to_datetime(cached_predictions.index.levels[2], unit="h").to_period("H"),
        #     level=2,
        # )

        merged_predictions = predictions.merge(
            cached_predictions,
            left_index=True,
            right_index=True,
            how="outer",
            suffixes=("_new", "_cached"),
        )
        new_predictions = merged_predictions.filter(regex=".*?_new")
        new_predictions.columns = new_predictions.columns.str.replace("_new", "")
        cached_predictions = merged_predictions.filter(regex=".*?_cached")
        cached_predictions.columns = cached_predictions.columns.str.replace(
            "_cached", ""
        )

        new_predictions.update(cached_predictions)
        predictions = new_predictions

    predictions = predictions.loc[
        predictions.index.get_level_values("datetime_utc")
        >= pd.Period(start_datetime, freq="H")
    ]
    predictions = predictions.dropna(subset=["energy_consumption"])

    utils.write_blob_to(
        bucket=bucket,
        blob_name=f"predictions_monitoring.parquet",
        data=predictions,
    )


if __name__ == "__main__":
    predict()
