import fire
from feature_pipeline.utils import utils
from datetime import datetime
import hopsworks
from feature_pipeline.utils import settings
import hsfs


def create(
    feature_group_version=None,
    start_datetime=None,
    end_datetime=None,
):
    if feature_group_version is None:
        feature_pipeline_metadata = utils.load_json("feature_pipeline_metadata.json")
        feature_group_version = feature_pipeline_metadata["feature_group_version"]

    if start_datetime is None or end_datetime is None:
        feature_group_metadata = utils.load_json("feature_pipeline_metadata.json")
        start_datetime = datetime.strptime(
            feature_group_metadata["export_datetime_utc_start"],
            feature_pipeline_metadata["datetime_format"],
        )
        end_datetime = datetime.strptime(
            feature_group_metadata["export_datetime_utc_end"],
            feature_pipeline_metadata["datetime_format"],
        )

    project = hopsworks.login(
        api_key_value=settings.SETTINGS["FS_API_KEY"],
        project=settings.SETTINGS["FS_PROJECT_NAME"],
    )

    fs = project.get_feature_store()

    try:
        feature_views = fs.get_feature_views(name="energy_consumption_denmark_view")
    except:
        feature_views = []

    for feature_view in feature_views:
        try:
            feature_view.delete_all_training_datasets()
        except hsfs.client.exceptions.RestAPIError:
            pass

        try:
            feature_view.delete()
        except hsfs.client.exceptions.RestAPIError:
            pass

    energy_consumption_fg = fs.get_feature_group(
        "energy_consumption", version=feature_group_version
    )

    ds_query = energy_consumption_fg.select_all()
    feature_view = fs.create_feature_view(
        name="energy_consumption_view",
        description="Energy consumption for forecasting model",
        query=ds_query,
        labels=[],
    )

    feature_view.create_training_data(
        descrption="Energy consumption training dataset",
        data_format="csv",
        start_time=start_datetime,
        end_time=end_datetime,
        write_options={"wait_for_job": True},
        coalesce=False,
    )

    metadata = {
        "feature_view_version": feature_view.version,
        "training_dataset_version": 1,
    }

    utils.save_json(metadata, file_name="feature_view_metadata.json")

    return metadata


if __name__ == "__main__":
    fire.Fire(create)
