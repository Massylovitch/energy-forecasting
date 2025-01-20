from datetime import datetime
from typing import Tuple
import pandas as pd
from hsfs.feature_store import FeatureStore


def load_data_from_feature_store(
    fs, feature_view_version, start_datetime, end_datetime, target="energy_consumption"
):

    feature_view = fs.get_feature_view(
        name="energy_consumption_view", version=feature_view_version
    )
    data = feature_view.get_batch_data(start_time=start_datetime, end_time=end_datetime)

    data["datetime_utc"] = pd.PeriodIndex(data["datetime_utc"], freq="H")
    data = data.set_index(["area", "consumer_type", "datetime_utc"]).sort_index()

    X = data.drop(columns=[target])
    y = data[[target]]

    return X, y
