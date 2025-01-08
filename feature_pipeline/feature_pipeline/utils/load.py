import hopsworks
import pandas as pd
from feature_pipeline.utils.settings import SETTINGS


def to_feature_store(
        data,
        validation_expectation_suite,
        feature_group_version
):
    
    project = hopsworks.login(
        api_key_value=SETTINGS["FS_API_KEY"], project=SETTINGS["FS_PROJECT_NAME"]
    )
    feature_store = project.get_feature_store()

    energy_feature_group = feature_store.get_or_create_feature_group(
        name="energy_consumption",
        version=feature_group_version,
        primary_key=["area", "consumer_type"],
        description="Denmark hourly energy consumption data. Data is uploaded with an 15 days delay.",
        online_enabled=False,
        event_time="datetime_utc",
        expectation_suite=validation_expectation_suite
    )

    energy_feature_group.insert(
        features=data,
        overwrite=False,
        write_options={"wait_for_job": True}
    )

    # Add feature descriptions.
    feature_descriptions = [
        {
            "name": "datetime_utc",
            "description": """
                            Datetime interval in UTC when the data was observed.
                            """,
            "validation_rules": "Always full hours, i.e. minutes are 00",
        },
        {
            "name": "area",
            "description": """
                            Denmark is divided in two price areas, divided by the Great Belt: DK1 and DK2.
                            If price area is “DK”, the data covers all Denmark.
                            """,
            "validation_rules": "0 (DK), 1 (DK1) or 2 (Dk2) (int)",
        },
        {
            "name": "consumer_type",
            "description": """
                            The consumer type is the Industry Code DE35 which is owned by Danish Energy. 
                            The code is used by Danish energy companies.
                            """,
            "validation_rules": ">0 (int)",
        },
        {
            "name": "energy_consumption",
            "description": "Total electricity consumption in kWh.",
            "validation_rules": ">=0 (float)",
        },
    ]

    for description in feature_descriptions:
        energy_feature_group.update_feature_description(
            description["name"],
            description["description"],
        )

    energy_feature_group.statistics_config = {
        "enabled": True,
        "histograms": True,
        "correlations": True,
    }
    energy_feature_group.update_statistics_config()
    energy_feature_group.compute_statistics()

    return energy_feature_group