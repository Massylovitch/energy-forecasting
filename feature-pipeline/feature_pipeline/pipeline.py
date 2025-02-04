from feature_pipeline.utils import extract, utils, cleaning, load, validation
import fire

logger = utils.get_logger(__name__)


def run(
    export_end_reference_datetime=None,
    days_delay=15,
    days_export=30,
    url="https://drive.google.com/uc?export=download&id=1y48YeDymLurOTUO-GeFOUXVNc9MCApG5",
    feature_group_version=1,
):

    logger.info("Extracting data from API")

    data, metadata = extract.from_file(
        export_end_reference_datetime,
        days_delay,
        days_export,
        url,
    )

    if metadata["num_unique_samples_per_time_series"] < days_export * 24:
        raise RuntimeError(
            "Could not extract the expected number of samples from the API"
        )

    logger.info("Succesfully extracted data from API")

    logger.info("Transforming data")
    data = transform(data)
    logger.info("Successfully transformed data")

    logger.info("Building validation expectation suite")
    validation_expectation_suite = validation.build_expectation_suite()
    logger.info("Succesfully built validation expectation suite")

    logger.info("Validation data and loading it to the feature store")
    load.to_feature_store(
        data,
        validation_expectation_suite=validation_expectation_suite,
        feature_group_version=feature_group_version,
    )
    metadata["feature_group_version"] = feature_group_version
    logger.info("Successfully validated data and loaded it to the feature store.")

    logger.info("Wrapping up the pipeline.")
    utils.save_json(metadata, file_name="feature_pipeline_metadata.json")
    logger.info("Done")

    return metadata


def transform(data):

    data = cleaning.rename_columns(data)
    data = cleaning.cast_columns(data)
    data = cleaning.encode_area_column(data)

    return data


if __name__ == "__main__":
    fire.Fire(run)
