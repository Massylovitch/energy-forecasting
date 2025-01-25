import datetime
from feature_pipeline.utils import settings
from feature_pipeline.utils import utils
import requests
import pandas as pd
from pandas.errors import EmptyDataError

logger = utils.get_logger(__name__)

def from_file(
    days_delay: int = 15,
    days_export: int = 30,
    url: str = "https://drive.google.com/uc?export=download&id=1y48YeDymLurOTUO-GeFOUXVNc9MCApG5",
    datetime_format: str = "%Y-%m-%d %H:%M",
    cache_dir=None,
):

    export_start, export_end = _compute_extraction_window(
        days_delay=days_delay, days_export=days_export
    )
    records = _extract_records_from_file_url(
        url=url,
        export_start=export_start,
        export_end=export_end,
        datetime_format=datetime_format,
        cache_dir=cache_dir,
    )

    metadata = {
        "days_delay": days_delay,
        "days_export": days_export,
        "url": url,
        "export_datetime_utc_start": export_start.strftime(datetime_format),
        "export_datetime_utc_end": export_end.strftime(datetime_format),
        "datetime_format": datetime_format,
        "num_unique_samples_per_time_series": len(records["HourUTC"].unique()),
    }
    
    return records, metadata

def _extract_records_from_file_url(
    url, export_start, export_end, datetime_format, cache_dir=None
):
    """Extract records from the file backup based on the given export window."""

    if cache_dir is None:
        cache_dir = settings.OUTPUT_DIR / "data"
        cache_dir.mkdir(parents=True, exist_ok=True)

    file_path = cache_dir / "ConsumptionDE35Hour.csv"
    if not file_path.exists():
        logger.info(f"Downloading data from: {url}")

        try:
            response = requests.get(url)
        except requests.exceptions.HTTPError as e:
            logger.error(
                f"Response status = {response.status_code}. Could not download the file due to: {e}"
            )

            return None

        if response.status_code != 200:
            raise ValueError(f"Response status = {response.status_code}. Could not download the file.")

        with file_path.open("w") as f:
            f.write(response.text)

        logger.info(f"Successfully downloaded data to: {file_path}")
    else:
        logger.info(f"Data already downloaded at: {file_path}")

    try:
        data = pd.read_csv(file_path, delimiter=";")
    except EmptyDataError:
        file_path.unlink(missing_ok=True)

        raise ValueError(f"Downloaded file at {file_path} is empty. Could not load it into a DataFrame.")

    records = data[(data["HourUTC"] >= export_start.strftime(datetime_format)) & (data["HourUTC"] < export_end.strftime(datetime_format))]

    return records


def _compute_extraction_window(days_delay, days_export):

    # The dataset is limited to up to July 2023.
    export_end_reference_datetime = datetime.datetime(2023, 6, 30, 21, 0, 0)

    export_end = export_end_reference_datetime - datetime.timedelta(days=days_delay)
    export_start = export_end_reference_datetime - datetime.timedelta(
        days=days_delay + days_export
    )

    min_export_start = datetime.datetime(2020, 6, 30, 22, 0, 0)
    if export_start < min_export_start:
        export_start = min_export_start
        export_end = export_start + datetime.timedelta(days=days_export)

    return export_start, export_end
