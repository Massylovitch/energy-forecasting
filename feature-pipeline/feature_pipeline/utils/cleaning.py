import pandas as pd


def rename_columns(df):
    data = df.copy()

    data.drop(columns=["HourDK"], inplace=True)

    data.rename(
        columns={
            "HourUTC": "datetime_utc",
            "PriceArea": "area",
            "ConsumerType_DE35": "consumer_type",
            "TotalCon": "energy_consumption"
        },
        inplace=True
    )

    return data


def cast_columns(df):

    data = df.copy()

    data["datetime_utc"] = pd.to_datetime(data["datetime_utc"])
    data["area"] = data["area"].astype("string")
    data["consumer_type"] = data["consumer_type"].astype("int32")
    data["energy_consumption"] = data["energy_consumption"].astype("float64")

    return data

def encode_area_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the area column to integers.
    """

    data = df.copy()

    area_mappings = {"DK": 0, "DK1": 1, "DK2": 2}

    data["area"] = data["area"].map(lambda string_area: area_mappings.get(string_area))
    data["area"] = data["area"].astype("int8")

    return data