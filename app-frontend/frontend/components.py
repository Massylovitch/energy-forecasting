import requests

import pandas as pd
import plotly.graph_objects as go
from settings import API_URL


def build_data_plot(area, consumer_type):

    response = requests.get(
        API_URL / "predictions" / f"{area}" / f"{consumer_type}", verify=False
    )

    if response.status_code != 200:
        train_df = build_dataframe([], [])
        preds_df = build_dataframe([], [])

        title = "NO DATA AVAILABLE FOR THE GIVEN AREA AND CONSUMER TYPE"
    else:
        json_response = response.json()

        datetime_utc = json_response.get("datetime_utc")
        energy_consumption = json_response.get("energy_consumption")
        pred_datetime_utc = json_response.get("preds_datetime_utc")
        pred_energy_consumption = json_response.get("preds_energy_consumption")

        train_df = build_dataframe(datetime_utc, energy_consumption)
        preds_df = build_dataframe(pred_datetime_utc, pred_energy_consumption)

        title = "Energy Consumption per DE35 Industry Code per Hour"

    fig = go.Figure()
    fig.update_layout(
        title=dict(
            text=title,
        ),
        showlegend=True,
    )

    fig.update_xaxes(title_text="Datetime UTC")
    fig.update_yaxes(title_text="Total Consumption")
    fig.add_scatter(
        x=train_df["datetime_utc"],
        y=train_df["energy_consumption"],
        name="Observations",
        hovertemplate="<br>".join(["Datetime: %{x}", "Energy Consumption: %{y} kWh"]),
    )
    fig.add_scatter(
        x=preds_df["datetime_utc"],
        y=preds_df["energy_consumption"],
        name="Predictions",
        line=dict(color="#FFC703"),
        hovertemplate="<br>".join(["Datetime: %{x}", "Total Consumption: %{y} kWh"]),
    )

    return fig


def build_dataframe(datetime_utc, energy_consumption_values):
    df = pd.DataFrame(
        list(zip(datetime_utc, energy_consumption_values)),
        columns=["datetime_utc", "energy_consumption"],
    )
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])

    df = df.set_index("datetime_utc")
    df = df.resample("h").asfreq()
    df = df.reset_index()

    return df
