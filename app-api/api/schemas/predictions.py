from typing import List

from pydantic import BaseModel
from datetime import datetime


class PredictionResults(BaseModel):
    datetime_utc: List[datetime]
    energy_consumption: List[float]
    preds_datetime_utc: List[datetime]
    preds_energy_consumption: List[float]


class MonitoringMetrics(BaseModel):
    datetime_utc: List[datetime]
    mape: List[float]


class MonitoringValues(BaseModel):
    y_monitoring_datetime_utc: List[datetime]
    y_monitoring_energy_consumption: List[float]
    predictions_monitoring_datetime_utc: List[datetime]
    predictions_monitoring_energy_consumption: List[float]
