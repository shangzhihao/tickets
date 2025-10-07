"""Pydantic schemas that describe aggregated support ticket metrics."""

from datetime import timedelta
from typing import Any

from pydantic import BaseModel, Field

from tickets.schemas.ticket import CustomerSentiment


def _format_float(value: float) -> str:
    """Return a normalized string representation for floating-point values."""

    normalized = f"{value:.6f}".rstrip("0").rstrip(".")
    return normalized or "0"


def _stringify(value: Any) -> str:
    """Convert supported metric values into CVS-friendly strings."""

    if isinstance(value, timedelta):
        return _format_float(value.total_seconds())
    if isinstance(value, float):
        return _format_float(value)
    return str(value)


class TimedeltaMetric(BaseModel):
    """Percentile summary statistics computed for timedelta metrics."""

    avg: timedelta = Field(...,
        description="Arithmetic mean across aggregated samples.")
    p50: timedelta = Field(...,
        description="Median (50th percentile) of the distribution.")
    P75: timedelta = Field(...,
        description="75th percentile of the distribution.")
    p90: timedelta = Field(...,
        description="90th percentile of the distribution.")


class NumberMetric(BaseModel):
    avg: float
    p50: float
    P75: float
    p90: float


class ResponseTimeStats(BaseModel):
    """Response time distribution and volume for a ticket segment."""

    volume: float = Field(...,
        description="Number of tickets represented in the segment.")
    res_time: TimedeltaMetric = Field(...,
        description="Response time summary statistics.")

    def to_dict(self) -> dict[str, str]:
        """Flatten response-time metrics into column-value pairs."""

        payload: dict[str, str] = {"volume": _stringify(self.volume)}
        payload.update(
            {
                f"res_time.{metric_key}": _stringify(metric_value)
                for metric_key, metric_value in self.res_time.
                model_dump(mode="python").items()
            }
        )
        return payload


class SatisfactionScoreStats(BaseModel):
    """Satisfaction score distribution and volume for a ticket segment."""

    volume: float = Field(...,
        description="Number of tickets represented in the segment.")
    sat_score: NumberMetric = Field(...,
        description="Satisfaction score summary statistics.")

    def to_dict(self) -> dict[str, str]:
        """Flatten satisfaction-score metrics into column-value pairs."""

        payload: dict[str, str] = {"volume": _stringify(self.volume)}
        payload.update(
            {
                f"sat_score.{metric_key}": _stringify(metric_value)
                for metric_key, metric_value in self.sat_score.
                model_dump(mode="python").items()
            }
        )
        return payload


class AnalysisRes(BaseModel):
    """Aggregated metric view across all tickets and sentiment segments."""

    res_time_all: TimedeltaMetric = Field(
        ..., description="Response time metrics computed across all tickets."
    )
    sat_score_all: NumberMetric = Field(
        ..., description="Satisfaction score metrics computed across all tickets."
    )
    res_time_by_senti: dict[CustomerSentiment, ResponseTimeStats] = Field(
        ..., description="Response time metrics grouped by customer sentiment."
    )
    sat_score_by_senti: dict[CustomerSentiment, SatisfactionScoreStats] = Field(
        ..., description="Satisfaction score metrics grouped by customer sentiment."
    )

    def to_dict(self) -> dict[str, str]:
        """
        Flatten aggregate metrics, including sentiment splits,
        into column-value pairs.
        """

        payload: dict[str, str] = {}
        payload.update(
            {
                f"res_time_all.{metric_key}": _stringify(metric_value)
                for metric_key, metric_value in self.res_time_all.
                model_dump(mode="python").items()
            }
        )
        payload.update(
            {
                f"sat_score_all.{metric_key}": _stringify(metric_value)
                for metric_key, metric_value in self.sat_score_all.
                model_dump(mode="python").items()
            }
        )

        for sentiment, stats in self.res_time_by_senti.items():
            base_key = f"res_time_by_senti.{sentiment.value}"
            payload.update(
                {
                    f"{base_key}.{metric_key}": metric_value
                    for metric_key, metric_value in stats.to_dict().items()
                }
            )

        for sentiment, stats in self.sat_score_by_senti.items():
            base_key = f"sat_score_by_senti.{sentiment.value}"
            payload.update(
                {
                    f"{base_key}.{metric_key}": metric_value
                    for metric_key, metric_value in stats.to_dict().items()
                }
            )

        return payload
