"""Utilities for computing aggregate ticket metrics from offline data."""

from __future__ import annotations

import io
from datetime import timedelta
from typing import Final

import pandas as pd
from pandas.api.types import is_timedelta64_dtype

from tickets.schemas.events import (
    DataLoadOfflineEvent,
    DataResTimeStatsEvent,
    DataSatScoreStatsEvent,
    DataStatsSaveEvent,
)
from tickets.schemas.metrics import (
    AnalysisRes,
    NumberMetric,
    ResponseTimeStats,
    SatisfactionScoreStats,
    TimedeltaMetric,
)
from tickets.schemas.ticket import CustomerSentiment

from ..utils.config_util import cfg
from ..utils.io_util import data_logger, s3_client

OFFLINE_PATH: Final[str] = cfg.data.offline_file
BUCKET_NAME: Final[str] = cfg.data.bucket
METRICS_PATH: Final[str] = cfg.data.metrics_file

SENTIMENT_COLUMN: Final[str] = "customer_sentiment"
RESOLVED_AT_COLUMN: Final[str] = "resolved_at"
CREATED_AT_COLUMN: Final[str] = "created_at"
SAT_SCORE_COLUMN: Final[str] = "satisfaction_score"

RES_TIME_COLUMN: Final[str] = "res_time"

REQUIRED_COLUMNS = [
    CREATED_AT_COLUMN,
    RESOLVED_AT_COLUMN,
    SAT_SCORE_COLUMN,
    SENTIMENT_COLUMN,
]


class OfflineMetricsAnalyzer:
    """Compute response-time and satisfaction metrics for offline ticket data."""

    def __init__(self, offline_df: pd.DataFrame) -> None:
        missing_columns = set(REQUIRED_COLUMNS) - set(offline_df.columns)
        if missing_columns:
            msg = f"Offline dataframe missing requiredcolumns: {sorted(missing_columns)}"
            data_logger.error(msg)
            raise KeyError(msg)
        self.analysis_res: AnalysisRes
        self.df = offline_df.loc[:, REQUIRED_COLUMNS].copy()
        self.df[RES_TIME_COLUMN] = self.df[RESOLVED_AT_COLUMN] - self.df[CREATED_AT_COLUMN]

    @classmethod
    def from_s3(cls) -> OfflineMetricsAnalyzer:
        """Load the offline ticket dataframe from S3 and return an analyzer instance."""

        obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=OFFLINE_PATH)
        body = obj["Body"].read()
        offline_df = pd.read_parquet(io.BytesIO(body))
        data_logger.info("Loaded {} offline records from S3.", offline_df.shape[0])
        DataLoadOfflineEvent(
            feature_group="offline_ticket_metrics",
            storage_path=OFFLINE_PATH,
            records_loaded=int(offline_df.shape[0]),
            cache_hit=False,
            metadata={
                "bucket": BUCKET_NAME,
                "columns": list(offline_df.columns),
            },
        ).emit()
        return cls(offline_df)

    def save_metrics_to_s3(
        self,
    ) -> None:
        """Persist offline analysis metrics to S3 in parquet format."""

        if self.analysis_res is None:
            self.analyze()
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=METRICS_PATH,
            Body=self.analysis_res.model_dump_json(),
            ContentType="application/json",
            # Optional hardening:
        )
        data_logger.info(f"Persisted offline analysis metrics to S3 at key {METRICS_PATH}.")
        DataStatsSaveEvent(
            destination_uri=METRICS_PATH,
            metric_names=[
                "res_time_all",
                "sat_score_all",
                "res_time_by_senti",
                "sat_score_by_senti",
            ],
            records_written=1,
            metadata={"bucket": BUCKET_NAME},
        ).emit()

    @staticmethod
    def _build_timedelta_metric(series: pd.Series) -> TimedeltaMetric:
        """Compute percentile statistics for a timedelta series."""

        clean_series = series.dropna()
        if clean_series.empty:
            data_logger.warning(
                "Timedelta metric computation requestedon empty series; returning zeros."
            )
            zero_delta = timedelta()
            return TimedeltaMetric(
                avg=zero_delta,
                p50=zero_delta,
                P75=zero_delta,
                p90=zero_delta,
            )

        if not is_timedelta64_dtype(clean_series):
            msg = "Timedelta series required for timedelta metric computation."
            data_logger.error(msg)
            raise TypeError(msg)

        numeric_series = clean_series.dt.total_seconds()
        percentiles = numeric_series.quantile([0.5, 0.75, 0.9], interpolation="linear")

        return TimedeltaMetric(
            avg=timedelta(seconds=float(numeric_series.mean())),
            p50=timedelta(seconds=float(percentiles.loc[0.5])),
            P75=timedelta(seconds=float(percentiles.loc[0.75])),
            p90=timedelta(seconds=float(percentiles.loc[0.9])),
        )

    @staticmethod
    def _build_number_metric(series: pd.Series) -> NumberMetric:
        """Compute percentile statistics for a numeric series."""

        clean_series = pd.to_numeric(series.dropna(), errors="coerce")
        clean_series = clean_series.dropna()
        if clean_series.empty:
            data_logger.warning(
                "Numeric metric computation requested on empty series; returning zeros."
            )
            return NumberMetric(avg=0.0, p50=0.0, P75=0.0, p90=0.0)

        percentiles = clean_series.quantile([0.5, 0.75, 0.9], interpolation="linear")

        return NumberMetric(
            avg=float(clean_series.mean()),
            p50=float(percentiles.loc[0.5]),
            P75=float(percentiles.loc[0.75]),
            p90=float(percentiles.loc[0.9]),
        )

    def res_time(self) -> TimedeltaMetric:
        """Return response time summary statistics for offline tickets."""

        return self._build_timedelta_metric(self.df[RES_TIME_COLUMN])

    def sat_score(self) -> NumberMetric:
        """Return satisfaction score summary statistics for offline tickets."""

        return self._build_number_metric(self.df[SAT_SCORE_COLUMN])

    def res_time_by_senti(self) -> dict[CustomerSentiment, ResponseTimeStats]:
        """Response time distribution grouped by customer sentiment."""

        grouped = self.df.dropna(subset=[SENTIMENT_COLUMN]).groupby(SENTIMENT_COLUMN)
        metrics: dict[CustomerSentiment, ResponseTimeStats] = {}
        for sentiment_value, segment in grouped:
            try:
                sentiment = CustomerSentiment(sentiment_value)
            except ValueError:
                data_logger.warning(
                    "Skipping response time aggregation for unknown sentiment {}.",
                    sentiment_value,
                )
                continue

            metrics[sentiment] = ResponseTimeStats(
                volume=float(segment.shape[0]),
                res_time=self._build_timedelta_metric(segment[RES_TIME_COLUMN]),
            )

        return metrics

    def sat_score_by_senti(self) -> dict[CustomerSentiment, SatisfactionScoreStats]:
        """Satisfaction score distribution grouped by customer sentiment."""

        grouped = self.df.dropna(subset=[SENTIMENT_COLUMN]).groupby(SENTIMENT_COLUMN)
        metrics: dict[CustomerSentiment, SatisfactionScoreStats] = {}
        for sentiment_value, segment in grouped:
            try:
                sentiment = CustomerSentiment(sentiment_value)
            except ValueError:
                data_logger.warning(
                    "Skipping satisfaction score aggregation for unknown sentiment {}.",
                    sentiment_value,
                )
                continue

            metrics[sentiment] = SatisfactionScoreStats(
                volume=float(segment.shape[0]),
                sat_score=self._build_number_metric(segment[SAT_SCORE_COLUMN]),
            )

        return metrics

    def analyze(self) -> AnalysisRes:
        """Return aggregate metrics across all offline tickets."""

        res_time_all = self.res_time()
        sat_score_all = self.sat_score()
        res_time_by_senti = self.res_time_by_senti()
        sat_score_by_senti = self.sat_score_by_senti()

        DataResTimeStatsEvent(
            percentile_values={
                key: value.total_seconds() for key, value in res_time_all.model_dump().items()
            },
            records_aggregated=int(self.df[RES_TIME_COLUMN].dropna().shape[0]),
            metadata={"sentiment_segments": len(res_time_by_senti)},
        ).emit()

        DataSatScoreStatsEvent(
            average_score=float(sat_score_all.avg),
            records_aggregated=int(self.df[SAT_SCORE_COLUMN].dropna().shape[0]),
            metadata={
                "sentiment_segments": len(sat_score_by_senti),
                "percentiles": sat_score_all.model_dump(),
            },
        ).emit()

        analysis_res = AnalysisRes(
            res_time_all=res_time_all,
            sat_score_all=sat_score_all,
            res_time_by_senti=res_time_by_senti,
            sat_score_by_senti=sat_score_by_senti,
        )
        self.analysis_res = analysis_res
        return analysis_res


offline_analyzer = OfflineMetricsAnalyzer.from_s3()

__all__ = ["OfflineMetricsAnalyzer", "offline_analyzer"]
