"""Validate the offline ticket dataset using Pandera quality checks and reporting."""

from __future__ import annotations

import io
from time import perf_counter
from typing import Final

import pandas as pd
from prefect import task
from pydantic import ValidationError

from tickets.schemas.events import (
    DataLoadOfflineEvent,
    DataQaBusinessEvent,
    DataQaCleanedEvent,
    DataQaMissingEvent,
    DataQaReportEvent,
    DataQaSchemaEvent,
    DataQaTimingEvent,
)

from ..schemas.data_quality import DataQualityReport
from ..schemas.ticket import Ticket
from ..utils.config_util import cfg
from ..utils.io_util import data_logger, s3_client

OFFLINE_PATH: Final[str] = cfg.data.offline_file
BUCKET_NAME: Final[str] = cfg.data.bucket
QUALITY_REPORT_PATH: Final[str] = cfg.data.quality_report_file
MAX_VIOLATIONS = 5


class DataQuality:
    def __init__(self, df: pd.DataFrame | None = None) -> None:
        self._df: pd.DataFrame
        if df is None:
            self._df = self._get_offline_from_s3()
        else:
            self._df = df
        self.invalid_schema_num = 0
        self.invalid_timing_num = 0
        self.missing_value = 0
        self.invalid_indices: pd.Series = pd.Series(False, index=self._df.index)
        self.cleaned_df: pd.DataFrame

    def _get_offline_from_s3(self) -> pd.DataFrame:
        obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=OFFLINE_PATH)
        body = obj["Body"].read()
        offline_df = pd.read_parquet(io.BytesIO(body))
        DataLoadOfflineEvent(
            feature_group="offline_ticket_quality",
            storage_path=OFFLINE_PATH,
            records_loaded=int(offline_df.shape[0]),
            cache_hit=False,
            metadata={
                "bucket": BUCKET_NAME,
                "columns": list(offline_df.columns),
            },
        ).emit()
        return offline_df

    @task
    def check_schema(self) -> pd.Series:
        """Validate every ticket row against the Pydantic Ticket schema."""

        invalid_mask: pd.Series = pd.Series(False, index=self._df.index)
        self.invalid_schema_num = 0
        violations: list[str] = []

        for row_index, payload in enumerate(self._df.to_dict(orient="records")):
            try:
                Ticket.model_validate(payload)
            except ValidationError as exc:
                data_logger.warning(f"row={row_index}: {exc.errors()}")
                invalid_mask.iloc[row_index] = True
                violations.append(f"row={row_index}: {exc.errors()}")
                continue
        self.invalid_schema_num = invalid_mask.sum()
        if invalid_mask.any():
            self.invalid_indices = self.invalid_indices | invalid_mask
        DataQaSchemaEvent(
            violations=violations[:MAX_VIOLATIONS],
            records_processed=int(self._df.shape[0]),
            metadata={
                "invalid_count": self.invalid_schema_num,
                "violations_sampled": len(violations),
            },
        ).emit()
        return invalid_mask.copy()

    @task
    def check_timing(self) -> pd.Series:
        start = perf_counter()
        df = self._df
        invalid_mask: pd.Series = pd.Series(False, index=self._df.index)
        res_b4_cr8_mask = df["resolved_at"].lt(df["created_at"])
        upd8_b4_cr8_mask = df["updated_at"].lt(df["created_at"])
        res_b4_upd8_mask = df["resolved_at"].lt(df["updated_at"])
        invalid_mask = res_b4_cr8_mask | upd8_b4_cr8_mask | res_b4_upd8_mask
        check_passed = not invalid_mask.any()
        self.invalid_indices = invalid_mask | self.invalid_indices
        self.invalid_timing_num = self.invalid_indices.sum()
        msg = "invalid temporal constraint"
        cols = ["created_at", "updated_at", "resolved_at"]
        self.register_issue(invalid_mask, msg, cols)
        duration_ms = int((perf_counter() - start) * 1000)
        DataQaTimingEvent(
            check_name="ticket_temporal_ordering",
            duration_ms=duration_ms,
            passed=check_passed,
            metadata={"rows_flagged": int(invalid_mask.sum())},
        ).emit()
        return invalid_mask.copy()

    @task
    def check_business(self) -> None:
        """This is a place holder"""
        invalid_mask = pd.Series(False, index=self._df.index)
        self.invalid_indices = self.invalid_indices | invalid_mask
        DataQaBusinessEvent(
            rule_set="default_business_rules",
            failures=[],
            blocking=False,
            metadata={"rows_processed": int(self._df.shape[0])},
        ).emit()

    @task
    def count_missing(self) -> int:
        """Return the total number of null values in the offline dataset."""
        missing_counts = self._df.isna().sum()
        self.missing_value = int(missing_counts.sum())

        DataQaMissingEvent(
            records_processed=int(self._df.shape[0]),
            missing_value=self.missing_value,
            metadata={
                "missing_total": self.missing_value,
            },
        ).emit()
        return self.missing_value

    def register_issue(self, mask: pd.Series, message: str, sample_columns: list[str]) -> None:
        """Log a warning with contextual samples and track invalid rows."""
        if not mask.any():
            return
        samples = self._df.loc[mask, sample_columns].head(3).to_dict(orient="records")
        data_logger.warning(f"{message}; count={int(mask.sum())}; samples={samples}")
        self.invalid_indices.update(self._df.index[mask].tolist())

    def gen_report(self) -> DataQualityReport:
        self.check_schema()
        self.check_timing()
        report = DataQualityReport(
            invalid_schema=self.invalid_schema_num,
            invalid_timing=self.invalid_timing_num,
            missing_value=self.missing_value,
        )
        summary = (
            f"invalid_schema={report.invalid_schema}, invalid_timing={report.invalid_timing}, "
            f"missing_value={report.missing_value}"
        )
        DataQaReportEvent(
            report_uri=QUALITY_REPORT_PATH,
            summary=summary,
            attachments=[],
            metadata={
                "bucket": BUCKET_NAME,
                "records_evaluated": int(self._df.shape[0]),
            },
        ).emit()
        return report

    def clean(self) -> pd.DataFrame:
        self.check_schema()
        self.check_timing()
        cleaned_df = self._df.loc[~self.invalid_indices]
        self.cleaned_df = cleaned_df.copy().reset_index(drop=True)
        DataQaCleanedEvent(
            dataset_uri=f"s3://{BUCKET_NAME}/{OFFLINE_PATH}",
            records_available=int(self.cleaned_df.shape[0]),
            cleaning_steps=[
                "schema_validation",
                "temporal_validation",
                "business_rules",
            ],
            metadata={
                "source_rows": int(self._df.shape[0]),
                "invalid_rows_removed": int(self.invalid_indices.sum()),
            },
        ).emit()
        return self.cleaned_df.copy()
