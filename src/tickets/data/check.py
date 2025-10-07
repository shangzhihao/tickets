"""Validate the offline ticket dataset using Pandera quality checks and reporting."""

from __future__ import annotations

import io
from typing import Final

import pandas as pd
from prefect import task
from pydantic import ValidationError

from ..schemas.data_quality import DataQualityReport
from ..schemas.ticket import Ticket
from ..utils.config_util import cfg
from ..utils.io_util import data_logger, s3_client

OFFLINE_PATH: Final[str] = cfg.data.offline_file
BUCKET_NAME: Final[str] = cfg.data.bucket
QUALITY_REPORT_PATH: Final[str] = cfg.data.quality_report_file


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
        return pd.read_parquet(io.BytesIO(body))

    @task
    def check_schema(self)->None:
        """Validate every ticket row against the Pydantic Ticket schema."""

        invalid_indices: list[int] = []
        self.invalid_schema_num = 0

        for row_index, payload in enumerate(self._df.to_dict(orient="records")):
            try:
                Ticket.model_validate(payload)
            except ValidationError as exc:
                data_logger.warning(f"row={row_index}: {exc.errors()}")
                invalid_indices.append(row_index)
                continue
        self.invalid_schema_num = len(invalid_indices)
        if invalid_indices:
            self.invalid_indices.loc[invalid_indices] = True

    @task
    def check_timing(self):
        df = self._df
        invalid_mask: pd.Series = pd.Series(False, index=self._df.index)
        res_b4_cr8_mask = df["resolved_at"].lt(df["created_at"])
        upd8_b4_cr8_mask = df["updated_at"].lt(df["created_at"])
        res_b4_upd8_mask = df["resolved_at"].lt(df["updated_at"])
        invalid_mask = res_b4_cr8_mask | upd8_b4_cr8_mask | res_b4_upd8_mask
        self.invalid_indices = invalid_mask | self.invalid_indices
        self.invalid_timing_num = self.invalid_indices.sum()
        msg = "invalid temporal constraint"
        cols = ["created_at", "updated_at", "resolved_at"]
        self.register_issue(invalid_mask, msg, cols)

    @task
    def check_business(self):
        """This is a place holder"""
        invalid_mask = pd.Series(False, index=self._df.index)
        self.invalid_indices = self.invalid_indices | invalid_mask
    @task
    def count_missing(self) -> int:
        """Return the total number of null values in the offline dataset."""
        self.missing_value = int(self._df.isna().sum().sum())
        return self.missing_value

    def register_issue(
        self, mask: pd.Series, message: str, sample_columns: list[str]
    ) -> None:
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
            missing_value=self.missing_value
        )
        return report

    def clean(self):
        self.check_schema()
        self.check_timing()
        cleaned_df = self._df.loc[~self.invalid_indices]
        self.cleaned_df = cleaned_df.copy().reset_index(drop=True)
        return self.cleaned_df.copy()
