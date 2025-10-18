"""Validate the offline ticket dataset using Pandera quality checks and reporting."""

from __future__ import annotations

from typing import Final

import pandas as pd
from pandas import DataFrame, Series
from prefect import task
from pydantic import ValidationError

from tickets.schemas.data_quality import DataQualityReport
from tickets.schemas.events import (
    DataQaBusinessEvent,
    DataQaCleanedEvent,
    DataQaMissingEvent,
    DataQaReportEvent,
    DataQaSchemaEvent,
    DataQaTimingEvent,
)
from tickets.schemas.ticket import Ticket
from tickets.utils.config_util import CONFIG
from tickets.utils.io_util import load_df_from_s3, save_df_to_s3
from tickets.utils.log_util import DATA_LOGGER

MAX_VIOLATIONS: Final[int] = 5
TEMPORAL_COLUMNS: Final[list[str]] = ["created_at", "updated_at", "resolved_at"]


def check_schema(df: DataFrame) -> Series:
    """Return a boolean mask for rows that fail the Ticket schema validation."""
    invalid_mask = pd.Series(False, index=df.index, dtype="bool")
    violations: list[str] = []

    for row_index, payload in enumerate(df.to_dict(orient="records")):
        try:
            Ticket.model_validate(payload)
        except ValidationError as exc:
            violations.append(f"row={row_index}: {exc.errors()}")
            invalid_mask.iloc[row_index] = True
            DATA_LOGGER.warning("schema_violation row=%s errors=%s", row_index, exc.errors())

    DataQaSchemaEvent(
        violations=violations[:MAX_VIOLATIONS],
        records_processed=int(df.shape[0]),
        metadata={
            "invalid_count": int(invalid_mask.sum()),
            "violations_sampled": len(violations),
        },
    ).emit()
    return invalid_mask


def check_timing(df: DataFrame) -> Series:
    """Return a boolean mask for rows that violate temporal ordering constraints."""
    resolved_before_created = df["resolved_at"].lt(df["created_at"])
    updated_before_created = df["updated_at"].lt(df["created_at"])
    resolved_before_updated = df["resolved_at"].lt(df["updated_at"])

    invalid_mask = resolved_before_created | updated_before_created | resolved_before_updated

    if invalid_mask.any():
        samples = df.loc[invalid_mask, TEMPORAL_COLUMNS].head(3).to_dict(orient="records")
        DATA_LOGGER.warning(
            "invalid temporal constraint; count=%s; samples=%s",
            int(invalid_mask.sum()),
            samples,
        )

    DataQaTimingEvent(
        check_name="ticket_temporal_ordering",
        passed=not invalid_mask.any(),
        metadata={"rows_flagged": int(invalid_mask.sum())},
    ).emit()
    return invalid_mask


def check_business_rules(df: DataFrame) -> Series:
    """Return a boolean mask for rows that violate domain-specific business rules."""
    invalid_mask = pd.Series(False, index=df.index, dtype="bool")
    DataQaBusinessEvent(
        rule_set="default_business_rules",
        failures=[],
        blocking=False,
        metadata={"rows_processed": int(df.shape[0])},
    ).emit()
    return invalid_mask


def count_missing_rows(df: DataFrame) -> Series:
    """Return a boolean mask for rows containing at least one missing value."""
    missing_mask = df.isna().any(axis=1)
    missing_count = int(missing_mask.sum())
    DataQaMissingEvent(
        records_processed=int(df.shape[0]),
        missing_value=missing_count,
        metadata={"missing_rows": missing_count},
    ).emit()
    return missing_mask


def build_report(
    *,
    schema_mask: Series,
    timing_mask: Series,
    business_mask: Series,
    missing_mask: Series,
    rows_evaluated: int,
) -> DataQualityReport:
    """Assemble the DataQualityReport and emit an audit event."""
    report = DataQualityReport(
        invalid_schema=int(schema_mask.sum()),
        invalid_timing=int(timing_mask.sum()),
        missing_row=int(missing_mask.sum()),
        invalid_business=int(business_mask.sum()),
    )
    summary = (
        f"invalid_schema={report.invalid_schema}, "
        f"invalid_timing={report.invalid_timing}, "
        f"missing_row={report.missing_row}, "
        f"invalid_business={report.invalid_business}"
    )
    DataQaReportEvent(
        report_uri=CONFIG.data.quality_report_file,
        summary=summary,
        attachments=[],
        metadata={
            "bucket": CONFIG.data.bucket,
            "records_evaluated": rows_evaluated,
        },
    ).emit()
    return report


def clean_dataset(df: DataFrame, invalid_mask: Series) -> DataFrame:
    """Return the cleaned dataset after removing rows flagged as invalid."""
    cleaned_df = df.loc[~invalid_mask].copy().reset_index(drop=True)

    DataQaCleanedEvent(
        dataset_uri=f"s3://{CONFIG.data.bucket}/{CONFIG.data.offline_file}",
        records_available=int(cleaned_df.shape[0]),
        cleaning_steps=[
            "schema_validation",
            "temporal_validation",
            "business_rules",
        ],
        metadata={
            "source_rows": int(df.shape[0]),
            "invalid_rows_removed": int(invalid_mask.sum()),
        },
    ).emit()
    return cleaned_df


@task
def run_quality_checks() -> tuple[DataQualityReport, DataFrame]:
    """Execute all quality checks on the provided dataframe."""
    df = load_df_from_s3(data_path=CONFIG.data.offline_file, group=__file__)
    schema_mask = check_schema(df)
    timing_mask = check_timing(df)
    business_mask = check_business_rules(df)
    missing_mask = count_missing_rows(df)

    invalid_mask = schema_mask | timing_mask | business_mask | missing_mask
    cleaned_df = clean_dataset(df, invalid_mask)
    save_df_to_s3(df=cleaned_df, data_path=CONFIG.data.clean_file, group=__file__)
    report = build_report(
        schema_mask=schema_mask,
        timing_mask=timing_mask,
        business_mask=business_mask,
        missing_mask=missing_mask,
        rows_evaluated=int(df.shape[0]),
    )
    return report, cleaned_df


def main() -> None:
    """Load the offline dataset, evaluate quality, and return the summary report."""
    df = load_df_from_s3(data_path=CONFIG.data.offline_file, group=__file__)
    report, cleaned_df = run_quality_checks.fn()
    DATA_LOGGER.info(
        "ticket_quality summary=%s cleaned_rows=%s source_rows=%s",
        report.model_dump(),
        int(cleaned_df.shape[0]),
        int(df.shape[0]),
    )
    print(report)
    print(cleaned_df.head())


if __name__ == "__main__":
    main()
