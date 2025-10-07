from __future__ import annotations

import importlib
import io
import sys
from collections.abc import Generator
from datetime import timedelta
from types import ModuleType
from unittest.mock import Mock

import pandas as pd
import pytest

from tickets.schemas.ticket import CustomerSentiment


@pytest.fixture()
def analyze_mod(monkeypatch: pytest.MonkeyPatch) -> Generator[ModuleType, None, None]:
    """Import tickets.data.analyze with an in-memory S3 stub."""
    seed_frame = pd.DataFrame(
        {
            "created_at": pd.to_datetime(["2024-01-01 00:00:00"]),
            "resolved_at": pd.to_datetime(["2024-01-01 01:00:00"]),
            "satisfaction_score": [5],
            "customer_sentiment": [CustomerSentiment.NEUTRAL.value],
        }
    )
    seed_buffer = io.BytesIO()
    seed_frame.to_parquet(seed_buffer, index=False)
    seed_buffer.seek(0)

    fake_s3 = Mock()
    fake_s3.get_object.side_effect = (
        lambda *_, **__: {"Body": io.BytesIO(seed_buffer.getvalue())}
    )
    fake_s3.put_object.return_value = None
    monkeypatch.setattr("boto3.client", lambda *_, **__: fake_s3)

    module = importlib.import_module("tickets.data.analyze")
    try:
        yield module
    finally:
        sys.modules.pop("tickets.data.analyze", None)


def test_offline_metrics_analyzer_missing_columns(analyze_mod: ModuleType) -> None:
    """Ensure the constructor rejects frames lacking mandatory columns."""
    invalid_df = pd.DataFrame(
        {
            "created_at": pd.to_datetime(["2024-01-01 00:00:00"]),
            "satisfaction_score": [4],
            "customer_sentiment": [CustomerSentiment.NEUTRAL.value],
        }
    )

    with pytest.raises(KeyError) as excinfo:
        analyze_mod.OfflineMetricsAnalyzer(invalid_df)

    assert "missing required" in str(excinfo.value)


def test_res_time_metric_handles_empty_series(analyze_mod: ModuleType) -> None:
    """Zero-duration metrics should be returned when response times are unavailable."""
    empty_df = pd.DataFrame(
        {
            "created_at": pd.to_datetime(["2024-01-01 00:00:00"]),
            "resolved_at": [pd.NaT],
            "satisfaction_score": [pd.NA],
            "customer_sentiment": [CustomerSentiment.NEUTRAL.value],
        }
    )

    analyzer = analyze_mod.OfflineMetricsAnalyzer(empty_df)
    metric = analyzer.res_time()
    zero_delta = timedelta()

    assert metric.avg == zero_delta
    assert metric.p50 == zero_delta
    assert metric.P75 == zero_delta
    assert metric.p90 == zero_delta


def test_analyze_aggregates_metrics_by_sentiment(analyze_mod: ModuleType) -> None:
    """Verify aggregate statistics and sentiment splits are computed correctly."""
    input_df = pd.DataFrame(
        {
            "created_at": pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 02:00:00",
                    "2024-01-01 05:00:00",
                    "2024-01-02 00:00:00",
                ]
            ),
            "resolved_at": pd.to_datetime(
                [
                    "2024-01-01 01:00:00",
                    "2024-01-01 04:00:00",
                    "2024-01-01 08:00:00",
                    "2024-01-02 03:00:00",
                ]
            ),
            "satisfaction_score": [3.0, 4.0, 5.0, 2.0],
            "customer_sentiment": [
                CustomerSentiment.ANGRY.value,
                CustomerSentiment.NEUTRAL.value,
                CustomerSentiment.NEUTRAL.value,
                "excited",
            ],
        }
    )

    analyzer = analyze_mod.OfflineMetricsAnalyzer(input_df)
    result = analyzer.analyze()

    assert result.res_time_all.avg.total_seconds() == pytest.approx(8100.0)
    assert result.res_time_all.p50.total_seconds() == pytest.approx(9000.0)
    assert result.res_time_all.P75.total_seconds() == pytest.approx(10800.0)
    assert result.sat_score_all.avg == pytest.approx(3.5)
    assert result.sat_score_all.p90 == pytest.approx(4.7)

    angry_stats = result.res_time_by_senti[CustomerSentiment.ANGRY]
    assert angry_stats.volume == pytest.approx(1.0)
    assert angry_stats.res_time.avg.total_seconds() == pytest.approx(3600.0)

    neutral_stats = result.res_time_by_senti[CustomerSentiment.NEUTRAL]
    assert neutral_stats.volume == pytest.approx(2.0)
    assert neutral_stats.res_time.avg.total_seconds() == pytest.approx(9000.0)
    assert neutral_stats.res_time.P75.total_seconds() == pytest.approx(9900.0)

    neutral_scores = result.sat_score_by_senti[CustomerSentiment.NEUTRAL].sat_score
    assert neutral_scores.avg == pytest.approx(4.5)
    assert neutral_scores.P75 == pytest.approx(4.75)

    assert len(result.res_time_by_senti) == 2
    assert len(result.sat_score_by_senti) == 2
