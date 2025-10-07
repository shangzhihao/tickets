from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, NonNegativeFloat

from ..utils.io_util import event_logger


class EventType(StrEnum):
    DATA_LOAD_RAW = "Data Load (Raw)"

    DATA_LOAD_BRONZE = "Data Load (Bronze)"
    DATA_SAVE_BRONZE = "Data Save (Bronze)"

    DATA_SAVE_OFFLINE = "Data Save (Offline Store)"
    DATA_LOAD_OFFLINE = "Data Load (Offline Store)"

    DATA_LOAD_ONLINE = "Data Load (Online Store)"
    DATA_SAVE_ONLNE = "Data Save (Online Store)"

    DATA_RES_TIME_STATS = "Data Metrics (Resolution Time)"
    DATA_SAT_SCORE_STATS = "Data Metrics (Satisfaction Score)"

    DATA_STATS_SAVE = "Data Metrics Persisted"
    DATA_STATA_READ = "Data Metrics Read"

    DATA_QA_TIMING = "Data QA (Timing)"
    DATA_QA_SCHEMA = "Data QA (Schema Validation)"
    DATA_QA_MISSING = "Data QA (Missing Values)"
    DATA_QA_BUSINESS = "Data QA (Business Rules)"

    DATA_QA_CLEANED = "Data QA (Cleaned Dataset)"
    DATA_QA_REPORT = "Data QA (Report Generated)"


class BaseEvent(BaseModel):
    """Base event payload shared across all event types."""

    event_type: EventType
    correlation_id: str | None = Field(
        default=None,
        description="Correlation identifier used to trace a ticket through the pipeline.",
    )
    time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    labels: dict[str, str] = Field(default_factory=dict,
        description="Optional tags for downstream filtering.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form metadata that does not belong in dedicated fields.",
    )

    def emit(self, *, level: str = "INFO") -> dict[str, Any]:
        """Emit the event to the configured Loguru sinks and return the payload."""
        payload = self.model_dump()
        event_logger.log(level, payload)
        return payload




class DataLoadRawEvent(BaseEvent):
    """Event emitted when raw ticket data is ingested from the landing zone."""

    event_type: EventType = EventType.DATA_LOAD_RAW
    source_uri: str = Field(..., 
        description="URI pointing to the raw JSON payload.")
    records_loaded: int = Field(..., ge=0, 
        description="Number of ticket records ingested.")
    duration_ms: int = Field(..., ge=0, 
        description="Latency of the ingestion step in milliseconds.")
    warnings: list[str] = Field(default_factory=list, 
        description="Any non-blocking issues observed during ingestion.")


class DataLoadBronzeEvent(BaseEvent):
    """Event emitted when curated bronze-layer data is read from storage."""

    event_type: EventType = EventType.DATA_LOAD_BRONZE
    table_uri: str = Field(..., 
        description="Location of the bronze dataset.")
    records_loaded: int = Field(..., ge=0, 
        description="Number of rows read from the bronze table.")


class DataSaveBronzeEvent(BaseEvent):
    """Event emitted after the bronze dataset has been materialized."""

    event_type: EventType = EventType.DATA_SAVE_BRONZE
    table_uri: str = Field(..., 
        description="Destination path of the bronze dataset.")
    records_written: int = Field(..., ge=0, 
        description="Number of rows persisted to the bronze table.")
    duration_ms: int = Field(..., ge=0, 
        description="Latency of the persistence step in milliseconds.")


class DataSaveOfflineEvent(BaseEvent):
    """Event emitted when offline feature data is persisted to the feature store."""

    event_type: EventType = EventType.DATA_SAVE_OFFLINE
    feature_group: str = Field(..., 
        description="Identifier of the feature group being materialized.")
    storage_path: str = Field(..., 
        description="Offline store path, typically a Parquet location.")
    records_written: int = Field(..., ge=0, 
        description="Number of feature rows written.")
    feature_names: list[str] = Field(..., 
        description="List of features included in this batch.")


class DataLoadOfflineEvent(BaseEvent):
    """Event emitted when offline feature data is loaded for training or validation."""

    event_type: EventType = EventType.DATA_LOAD_OFFLINE
    feature_group: str = Field(..., 
        description="Identifier of the feature group being loaded.")
    storage_path: str = Field(..., 
        description="Offline store path that was accessed.")
    records_loaded: int = Field(..., ge=0, 
        description="Number of feature rows retrieved.")
    cache_hit: bool = Field(..., 
        description="Indicates whether the data was served from cache.")


class DataLoadOnlineEvent(BaseEvent):
    """Event emitted when online features are retrieved for inference."""

    event_type: EventType = EventType.DATA_LOAD_ONLINE
    feature_group: str = Field(..., 
        description="Online feature group identifier.")
    entity_keys: list[str] = Field(..., 
        description="Entities requested from the online store.")
    features_returned: dict[str, Any] = Field(..., 
        description="Features returned per entity.")
    latency_ms: int = Field(..., ge=0, 
        description="Latency of the online read in milliseconds.")


class DataSaveOnlineEvent(BaseEvent):
    """Event emitted when features are written to the online store."""

    event_type: EventType = EventType.DATA_SAVE_ONLNE
    feature_group: str = Field(..., 
        description="Online feature group identifier.")
    entity_keys: list[str] = Field(...,
        description="Entities updated in the online store.")
    records_written: int = Field(..., ge=0, 
        description="Number of upserts performed.")
    ttl_seconds: int | None = Field(default=None, 
        description="Time-to-live applied to the online entries.")


class DataResTimeStatsEvent(BaseEvent):
    """Event emitted when resolution-time statistics are computed."""

    event_type: EventType = EventType.DATA_RES_TIME_STATS
    percentile_values: dict[str, float] = Field(
        ...,
        description="Calculated percentile metrics such as p50, p90, p95.",
    )
    records_aggregated: int = Field(..., ge=0,
        description="Number of tickets included in the aggregation.")


class DataSatScoreStatsEvent(BaseEvent):
    """Event emitted when customer satisfaction score statistics are computed."""

    event_type: EventType = EventType.DATA_SAT_SCORE_STATS
    average_score: float = Field(..., 
        description="Average satisfaction score in the window.")
    records_aggregated: int = Field(..., ge=0,
        description="Number of tickets included in the aggregation.")


class DataStatsSaveEvent(BaseEvent):
    """Event emitted when computed statistics are written to persistent storage."""

    event_type: EventType = EventType.DATA_STATS_SAVE
    destination_uri: str = Field(..., 
        description="Storage location where metrics were persisted.")
    metric_names: list[str] = Field(..., 
        description="Names of metrics that were saved.")
    records_written: int = Field(..., ge=0, 
        description="Number of metric rows written.")


class DataStataReadEvent(BaseEvent):
    """Event emitted when statistics are read from persistent storage."""

    event_type: EventType = EventType.DATA_STATA_READ
    source_uri: str = Field(..., 
        description="Storage location from which metrics were read.")
    metric_names: list[str] = Field(..., 
        description="Names of metrics that were retrieved.")
    records_loaded: int = Field(..., ge=0, 
        description="Number of metric rows retrieved.")


class DataQaTimingEvent(BaseEvent):
    """Event emitted when timing-based QA checks complete."""

    event_type: EventType = EventType.DATA_QA_TIMING
    check_name: str = Field(..., 
        description="Identifier of the timing check that ran.")
    duration_ms: int = Field(..., ge=0, 
        description="Measured duration in milliseconds.")
    passed: bool = Field(..., 
        description="Indicates if the timing check passed.")


class DataQaSchemaEvent(BaseEvent):
    """Event emitted when schema validation results are available."""

    event_type: EventType = EventType.DATA_QA_SCHEMA
    violations: list[str] = Field(...,
        description="List of schema violations detected.")
    records_processed: int = Field(..., ge=0,
        description="Number of records evaluated during the check.")


class DataQaMissingEvent(BaseEvent):
    """Event emitted when missing-value analysis completes."""

    event_type: EventType = EventType.DATA_QA_MISSING
    missing_value: NonNegativeFloat
    records_processed: int = Field(..., ge=0,
        description="Number of records evaluated during the check.")


class DataQaBusinessEvent(BaseEvent):
    """Event emitted when business-rule validation completes."""

    event_type: EventType = EventType.DATA_QA_BUSINESS
    rule_set: str = Field(...,
        description="Name of the business rule set that was evaluated.")
    failures: list[str] = Field(...,
        description="List of failed business rules.")
    blocking: bool = Field(...,
        description="Inicates whether failures should block downstream processing.")


class DataQaCleanedEvent(BaseEvent):
    """Event emitted when the cleaned dataset is available for downstream tasks."""

    event_type: EventType = EventType.DATA_QA_CLEANED
    dataset_uri: str = Field(..., 
        description="Location of the cleaned dataset.")
    records_available: int = Field(..., ge=0, 
        description="Number of records in the cleaned dataset.")
    cleaning_steps: list[str] = Field(..., 
        description="Ordered list of transformations applied.")


class DataQaReportEvent(BaseEvent):
    """Event emitted when QA reporting artifacts are generated."""

    event_type: EventType = EventType.DATA_QA_REPORT
    report_uri: str = Field(..., 
        description="Location of the QA report artifact.")
    summary: str = Field(..., 
        description="Human-readable summary of the QA findings.")
    attachments: list[str] = Field(
        default_factory=list,
        description="List of supplemental artifacts such as charts or notebooks.",
    )
