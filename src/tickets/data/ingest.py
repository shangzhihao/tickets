"""Ingest raw ticket data and persist it as parquet."""

from __future__ import annotations

import io
from time import perf_counter_ns

import pandas as pd
import redis
from prefect import task

from ..schemas.events import (
    DataLoadBronzeEvent,
    DataLoadOfflineEvent,
    DataLoadRawEvent,
    DataSaveBronzeEvent,
    DataSaveOfflineEvent,
    DataSaveOnlineEvent,
)
from ..utils.config_util import cfg
from ..utils.io_util import data_logger, redis_pool, s3_client


def _duration_ms(start_ns: int) -> int:
    """Convert a perf counter delta into milliseconds."""

    elapsed = perf_counter_ns() - start_ns
    return max(int(elapsed / 1_000_000), 0)


def _s3_uri(key: str) -> str:
    """Return a fully qualified S3-style URI for the configured bucket."""

    return f"s3://{cfg.data.bucket}/{key}"


def ingest() -> None:
    """Materialize bronze, offline, and online datasets in a single run."""

    df = bronze()
    df = offline(df)
    df = online(df)


@task
def bronze() -> pd.DataFrame:
    """Read raw JSON tickets and store them in S3 as a bronze parquet file."""

    raw_path = cfg.data.raw_file
    bronze_path = cfg.data.bronze_file
    # Read the raw JSON payload and hydrate a dataframe.
    read_start = perf_counter_ns()
    obj = s3_client.get_object(Bucket=cfg.data.bucket, Key=raw_path)
    body = obj["Body"].read()
    raw_df = pd.read_json(io.BytesIO(body), lines=False)
    DataLoadRawEvent(
        source_uri=_s3_uri(raw_path),
        records_loaded=len(raw_df),
        duration_ms=_duration_ms(read_start),
    ).emit()
    data_logger.info(f"{len(raw_df)} raw records read from s3 json")
    # Persist the bronze snapshot back to S3 as parquet for downstream steps.
    write_start = perf_counter_ns()
    buf = io.BytesIO()
    raw_df.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
    buf.seek(0)
    s3_client.upload_fileobj(
        buf,
        cfg.data.bucket,
        bronze_path,
        ExtraArgs={"ContentType": "application/x-parquet"},
    )
    DataSaveBronzeEvent(
        table_uri=_s3_uri(bronze_path),
        records_written=len(raw_df),
        duration_ms=_duration_ms(write_start),
    ).emit()
    data_logger.info(f"{len(raw_df)} bronze records wrote to s3 parquet")

    return raw_df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


@task
def offline(df: pd.DataFrame | None) -> pd.DataFrame:
    """Persist a cleaned bronze dataset into the offline store."""
    bronze_path = cfg.data.bronze_file
    offline_path = cfg.data.offline_file
    if df is None:
        # Reload the bronze parquet snapshot if the upstream task was skipped.
        obj = s3_client.get_object(Bucket=cfg.data.bucket, Key=bronze_path)
        body = obj["Body"].read()
        bronze_df = pd.read_parquet(io.BytesIO(body))
        data_logger.info(f"{len(bronze_df)} bronze records read from s3")
    else:
        bronze_df = df.copy()
    DataLoadBronzeEvent(
        table_uri=_s3_uri(bronze_path),
        records_loaded=len(bronze_df),
    ).emit()
    offline_df = clean(bronze_df)
    # Write the cleaned dataset to the offline S3 location.
    write_start = perf_counter_ns()
    buf = io.BytesIO()
    offline_df.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
    buf.seek(0)
    s3_client.upload_fileobj(
        buf,
        cfg.data.bucket,
        offline_path,
        ExtraArgs={"ContentType": "application/x-parquet"},
    )
    DataSaveOfflineEvent(
        feature_group=f"{cfg.project_name}_offline",
        storage_path=_s3_uri(offline_path),
        records_written=len(offline_df),
        feature_names=list(offline_df.columns),
        metadata={"write_duration_ms": _duration_ms(write_start)},
    ).emit()
    data_logger.info(f"{len(offline_df)} offline records wrote to s3")

    return offline_df


def make_online(df: pd.DataFrame) -> pd.DataFrame:
    """Return the most recent tickets, ordered by creation timestamp."""

    if "created_at" not in df.columns:
        raise KeyError("'created_at' column is required to order the online dataset.")

    num = getattr(cfg.data, "num_online", None)
    if not isinstance(num, int) or num <= 0:
        raise ValueError("'data.num_online' must be a positive integer")

    frame = df.copy()
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
    frame = frame.sort_values("created_at", ascending=False, kind="mergesort")
    frame = frame.dropna(subset=["created_at"])
    return frame.head(num).reset_index(drop=True)


@task
def online(df: pd.DataFrame | None) -> pd.DataFrame:
    """Write an ordered, truncated dataset suitable for online serving."""
    offline_path = cfg.data.offline_file

    if df is None:
        read_start = perf_counter_ns()
        obj = s3_client.get_object(Bucket=cfg.data.bucket, Key=offline_path)
        body = obj["Body"].read()
        offline_df = pd.read_parquet(io.BytesIO(body))
        data_logger.info(f"{len(offline_df)} offline records read to s3")
        DataLoadOfflineEvent(
            feature_group=f"{cfg.project_name}_offline",
            storage_path=_s3_uri(offline_path),
            records_loaded=len(offline_df),
            cache_hit=False,
            metadata={
                "read_duration_ms": _duration_ms(read_start),
            },
        ).emit()
    else:
        offline_df = df.copy()
    online_df = make_online(offline_df)
    cols = ["created_at", "updated_at", "resolved_at"]
    online_df[cols] = online_df[cols].apply(
        lambda x: x.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    )

    r = redis.Redis(connection_pool=redis_pool)
    pipe = r.pipeline()
    path = "$"
    records = online_df.to_dict(orient="records")
    for ticket in records:
        pipe.json().set(
            name=ticket["ticket_id"], path=path, obj=ticket
        )  # pyright: ignore[reportArgumentType]
    results = pipe.execute()
    written_count = sum(results)
    failed_count = len(results) - written_count
    data_logger.info("Records written to redis: {}", written_count)
    if failed_count:
        data_logger.warning("Records failed to write to redis: {}", failed_count)
    entity_keys = [
        str(ticket["ticket_id"])
        for ticket in records
        if isinstance(ticket.get("ticket_id"), (str, int))
    ]
    DataSaveOnlineEvent(
        feature_group=f"{cfg.project_name}_online",
        entity_keys=entity_keys,
        records_written=written_count,
        ttl_seconds=None,
        metadata={
            "redis_path": path,
            "failed_count": failed_count,
        },
    ).emit(level="WARNING" if failed_count else "INFO")
    return online_df
