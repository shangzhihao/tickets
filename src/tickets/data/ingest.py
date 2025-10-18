"""Ingest raw ticket data and persist it as parquet."""

from __future__ import annotations

import pandas as pd
import redis
from prefect import flow, task

from tickets.schemas.events import DataSaveToRedisEvent
from tickets.utils.config_util import CONFIG
from tickets.utils.io_util import (
    load_df_from_s3,
    redis_pool,
    save_df_to_s3,
)
from tickets.utils.log_util import DATA_LOGGER


@flow
def ingest() -> None:
    raw = load_df_from_s3(data_path=CONFIG.data.raw_file, group=__file__)

    bronze = make_bronze(raw)
    offline = make_offline(bronze)
    online = make_online(offline)
    print(len(online))


def raw_to_bronze(raw_data: pd.DataFrame) -> pd.DataFrame:
    bronze_data = raw_data.copy()
    return bronze_data


@task
def make_bronze(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Read raw JSON tickets and store them in S3 as a bronze parquet file."""
    if raw is None or raw.empty:
        _raw = load_df_from_s3(data_path=CONFIG.data.raw_file, group=__file__)
    else:
        _raw = raw

    # Persist the bronze snapshot back to S3 as parquet for downstream steps.
    bronze_data = raw_to_bronze(_raw)
    save_df_to_s3(df=bronze_data, data_path=CONFIG.data.bronze_file, group=__file__)
    DATA_LOGGER.info(
        f"{len(bronze_data)} bronze records are saved.",
    )
    return bronze_data


def bronze_to_offline(bronze_data: pd.DataFrame) -> pd.DataFrame:
    return bronze_data.copy()


@task
def make_offline(bronze_data: pd.DataFrame | None) -> pd.DataFrame:
    """Persist a cleaned bronze dataset into the offline store."""
    if bronze_data is None or bronze_data.empty:
        _bronze_data = load_df_from_s3(data_path=CONFIG.data.bronze_file, group=__file__)
    else:
        _bronze_data = bronze_data.copy()

    offline_data = bronze_to_offline(_bronze_data)

    # Write the cleaned dataset to the offline S3 location.
    save_df_to_s3(df=offline_data, data_path=CONFIG.data.offline_file, group=__file__)
    DATA_LOGGER.info(
        f"{len(offline_data)} offline records are saved.",
    )

    return offline_data


def off_to_online(offline_data: pd.DataFrame) -> pd.DataFrame:
    """Convert an offline dataset into an online dataset suitable for Redis storage."""
    _offline_data = offline_data.copy()
    if "created_at" not in _offline_data.columns:
        raise KeyError("'created_at' column is required to order the online dataset.")

    num = getattr(CONFIG.data, "num_online", None)
    if not isinstance(num, int) or num <= 0:
        raise ValueError("'data.num_online' must be a positive integer")

    _offline_data["created_at"] = pd.to_datetime(
        _offline_data["created_at"], utc=True, errors="coerce"
    )
    _offline_data = _offline_data.sort_values("created_at", ascending=False, kind="mergesort")
    _offline_data = _offline_data.dropna(subset=["created_at"])
    online_data = _offline_data.head(num).reset_index(drop=True)
    return online_data


@task
def make_online(offline_data: pd.DataFrame | None) -> pd.DataFrame:
    """Write an ordered, truncated dataset suitable for online serving."""
    if offline_data is None or offline_data.empty:
        _offline_data = load_df_from_s3(data_path=CONFIG.data.offline_file, group=__file__)
    else:
        _offline_data = offline_data.copy()
    online_data = off_to_online(_offline_data)
    save_df_to_s3(df=online_data, data_path=CONFIG.data.online_file, group=__file__)
    DATA_LOGGER.info(
        f"{len(online_data)} online records are saved.",
    )
    dt_cols = ["created_at", "updated_at", "resolved_at"]
    online_data[dt_cols] = online_data[dt_cols].apply(
        lambda x: x.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    )

    r = redis.Redis(connection_pool=redis_pool)
    pipe = r.pipeline()
    path = "$"
    records = online_data.to_dict(orient="records")
    for ticket in records:
        pipe.json().set(
            name=ticket["ticket_id"],
            path=path,
            obj=ticket,
        )
    results = pipe.execute()
    written_count = sum(results)
    failed_count = len(results) - written_count
    DATA_LOGGER.info("Records written to redis: {}", written_count)
    if failed_count:
        DATA_LOGGER.warning("Records failed to write to redis: {}", failed_count)
    entity_keys = [
        str(ticket["ticket_id"])
        for ticket in records
        if isinstance(ticket.get("ticket_id"), str | int)
    ]
    DATA_LOGGER.info(
        f"{written_count} online records are saved.",
    )

    DataSaveToRedisEvent(
        feature_group=__file__,
        key="ticket_id",
        value=",".join(entity_keys),
        records_written=len(records),
    ).emit(level="WARNING" if failed_count else "INFO")
    return online_data


if __name__ == "__main__":
    ingest()
