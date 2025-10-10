"""Ingest raw ticket data and persist it as parquet."""

from __future__ import annotations

import io
from time import perf_counter_ns

import pandas as pd
import redis
from prefect import flow, task

from ..schemas.events import (
    DataLoadBronzeEvent,
    DataLoadOfflineEvent,
    DataLoadRawEvent,
    DataSaveBronzeEvent,
    DataSaveOfflineEvent,
    DataSaveOnlineEvent,
)
from ..utils.config_util import cfg
from ..utils.io_util import (
    data_logger,
    read_df_from_s3,
    redis_pool,
    s3_client,
    s3_uri,
    save_df_to_s3,
)


def _duration_ms(start_ns: int) -> int:
    """Convert a perf counter delta into milliseconds."""

    elapsed = perf_counter_ns() - start_ns
    return max(int(elapsed / 1_000_000), 0)


@flow
def ingest() -> None:
    ingestion = DataIngestion()
    ingestion.make_bronze()
    ingestion.make_offline()
    ingestion.make_online()


class DataIngestion:
    def __init__(self) -> None:
        self.raw_data: pd.DataFrame = pd.DataFrame()
        self.bronze_data: pd.DataFrame = pd.DataFrame()
        self.offline_data: pd.DataFrame = pd.DataFrame()
        self.online_data: pd.DataFrame = pd.DataFrame()
        self.raw_data_path = cfg.data.raw_file
        self.bronze_data_path = cfg.data.bronze_file
        self.offline_data_path = cfg.data.offline_file

    @task
    def make_bronze(self) -> pd.DataFrame:
        """Read raw JSON tickets and store them in S3 as a bronze parquet file."""
        read_start = perf_counter_ns()
        if self.raw_data.empty:
            self.raw_data = read_df_from_s3(self.raw_data_path)
        DataLoadRawEvent(
            source_uri=s3_uri(self.raw_data_path),
            records_loaded=len(self.raw_data),
            duration_ms=_duration_ms(read_start),
        ).emit()
        data_logger.info(f"{len(self.raw_data)} raw records read from s3 json")

        # Persist the bronze snapshot back to S3 as parquet for downstream steps.
        self.bronze_data = self.raw_to_bronze()
        write_start = perf_counter_ns()
        buf = io.BytesIO()
        self.bronze_data.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
        buf.seek(0)
        s3_client.upload_fileobj(
            buf,
            cfg.data.bucket,
            self.bronze_data_path,
            ExtraArgs={"ContentType": "application/x-parquet"},
        )
        DataSaveBronzeEvent(
            table_uri=s3_uri(self.bronze_data_path),
            records_written=len(self.bronze_data),
            duration_ms=_duration_ms(write_start),
        ).emit()
        data_logger.info(f"{len(self.bronze_data)} bronze records wrote to s3 parquet")

        return self.bronze_data

    def raw_to_bronze(self) -> pd.DataFrame:
        self.bronze_data = self.raw_data.copy()
        return self.bronze_data

    def bronze_to_offline(self) -> pd.DataFrame:
        return self.bronze_data.copy()

    @task
    def make_offline(self) -> pd.DataFrame:
        """Persist a cleaned bronze dataset into the offline store."""
        if self.bronze_data.empty:
            self.bronze_data = read_df_from_s3(self.bronze_data_path)
            data_logger.info(f"{len(self.bronze_data)} bronze records read from s3")

            DataLoadBronzeEvent(
                table_uri=s3_uri(self.bronze_data_path),
                records_loaded=len(self.bronze_data),
            ).emit()
        self.offline_data = self.bronze_to_offline()
        # Write the cleaned dataset to the offline S3 location.
        write_start = perf_counter_ns()
        save_df_to_s3(self.offline_data, self.offline_data_path)

        DataSaveOfflineEvent(
            feature_group=f"{cfg.project_name}_offline",
            storage_path=s3_uri(self.offline_data_path),
            records_written=len(self.offline_data),
            feature_names=list(self.offline_data.columns),
            metadata={"write_duration_ms": _duration_ms(write_start)},
        ).emit()
        data_logger.info(f"{len(self.offline_data)} offline records wrote to s3")

        return self.offline_data

    def off_to_online(self) -> pd.DataFrame:
        if "created_at" not in self.offline_data.columns:
            raise KeyError("'created_at' column is required to order the online dataset.")

        num = getattr(cfg.data, "num_online", None)
        if not isinstance(num, int) or num <= 0:
            raise ValueError("'data.num_online' must be a positive integer")

        frame = self.offline_data.copy()
        frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
        frame = frame.sort_values("created_at", ascending=False, kind="mergesort")
        frame = frame.dropna(subset=["created_at"])
        return frame.head(num).reset_index(drop=True)

    @task
    def make_online(self) -> pd.DataFrame:
        """Write an ordered, truncated dataset suitable for online serving."""
        if self.offline_data.empty:
            read_start = perf_counter_ns()
            self.offline_data = read_df_from_s3(self.offline_data_path)
            data_logger.info(f"{len(self.offline_data)} offline records read to s3")
            DataLoadOfflineEvent(
                feature_group=f"{cfg.project_name}_offline",
                storage_path=s3_uri(self.offline_data_path),
                records_loaded=len(self.offline_data),
                cache_hit=False,
                metadata={
                    "read_duration_ms": _duration_ms(read_start),
                },
            ).emit()

        self.online_data = self.off_to_online()
        cols = ["created_at", "updated_at", "resolved_at"]
        self.online_data[cols] = self.online_data[cols].apply(
            lambda x: x.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        )

        r = redis.Redis(connection_pool=redis_pool)
        pipe = r.pipeline()
        path = "$"
        records = self.online_data.to_dict(orient="records")
        for ticket in records:
            pipe.json().set(
                name=ticket["ticket_id"],
                path=path,
                obj=ticket,
            )
        results = pipe.execute()
        written_count = sum(results)
        failed_count = len(results) - written_count
        data_logger.info("Records written to redis: {}", written_count)
        if failed_count:
            data_logger.warning("Records failed to write to redis: {}", failed_count)
        entity_keys = [
            str(ticket["ticket_id"])
            for ticket in records
            if isinstance(ticket.get("ticket_id"), str | int)
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
        return self.online_data
