"""Ingest raw ticket data and persist it as parquet."""

from __future__ import annotations

import io
from functools import lru_cache

import boto3
import duckdb
import pandas as pd
from botocore.client import Config
from omegaconf import DictConfig
from prefect import flow, task

from ..utils.logger import logger

@lru_cache(maxsize=1)
def get_s3_client(cfg: DictConfig):
    """Return a cached S3 client configured for the tickets lakehouse."""

    s3 = boto3.client(
        "s3",
        endpoint_url=cfg.data.endpoint,
        aws_access_key_id=cfg.data.access_key,
        aws_secret_access_key=cfg.data.secret_key,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        region_name="us-east-1",
    )
    return s3

@flow
def ingest(cfg: DictConfig):
    """Materialize bronze, offline, and online datasets in a single run."""

    df = bronze(cfg)
    df = offline(cfg, df)
    df = online(cfg, df)

@task
def bronze(cfg: DictConfig) -> pd.DataFrame:
    """Read raw JSON tickets and store them in S3 as a bronze parquet file."""

    raw_path = cfg.data.raw_file
    bronze_path = cfg.data.bronze_file
    s3 = get_s3_client(cfg)
    # Read the raw JSON payload and hydrate a dataframe.
    obj = s3.get_object(Bucket=cfg.data.bucket, Key=raw_path)
    body = obj["Body"].read()
    raw_df = pd.read_json(io.BytesIO(body), lines=False)
    logger.info(f"{len(raw_df)} raw records read from s3 json")
    # Persist the bronze snapshot back to S3 as parquet for downstream steps.
    buf = io.BytesIO()
    raw_df.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
    buf.seek(0)
    s3.upload_fileobj(buf, cfg.data.bucket, bronze_path,
        ExtraArgs={"ContentType": "application/x-parquet"})
    logger.info(f"{len(raw_df)} bronze records wrote to s3 parquet")

    return raw_df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply lightweight, in-memory cleanup before persistence."""

    return df.copy()

@task
def offline(cfg: DictConfig, df: pd.DataFrame | None) -> pd.DataFrame:
    """Persist a cleaned bronze dataset into the offline store."""
    bronze_path = cfg.data.bronze_file
    offline_path = cfg.data.offline_file
    s3 = get_s3_client(cfg)
    if df is None:
        # Reload the bronze parquet snapshot if the upstream task was skipped.
        obj = s3.get_object(Bucket=cfg.data.bucket, Key=bronze_path)
        body = obj["Body"].read()
        bronze_df = pd.read_parquet(io.BytesIO(body))
        logger.info(f"{len(bronze_df)} bronze records read from s3")
    else:
        bronze_df = df.copy()
    offline_df = clean(bronze_df)
    # Write the cleaned dataset to the offline S3 location.
    buf = io.BytesIO()
    offline_df.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
    buf.seek(0)
    s3.upload_fileobj(buf, cfg.data.bucket, offline_path,
        ExtraArgs={"ContentType": "application/x-parquet"})
    logger.info(f"{len(offline_df)} offline records wrote to s3")

    return offline_df

def make_online(cfg: DictConfig, df: pd.DataFrame) -> pd.DataFrame:
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
def online(cfg: DictConfig, df: pd.DataFrame | None) -> pd.DataFrame:
    """Write an ordered, truncated dataset suitable for online serving."""
    offline_path = cfg.data.offline_file

    if df is None:
        s3 = get_s3_client(cfg)
        obj = s3.get_object(Bucket=cfg.data.bucket, Key=offline_path)
        body = obj["Body"].read()
        offline_df = pd.read_parquet(io.BytesIO(body))
        logger.info(f"{len(offline_df)} offline records read to s3")
    else:
        offline_df = df.copy()
    online_df = make_online(cfg, offline_df)
    db_path = cfg.onlinedb.duck_db_file
    table = cfg.onlinedb.table_name

    con = duckdb.connect(db_path)
    # Create the serving table if needed and populate it with the ordered rows.
    con.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM df")
    con.close()
    logger.info(f"{len(online_df)} online records wrote to duckdb")

    return online_df
