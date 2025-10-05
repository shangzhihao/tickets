"""Ingest raw ticket data and persist it as parquet."""

from __future__ import annotations

import io
from functools import lru_cache

import boto3
import pandas as pd
import redis
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
def ingest(cfg: DictConfig)->None:
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


def clean(cfg: DictConfig, df: pd.DataFrame) -> pd.DataFrame:
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
    offline_df = clean(cfg,bronze_df)
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
    cols = ['created_at', 'updated_at', 'resolved_at']
    online_df[cols] = online_df[cols].apply(
        lambda x: x.dt.strftime("%Y-%m-%dT%H:%M:%S%z"))

    r = redis.Redis(
        host=cfg.redis_host,
        port=cfg.redis_port,
        decode_responses=True
        )
    pipe = r.pipeline()
    path = "$"
    records = online_df.to_dict(orient="records")
    for ticket in records:
        pipe.json().set(
            name=ticket["ticket_id"],
            path=path,
            obj=ticket)
    results = pipe.execute()
    logger.info(f"{sum(results)} records wrote into redis")
    logger.info(f"{len(results) - sum(results)} records failed to write into redis")
    return online_df
