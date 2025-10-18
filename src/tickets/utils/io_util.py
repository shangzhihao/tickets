"""Shared IO utilities configured via Hydra logging defaults."""

from __future__ import annotations

import io
from typing import Final

import boto3
import pandas as pd
import redis
from botocore.client import BaseClient, Config

from tickets.schemas.events import DataLoadOnS3Event
from tickets.utils.config_util import CONFIG

redis_pool: Final[redis.ConnectionPool] = redis.ConnectionPool(
    host=CONFIG.redis_host,
    port=CONFIG.redis_port,
    password=CONFIG.redis_password,
    db=0,
    max_connections=50,
    decode_responses=False,
)


s3_client: Final[BaseClient] = boto3.client(
    "s3",
    endpoint_url=CONFIG.data.endpoint,
    aws_access_key_id=CONFIG.data.access_key,
    aws_secret_access_key=CONFIG.data.secret_key,
    config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    region_name="us-east-1",
)


def load_df_from_s3(data_path: str, group: str) -> pd.DataFrame:
    # Read the raw JSON payload and hydrate a dataframe.
    obj = s3_client.get_object(Bucket=CONFIG.data.bucket, Key=data_path)
    body = obj["Body"].read()
    if data_path.endswith("json"):
        df = pd.read_json(io.BytesIO(body), lines=False)
    elif data_path.endswith("parquet"):
        df = pd.read_parquet(io.BytesIO(body))
    elif data_path.endswith("csv"):
        df = pd.read_csv(io.BytesIO(body))
    else:
        raise ValueError("Unknown file type.")
    DataLoadOnS3Event(feature_group=group, entity_key=data_path, records_loaded=len(df)).emit()
    return df


def save_df_to_s3(df: pd.DataFrame, data_path: str, group: str) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
    buf.seek(0)
    s3_client.upload_fileobj(
        buf,
        CONFIG.data.bucket,
        data_path,
        ExtraArgs={"ContentType": "application/x-parquet"},
    )
    DataLoadOnS3Event(feature_group=group, entity_key=data_path, records_loaded=len(df)).emit()
