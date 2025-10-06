"""Shared IO utilities"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import boto3
import redis
from loguru import logger as _logger
from botocore.client import BaseClient, Config

from .config import cfg

LOG_PATH: Final[Path] = Path("logs/app.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

_logger.remove()
_logger.add(
    LOG_PATH,
    serialize=True,
    enqueue=True,
    rotation="10 MB",
    retention="7 days",
    compression="zip",
)
_logger.add(lambda message: print(message, end=""))

logger = _logger

redis_pool: Final[redis.ConnectionPool] = redis.ConnectionPool(
    host=cfg.redis_host,
    port=cfg.redis_port,
    db=0,
    max_connections=50,
    decode_responses=False,
)


s3_client: Final[BaseClient] = boto3.client(
    "s3",
    endpoint_url=cfg.data.endpoint,
    aws_access_key_id=cfg.data.access_key,
    aws_secret_access_key=cfg.data.secret_key,
    config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    region_name="us-east-1",
)

__all__ = ["logger", "redis_pool", "s3_client"]
