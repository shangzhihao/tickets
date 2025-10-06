
from __future__ import annotations

import io
from typing import Final

import pandas as pd
from ..utils.config_util import cfg
from ..utils.io_util import data_logger, s3_client


OFFLINE_PATH: Final[str] = cfg.data.offline_file
BUCKET_NAME: Final[str] = cfg.data.bucket



obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=OFFLINE_PATH)
body = obj["Body"].read()
offline_df = pd.read_parquet(io.BytesIO(body))