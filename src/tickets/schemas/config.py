from __future__ import annotations

from pathlib import Path

from loguru import logger
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = ".env"
ENV_FILE_PATH = Path(ENV_FILE)

if not ENV_FILE_PATH.is_file():
    logger.warning(
        "Environment file missing at expected path: {}. Defaults or external env vars will apply.",
        ENV_FILE_PATH.resolve(),
    )


def _default_xgboost_param_grid() -> dict[str, list[int | float]]:
    """Provide default grid settings for XGBoost tuning."""
    return {
        "n_estimators": [500, 1000],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9],
        "reg_lambda": [0.5, 1.0],
    }


class EmbeddingsConfig(BaseModel):
    """Configuration for embedding backends."""

    model_name: str = "all-MiniLM-L6-v2"

    model_config = ConfigDict(frozen=True)


class SplitConfig(BaseModel):
    """Data split ratios."""

    train: float = 0.7
    val: float = 0.15
    test: float = 0.15

    model_config = ConfigDict(frozen=True)


class DataConfig(BaseModel):
    """Object store configuration for dataset artefacts."""

    type: str = "minio"
    endpoint: str = Field(
        default="minioadmin",
        validation_alias=AliasChoices("MINIO_ENDPOINT"),
    )
    bucket: str = Field(
        default="minioadmin",
        validation_alias=AliasChoices("MINIO_BUCKET"),
    )
    raw_file: str = "raw/tickets.json"
    bronze_file: str = "bronze/tickets.parquet"
    offline_file: str = "offline/tickets.parquet"
    clean_file: str = "cleaned/tickets.parquet"
    online_file: str = "online/tickets.parquet"
    metrics_file: str = "offline/metrics.json"
    quality_report_file: str = "offline/quality_report.json"
    access_key: str = Field(
        default="minioadmin",
        validation_alias=AliasChoices("MINIO_ROOT_USER", "MINIO_ACCESS_KEY"),
    )
    secret_key: str = Field(
        default="minioadmin",
        validation_alias=AliasChoices("MINIO_ROOT_PASSWORD", "MINIO_SECRET_KEY"),
    )
    region: str = "us-east-1"
    num_online: int = Field(
        default=10000,
        validation_alias=AliasChoices("MINIO_NUM_ONLINE"),
    )

    model_config = ConfigDict(frozen=True)


class LoggingFileSinkConfig(BaseModel):
    """Loguru file sink behaviour."""

    rotation: str = "10 MB"
    retention: str = "7 days"
    compression: str = "zip"
    serialize: bool = True
    enqueue: bool = True
    level: str = "INFO"
    output_dir: str = "logs"
    file_name: str = "tickets.log"

    model_config = ConfigDict(frozen=True)


class LoggingConsoleSinkConfig(BaseModel):
    """Loguru console sink behaviour."""

    enabled: bool = True
    colorize: bool = True
    format: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level:<8}</level> | <cyan>{name}:{function}:{line}</cyan> | "
        "<level>{message}</level>"
    )
    level: str = "INFO"

    model_config = ConfigDict(frozen=True)


class LoggingConfig(BaseModel):
    """Application-wide logging configuration."""

    file_sink: LoggingFileSinkConfig = Field(default_factory=LoggingFileSinkConfig)
    console_sink: LoggingConsoleSinkConfig = Field(default_factory=LoggingConsoleSinkConfig)
    backtrace: bool = False
    diagnose: bool = False

    model_config = ConfigDict(frozen=True)


class XGBoostParamsConfig(BaseModel):
    """Default parameters for the XGBoost model."""

    n_estimators: int = 1000
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    objective: str = "multi:softprob"
    eval_metric: str = "mlogloss"
    random_state: int = 42
    early_stopping_rounds: int = 10
    tree_method: str = "hist"
    predictor: str | None = None
    device: str | None = None

    model_config = ConfigDict(frozen=True)


class XGBoostGridSearchConfig(BaseModel):
    """Grid search configuration for XGBoost."""

    sample_cap: int = 5000
    enabled: bool = True
    cv: int = 3
    scoring: str = "f1_macro"
    n_jobs: int = -1
    verbose: int = 1
    refit: str = "f1_macro"
    param_grid: dict[str, list[int | float]] = Field(default_factory=_default_xgboost_param_grid)

    model_config = ConfigDict(frozen=True)


class XGBoostConfig(BaseModel):
    """Aggregated XGBoost configuration."""

    gbrt_params: XGBoostParamsConfig = Field(default_factory=XGBoostParamsConfig)
    grid_search: XGBoostGridSearchConfig = Field(default_factory=XGBoostGridSearchConfig)

    model_config = ConfigDict(frozen=True)


class DNNConfig(BaseModel):
    """Hyper-parameters for neural network models."""

    dropout: float = 0.1
    lr: float = 1e-5
    epoch: int = 1
    hidden: int = 128
    batch_size: int = 1024
    patience: int = 5
    device: str = "mps"
    dl_num_worker: int = 7

    model_config = ConfigDict(frozen=True)


class TfIdfConfig(BaseModel):
    """TF-IDF feature extraction configuration."""

    min_df: int = 2
    ngram_range: tuple[int, int] = (1, 3)
    max_features: int = 200
    sublinear_tf: bool = True

    model_config = ConfigDict(frozen=True)


class MlflowConfig(BaseModel):
    """MLflow tracking server configuration."""

    host: str = Field(
        default="127.0.0.1",
        validation_alias=AliasChoices("HOST"),
    )
    port: int = Field(
        default=5001,
        validation_alias=AliasChoices("PORT"),
    )

    model_config = ConfigDict(frozen=True)


class AppConfig(BaseSettings):
    """Application configuration backed by Pydantic settings."""

    project_name: str = "tickets"
    seed: int = 42
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    redis_host: str = Field(
        default="localhost",
        validation_alias=AliasChoices("REDIS_HOST", "REDIS__HOST"),
    )
    redis_port: int = Field(
        default=6379,
        validation_alias=AliasChoices("REDIS_PORT", "REDIS__PORT"),
    )
    redis_password: str = Field(
        default="password",
        validation_alias=AliasChoices("REDIS_PASSWORD", "REDIS__PASSWORD"),
    )
    split: SplitConfig = Field(default_factory=SplitConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    dnn: DNNConfig = Field(default_factory=DNNConfig)
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    tfidf: TfIdfConfig = Field(default_factory=TfIdfConfig)
    mlflow: MlflowConfig = Field(default_factory=MlflowConfig)

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )


__all__ = [
    "AppConfig",
    "DataConfig",
    "DNNConfig",
    "EmbeddingsConfig",
    "LoggingConfig",
    "LoggingConsoleSinkConfig",
    "LoggingFileSinkConfig",
    "MlflowConfig",
    "SplitConfig",
    "TfIdfConfig",
    "XGBoostConfig",
    "XGBoostGridSearchConfig",
    "XGBoostParamsConfig",
]
