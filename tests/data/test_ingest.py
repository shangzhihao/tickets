from __future__ import annotations

import importlib
import io
import sys
from collections.abc import Generator
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


class _FakePipeline:
    """Minimal RedisJSON pipeline stub that records JSON set operations."""

    def __init__(self) -> None:
        self.commands: list[tuple[str, str, dict[str, object]]] = []

    def json(self) -> _FakePipeline:
        return self

    def set(self, name: str, path: str, obj: dict[str, object]) -> _FakePipeline:
        self.commands.append((name, path, obj.copy()))
        return self

    def execute(self) -> list[int]:
        return [1 for _ in self.commands]


class _FakeRedisClient:
    """Redis client stub that provides a deterministic pipeline instance."""

    def __init__(self) -> None:
        self.pipeline_invocations = 0
        self.pipeline_instance = _FakePipeline()

    def pipeline(self) -> _FakePipeline:
        self.pipeline_invocations += 1
        return self.pipeline_instance


@pytest.fixture()
def ingest_mod(monkeypatch: pytest.MonkeyPatch) -> Generator[
    tuple[ModuleType, SimpleNamespace, Mock, Mock, list[_FakeRedisClient]], None, None
]:
    """Import tickets.data.ingest with patched IO dependencies for unit testing."""

    module = importlib.import_module("tickets.data.ingest")
    module = importlib.reload(module)

    cfg_stub = SimpleNamespace(
        data=SimpleNamespace(
            bucket="test-bucket",
            raw_file="raw.json",
            bronze_file="bronze.parquet",
            offline_file="offline.parquet",
            num_online=2,
        )
    )

    fake_s3: Mock = Mock()
    fake_logger: Mock = Mock()
    redis_clients: list[_FakeRedisClient] = []

    def _redis_constructor(*_: object, **__: object) -> _FakeRedisClient:
        client = _FakeRedisClient()
        redis_clients.append(client)
        return client

    monkeypatch.setattr(module, "cfg", cfg_stub, raising=False)
    monkeypatch.setattr(module, "s3_client", fake_s3, raising=False)
    monkeypatch.setattr(module, "data_logger", fake_logger, raising=False)
    monkeypatch.setattr(module, "redis_pool", object(), raising=False)
    monkeypatch.setattr(module.redis, "Redis", _redis_constructor, raising=False)

    try:
        yield module, cfg_stub, fake_s3, fake_logger, redis_clients
    finally:
        sys.modules.pop("tickets.data.ingest", None)


def test_bronze_reads_raw_json_and_persists_parquet(
    ingest_mod: tuple[
        ModuleType, SimpleNamespace, Mock, Mock, list[_FakeRedisClient]
    ],
) -> None:
    module, cfg_stub, fake_s3, fake_logger, _ = ingest_mod

    raw_df = pd.DataFrame(
        [
            {"ticket_id": "t1", "created_at": "2024-01-01T00:00:00Z", "value": 1},
            {"ticket_id": "t2", "created_at": "2024-01-02T00:00:00Z", "value": 2},
        ]
    )
    raw_df["created_at"] = pd.to_datetime(raw_df["created_at"], utc=True)
    raw_bytes = raw_df.to_json(date_unit="ns", date_format="iso").encode()
    fake_s3.get_object.return_value = {"Body": io.BytesIO(raw_bytes)}

    result = module.bronze.fn()

    assert_frame_equal(result, raw_df)
    fake_s3.get_object.assert_called_once_with(
        Bucket=cfg_stub.data.bucket, Key=cfg_stub.data.raw_file
    )
    fake_s3.upload_fileobj.assert_called_once()
    upload_buffer = fake_s3.upload_fileobj.call_args.args[0]
    upload_buffer.seek(0)
    persisted_df = pd.read_parquet(upload_buffer)
    assert_frame_equal(persisted_df, raw_df)
    assert "raw records read" in fake_logger.info.call_args_list[0].args[0]


def test_offline_uses_provided_dataframe_and_clean_hook(
    ingest_mod: tuple[
        ModuleType, SimpleNamespace, Mock, Mock, list[_FakeRedisClient]
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, cfg_stub, fake_s3, fake_logger, _ = ingest_mod
    fake_s3.reset_mock()
    fake_logger.reset_mock()

    input_df = pd.DataFrame(
        [
            {"ticket_id": "t1", "created_at": "2024-01-01T00:00:00Z"},
            {"ticket_id": "t2", "created_at": "2024-01-02T00:00:00Z"},
        ]
    )

    def _clean_hook(df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        cleaned["cleaned"] = True
        return cleaned

    monkeypatch.setattr(module, "clean", _clean_hook, raising=False)

    result = module.offline.fn(input_df)

    fake_s3.get_object.assert_not_called()
    fake_s3.upload_fileobj.assert_called_once()
    upload_buffer = fake_s3.upload_fileobj.call_args.args[0]
    upload_buffer.seek(0)
    persisted_df = pd.read_parquet(upload_buffer)
    assert_frame_equal(result, persisted_df)
    assert "offline records wrote" in fake_logger.info.call_args_list[-1].args[0]
    assert "cleaned" in result.columns


def test_offline_reads_bronze_snapshot_when_upstream_missing(
    ingest_mod: tuple[
        ModuleType, SimpleNamespace, Mock, Mock, list[_FakeRedisClient]
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, cfg_stub, fake_s3, fake_logger, _ = ingest_mod
    fake_s3.reset_mock()
    fake_logger.reset_mock()

    bronze_df = pd.DataFrame(
        [
            {"ticket_id": "t3", "created_at": "2024-01-03T00:00:00Z"},
            {"ticket_id": "t4", "created_at": "2024-01-04T00:00:00Z"},
        ]
    )
    bronze_buf = io.BytesIO()
    bronze_df.to_parquet(bronze_buf, index=False)
    bronze_buf.seek(0)
    fake_s3.get_object.return_value = {"Body": io.BytesIO(bronze_buf.getvalue())}
    monkeypatch.setattr(module, "clean", lambda df: df, raising=False)

    result = module.offline.fn(None)

    fake_s3.get_object.assert_called_once_with(
        Bucket=cfg_stub.data.bucket, Key=cfg_stub.data.bronze_file
    )
    fake_s3.upload_fileobj.assert_called_once()
    assert_frame_equal(result, bronze_df)


def test_make_online_orders_by_timestamp_and_limits_rows(
    ingest_mod: tuple[
        ModuleType, SimpleNamespace, Mock, Mock, list[_FakeRedisClient]
    ],
) -> None:
    module, cfg_stub, _, _, _ = ingest_mod
    cfg_stub.data.num_online = 2

    input_df = pd.DataFrame(
        {
            "ticket_id": ["t1", "t2", "t3"],
            "created_at": [
                "2024-01-02T12:00:00Z",
                "2024-01-03T00:00:00Z",
                None,
            ],
            "updated_at": [
                "2024-01-02T12:05:00Z",
                "2024-01-03T00:05:00Z",
                "2024-01-01T00:05:00Z",
            ],
        }
    )

    result = module.make_online(input_df)

    assert len(result) == 2
    assert list(result["ticket_id"]) == ["t2", "t1"]
    assert result["created_at"].iloc[0].tzinfo is not None


def test_make_online_requires_created_at_column(
    ingest_mod: tuple[
        ModuleType, SimpleNamespace, Mock, Mock, list[_FakeRedisClient]
    ],
) -> None:
    module, _, _, _, _ = ingest_mod

    with pytest.raises(KeyError):
        module.make_online(pd.DataFrame({"ticket_id": ["t1"]}))


def test_make_online_validates_positive_row_budget(
    ingest_mod: tuple[
        ModuleType, SimpleNamespace, Mock, Mock, list[_FakeRedisClient]
    ],
) -> None:
    module, cfg_stub, _, _, _ = ingest_mod
    cfg_stub.data.num_online = 0

    with pytest.raises(ValueError):
        module.make_online(
            pd.DataFrame({"ticket_id": ["t1"], "created_at": ["2024-01-01T00:00:00Z"]})
        )

    cfg_stub.data.num_online = 2


def test_online_serializes_records_and_writes_via_pipeline(
    ingest_mod: tuple[
        ModuleType, SimpleNamespace, Mock, Mock, list[_FakeRedisClient]
    ],
) -> None:
    module, cfg_stub, _, fake_logger, redis_clients = ingest_mod
    fake_logger.reset_mock()

    cfg_stub.data.num_online = 5
    input_df = pd.DataFrame(
        {
            "ticket_id": ["t1", "t2"],
            "created_at": [
                "2024-01-01T00:00:00Z",
                "2024-01-02T01:30:00Z",
            ],
            "updated_at": [
                "2024-01-01T00:05:00Z",
                "2024-01-02T01:35:00Z",
            ],
            "resolved_at": [
                "2024-01-01T02:00:00Z",
                "2024-01-02T03:00:00Z",
            ],
        }
    )
    for column in ("created_at", "updated_at", "resolved_at"):
        input_df[column] = pd.to_datetime(input_df[column], utc=True)

    result = module.online.fn(input_df)

    assert len(redis_clients) == 1
    pipeline = redis_clients[0].pipeline_instance
    assert pipeline.commands == [
        (
            row["ticket_id"],
            "$",
            row,
        )
        for row in result.to_dict(orient="records")
    ]
    assert all(value.endswith("+0000") for value in result["created_at"])
    assert "Records written to redis" in fake_logger.info.call_args_list[-1].args[0]
