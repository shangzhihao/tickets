"""Ingest raw ticket data and persist it as parquet."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import ijson
import pandas as pd
from omegaconf import DictConfig

from ..schemas.ticket import Ticket
from prefect import flow, task

@flow
def ingest(cfg: DictConfig):
    bronze_path, df = bronze(cfg)
    offline_path, df = offline(cfg, df)
    online_path, df = online(cfg, df)

@task
def bronze(cfg: DictConfig) -> Tuple[Path, pd.DataFrame]:
    """Load tickets from JSON, validate them, and save to parquet.

    Parameters
    ----------
    cfg
        Hydra configuration with `data.raw_input` and `data.offline` paths.

    Returns
    -------
    Tuple[Path, pandas.DataFrame]
        The path to the written parquet file and the in-memory frame.
    """

    raw_path = Path(cfg.data.raw_input)
    output_path = cfg.data.bronze

    tickets: List[dict] = []
    with raw_path.open("r", encoding="utf-8") as handle:
        for ticket in ijson.items(handle, "item"):
            tickets.append(Ticket.model_validate(ticket).model_dump(mode="json"))

    if not tickets:
        raise ValueError(f"No tickets were ingested from {raw_path}")

    frame = pd.DataFrame(tickets)
    frame.to_parquet(output_path, index=False)

    return output_path, frame


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply lightweight, in-memory cleanup before persistence."""

    return df.copy()

@task
def offline(cfg: DictConfig, df: pd.DataFrame | None) -> Tuple[Path, pd.DataFrame]:
    """Persist a cleaned bronze dataset into the offline store."""
    in_path = Path(cfg.data.bronze)
    out_path = Path(cfg.data.offline)
    frame = df.copy() if df is not None else None
    if frame is None:
        frame = pd.read_parquet(in_path)

    frame = clean(frame)
    frame.to_parquet(out_path, index=False)
    return out_path, frame


def make_online(cfg: DictConfig, df: pd.DataFrame) -> pd.DataFrame:
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
def online(cfg: DictConfig, df: pd.DataFrame | None) -> Tuple[Path, pd.DataFrame]:
    """Write an ordered, truncated dataset suitable for online serving."""
    in_path = Path(cfg.data.offline)
    out_path = Path(cfg.data.online)
    frame = df.copy() if df is not None else None
    if frame is None:
        frame = pd.read_parquet(in_path)

    frame = make_online(cfg, frame)
    frame.to_parquet(out_path, index=False)
    return out_path, frame
