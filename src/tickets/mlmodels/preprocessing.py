"""Shared preprocessing utilities for ticket modeling pipelines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..schemas.ticket import TEXT_FEATURES, TEXT_LIST_FEATURES
from ..utils.log_util import ml_logger


@dataclass(frozen=True)
class DatasetSplit:
    """Chronological dataset partition used for model development."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def normalize_text(value: object) -> str:
    """Normalize heterogeneous textual inputs into a single-line string."""

    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    return ""


def normalize_str_list(values: list[str]) -> str:
    """Normalize list-like fields into a whitespace-delimited string."""

    tokens: list[str] = []
    for value in values:
        tokens.append(normalize_text(value))
    return "; ".join(tokens)


# src/tickets/mlmodels/preprocessing.py
class TextTransformer(BaseEstimator, TransformerMixin):
    """Collapse multiple textual columns into a single normalized document."""

    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = tuple(columns)

    def fit(self, X: pd.DataFrame, y: object = None) -> TextTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.Series:
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.columns)
        missing = [col for col in self.columns if col not in frame.columns]
        if missing:
            raise ValueError(f"TextTransformer missing columns: {missing}")

        combined: list[str] = []
        for row in frame.itertuples(index=False, name=None):
            tokens: list[str] = []
            for value, column in zip(row, self.columns, strict=False):
                if column in TEXT_FEATURES:
                    tokens.append(normalize_text(value))
                elif column in TEXT_LIST_FEATURES:
                    if isinstance(value, Sequence):
                        tokens.append(normalize_str_list(list(value)))
            combined.append(" ".join(token for token in tokens if token))
        return pd.Series(combined, index=frame.index, dtype=str)


def chronological_split(
    frame: pd.DataFrame,
    *,
    time_column: str = "created_at",
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
) -> DatasetSplit:
    """Split a dataframe into chronological train/validation/test partitions."""

    if frame.empty:
        raise ValueError("Input dataframe must not be empty.")

    ordered = frame.sort_values(
        time_column,
        ascending=True,
        kind="mergesort",
    ).reset_index(drop=True)
    total = len(ordered)
    if total < 3:
        raise ValueError("At least three records are required to produce chronological splits.")

    train_end = max(int(total * train_ratio), 1)
    train_end = min(train_end, total - 2)
    validation_end = max(int(total * (train_ratio + validation_ratio)), train_end + 1)
    validation_end = min(validation_end, total - 1)

    train_df = ordered.iloc[:train_end].reset_index(drop=True)
    validation_df = ordered.iloc[train_end:validation_end].reset_index(drop=True)
    test_df = ordered.iloc[validation_end:].reset_index(drop=True)

    ml_logger.info(
        "Chronological split sizes -> train: {}, validation: {}, test: {}.",
        len(train_df),
        len(validation_df),
        len(test_df),
    )

    return DatasetSplit(train=train_df, validation=validation_df, test=test_df)
