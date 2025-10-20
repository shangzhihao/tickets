"""Feature transformation utilities used across classical and deep learning pipelines.

The helpers in this module deliberately keep transformation logic free from model specific
concerns so that they can be reused in both offline analytics jobs and online inference.
The module uses pure functions that accept pandas objects and return NumPy arrays, aligning
with the RORO principle followed throughout the project.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from tickets.schemas.ticket import ENUM_FIELD_TYPES
from tickets.utils.config_util import CONFIG

TFIDF = TfidfVectorizer(
    min_df=CONFIG.tfidf.min_df,
    ngram_range=tuple(CONFIG.tfidf.ngram_range),
    max_features=CONFIG.tfidf.max_features,
    sublinear_tf=CONFIG.tfidf.sublinear_tf,
)


def tfidf_text_transformer(feature: pd.DataFrame) -> np.ndarray:
    """Vectorize textual tickets into TF-IDF representations.

    The transformer expects a dataframe with textual columns. All string-like values per row
    are concatenated into a single string before being fed into the shared `TFIDF` instance,
    enabling consistent vocabulary usage between train and inference contexts.
    """

    def combine_texts(row: pd.Series) -> str:
        """Join heterogeneous textual fields (str and sequences of str) into one string."""
        res = ""
        for value in row:
            if isinstance(value, str):
                res = res + value  # Preserve order to reflect original column importance.
            elif isinstance(value, Sequence):
                res = res + " ".join(value)
        return res

    combined = feature.apply(combine_texts, axis=1)
    vec = TFIDF.fit_transform(combined)
    return vec.toarray()


def bool_transformer(feature: pd.DataFrame) -> np.ndarray:
    """Cast boolean columns to integers so downstream estimators can ingest the features."""
    return feature.astype(int).to_numpy()


class CategoriesTransformer:
    """Encode categorical features using stable one-hot and ordinal projections."""

    def __init__(self, df: pd.DataFrame, *, needs_unknown: bool = False) -> None:
        self._df = df.copy()
        enum_columns = sorted(col for col in ENUM_FIELD_TYPES if col in self._df.columns)
        self.cols = enum_columns
        self.needs_unknown = needs_unknown
        self.col_value_map: dict[str, list[Any]] = {}
        self.col_num_class_map: dict[str, int] = {}
        for col in self.cols:
            series = self._df[col]
            values = series.dropna().drop_duplicates().tolist()
            if self.needs_unknown and "unknown" not in values:
                values.append("unknown")
            self.col_value_map[col] = values
            self.col_num_class_map[col] = len(values)

    def get_num_class(self, col: str) -> int:
        if col in self.col_num_class_map:
            return self.col_num_class_map[col]
        raise ValueError(f"{col} is not encoded")

    def one_hot(self, df: pd.DataFrame | None) -> np.ndarray:
        """Return one-hot encoded representation for the cached or supplied dataframe."""
        frame, columns = self._get_aligned_frame(df)
        encoded_columns: list[np.ndarray] = []

        for col in columns:
            categories = self.col_value_map[col]
            series = self._sanitize_series(frame[col], categories)
            categorical = pd.Categorical(series, categories=categories)
            encoded = pd.get_dummies(categorical, dtype=np.int64)
            encoded_columns.append(encoded.to_numpy())

        if not encoded_columns:
            return np.empty((frame.shape[0], 0), dtype=np.int64)

        return np.concatenate(encoded_columns, axis=1)

    def number(self, df: pd.DataFrame | None) -> np.ndarray:
        """Return ordinal encoded representation for the cached or supplied dataframe."""
        frame, columns = self._get_aligned_frame(df)
        encoded_columns: list[np.ndarray] = []

        for col in columns:
            categories = self.col_value_map[col]
            mapping = {value: idx for idx, value in enumerate(categories)}
            series = self._sanitize_series(frame[col], categories)
            encoded_columns.append(series.map(mapping).to_numpy(dtype=np.int64))

        if not encoded_columns:
            return np.empty((frame.shape[0], 0), dtype=np.int64)

        return np.column_stack(encoded_columns)

    def _get_aligned_frame(self, df: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
        """Align incoming frame with fitted column order."""
        source = self._df if df is None else df
        columns = [col for col in self.cols if col in source.columns]
        if not columns:
            empty_frame = pd.DataFrame(index=source.index)
            return empty_frame, []
        return source[columns], columns

    def _sanitize_series(
        self,
        series: pd.Series,
        categories: list[Any],
    ) -> pd.Series:
        """Map unseen or missing categories to the configured unknown token."""
        series = series.astype("object").copy()
        category_set = set(categories)
        allow_unknown = "unknown" in category_set

        invalid_mask = ~series.isin(category_set) & series.notna()
        if invalid_mask.any():
            if allow_unknown:
                series.loc[invalid_mask] = "unknown"
            else:
                invalid_values = series.loc[invalid_mask].unique().tolist()
                raise ValueError(
                    f"Found unseen categories {invalid_values} and unknown handling disabled.",
                )

        if allow_unknown:
            series = series.fillna("unknown")
        elif series.isna().any():
            raise ValueError("NaN values encountered without unknown category enabled.")

        return series


def num_transformer(feature: pd.DataFrame) -> np.ndarray:
    """Replace missing numerical values with zeros and expose a dense NumPy view."""
    cleaned = feature.fillna(0.0)
    return cleaned.to_numpy()
