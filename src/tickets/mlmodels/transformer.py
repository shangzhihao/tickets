"""Feature transformation utilities used across classical and deep learning pipelines.

The helpers in this module deliberately keep transformation logic free from model specific
concerns so that they can be reused in both offline analytics jobs and online inference.
The module uses pure functions that accept pandas objects and return NumPy arrays, aligning
with the RORO principle followed throughout the project.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from tickets.schemas.ticket import ENUM_FIELD_TYPES
from tickets.utils.config_util import CONFIG
from tickets.utils.log_util import ML_LOGGER

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


def cat_onehot_transformer(feature: pd.Series, enum_type: type[Enum]) -> np.ndarray:
    """Convert a categorical enum-backed series to a one-hot encoded matrix."""
    categories = list(enum_type)
    num_categories = len(categories)
    one_hot = np.zeros((feature.shape[0], num_categories), dtype=int)

    for row_index, raw_value in enumerate(feature):
        try:
            enum_item = enum_type(raw_value)
        except (KeyError, ValueError, TypeError):
            ML_LOGGER.warning(f"Unknown category '{raw_value}' encountered in categorical feature.")
            # NOTE: This branch currently tries to flag an unknown bucket but the array does not
            # reserve an extra column. The behaviour should be revisited when adding fallback bins.
            continue
        category_index = categories.index(enum_item)
        one_hot[row_index, category_index] = 1

    return one_hot


def cats_onehot_transformer(features: pd.DataFrame) -> np.ndarray:
    """Apply `cat_onehot_transformer` column-wise and horizontally stack the results."""
    cat_vecs = []
    for cat_col in features.columns:
        enum_type = ENUM_FIELD_TYPES[cat_col]
        cat_vec = cat_onehot_transformer(features[cat_col], enum_type)
        cat_vecs.append(cat_vec)
    return np.hstack(cat_vecs)


def cat_num_transformer(feature: pd.Series, enum_type: type[Enum]) -> np.ndarray:
    """Map a categorical enum-backed series into ordinal indices."""

    def cat_to_num(v: str | int) -> int:
        cats = list(enum_type)
        enum_cat = enum_type(v)
        return cats.index(enum_cat)

    return feature.apply(cat_to_num).to_numpy().reshape(-1, 1)


def cats_num_transformer(features: pd.DataFrame) -> np.ndarray:
    """Stack ordinal encodings for multiple categorical columns sharing enum metadata."""
    cat_vecs = []
    for cat_feature in features.columns:
        enum_type = ENUM_FIELD_TYPES[cat_feature]
        cat_vec = cat_num_transformer(features[cat_feature], enum_type=enum_type)
        cat_vecs.append(cat_vec)
    return np.hstack(cat_vecs)


def num_transformer(feature: pd.DataFrame) -> np.ndarray:
    """Replace missing numerical values with zeros and expose a dense NumPy view."""
    cleaned = feature.fillna(0.0)
    return cleaned.to_numpy()
