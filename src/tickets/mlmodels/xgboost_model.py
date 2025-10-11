"""XGBoost training utilities for ticket classification tasks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

from ..schemas.ticket import (
    BOOL_FEATURES,
    CAT_FEATURES,
    NUM_FEATURES,
    TEXT_FEATURES,
    TEXT_LIST_FEATURES,
)
from ..utils.config_util import cfg
from ..utils.io_util import load_df_from_s3
from .preprocessing import TextTransformer, chronological_split


@dataclass(frozen=True)
class ModelResult:
    """Container holding an individual target model and its evaluation metrics."""

    target: str
    pipeline: Pipeline
    label_encoder: LabelEncoder
    classes: tuple[str, ...]
    metrics: dict[str, dict[str, float]]


def train_xgboost_models() -> None:
    """Train XGBoost models for category, subcategory, and sentiment prediction."""

    offline_frame = load_df_from_s3(
        data_path=cfg.data.offline_file,
        group=__file__,
    )

    text_pipeline = Pipeline(
        [
            ("combine", TextTransformer(TEXT_FEATURES + TEXT_LIST_FEATURES)),
            ("tfidf", TfidfVectorizer(max_features=2000, ngram_range=(1, 2))),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("text", text_pipeline, TEXT_FEATURES + TEXT_LIST_FEATURES),
            ("category", OneHotEncoder(handle_unknown="ignore", dtype=np.int8), CAT_FEATURES),
            ("num", "passthrough", NUM_FEATURES),
            ("bool", "passthrough", BOOL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=1.0,  # keep it sparse; good for XGBoost
    )
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
    )
    splits = chronological_split(offline_frame[:20000])
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", clf),
        ]
    )
    features = TEXT_FEATURES + BOOL_FEATURES + TEXT_LIST_FEATURES + CAT_FEATURES + NUM_FEATURES
    train_x = splits.train[features]
    label_encoder = LabelEncoder()
    train_y = label_encoder.fit_transform(splits.train["customer_sentiment"])
    pipe.fit(train_x, train_y)

    test_x = splits.test[features]
    pred_y = pipe.predict(test_x)
    test_y = label_encoder.fit_transform(splits.test["customer_sentiment"])

    print(classification_report(test_y, pred_y))
