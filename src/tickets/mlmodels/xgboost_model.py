"""XGBoost training utilities for ticket classification tasks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
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
    CAT_MAPPING,
    NUM_FEATURES,
    TARGETS,
    TEXT_FEATURES,
    TEXT_LIST_FEATURES,
    Category,
    CustomerSentiment,
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


def xgboost_cat(target: str, category: Category | None = None) -> None:
    """Train XGBoost models for category prediction."""

    offline_frame = load_df_from_s3(
        data_path=cfg.data.offline_file,
        group=__file__,
    )

    text_pipeline = Pipeline(
        [
            ("combine", TextTransformer(TEXT_FEATURES + TEXT_LIST_FEATURES)),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20000, ngram_range=(1, 3), min_df=2, sublinear_tf=True
                ),
            ),
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
    df = pd.DataFrame()
    num_classes = 0
    if target not in TARGETS:
        raise ValueError(f"Target {target} not in {TARGETS}")
    if target == "subcategory" and category is None:
        raise ValueError("Category must be provided when target is subcategory")
    elif target == "category":
        df = offline_frame
        num_classes = len(Category)
    elif target == "subcategory" and category is not None:
        df = offline_frame[offline_frame["category"] == Category(category)]
        num_classes = len(CAT_MAPPING[Category(category)])
    elif target == "customer_sentiment":
        df = offline_frame
        num_classes = len(CustomerSentiment)
    # prepare data
    splits = chronological_split(df)
    features = TEXT_FEATURES + BOOL_FEATURES + TEXT_LIST_FEATURES + CAT_FEATURES + NUM_FEATURES
    train_x = splits.train[features]
    label_encoder = LabelEncoder()
    train_y = label_encoder.fit_transform(splits.train[target])
    test_y = label_encoder.fit_transform(splits.test[target])
    # define model
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="multi:softprob",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        num_class=num_classes,
    )
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", clf),
        ]
    )
    # train model
    pipe.fit(train_x, train_y)
    # evaluate model
    test_x = splits.test[features]
    pred_y = pipe.predict(test_x)

    print(classification_report(test_y, pred_y))
