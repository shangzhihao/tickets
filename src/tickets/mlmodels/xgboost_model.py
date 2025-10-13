"""XGBoost training utilities for ticket classification tasks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
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
from ..utils.log_util import ml_logger
from .preprocessing import TextTransformer

FEATURES_COLS = TEXT_FEATURES + BOOL_FEATURES + TEXT_LIST_FEATURES + CAT_FEATURES + NUM_FEATURES


@dataclass(frozen=True)
class ModelResult:
    """Container holding an individual target model and its evaluation metrics."""

    target: str
    pipeline: Pipeline
    label_encoder: LabelEncoder
    classes: tuple[str, ...]
    metrics: dict[str, dict[str, float]]


class XGBModel:
    def __init__(
        self,
        training_data: pd.DataFrame,
        target: str,
        validation_data: pd.DataFrame | None = None,
    ) -> None:
        self.training_data = training_data.copy()
        self.validation_data = validation_data.copy() if validation_data is not None else None
        self.target = target
        self.model_result: ModelResult | None = None
        self.label_encoder = LabelEncoder()
        self._is_trained = False
        # initialized in _prepare_data
        self.train_x = pd.DataFrame()
        self.train_y = pd.Series()
        self.valid_x = pd.DataFrame() if validation_data is not None else None
        self.valid_y = pd.Series() if validation_data is not None else None
        self._prepare_data()
        self.num_classes = len(self.training_data[self.target].unique())
        self.clf_config = OmegaConf.to_container(cfg.xgboost, resolve=True)
        if isinstance(self.clf_config, dict) is False:
            raise ValueError("XGBoost config must be a dictionary.")
        self.text_pipeline = Pipeline(
            [
                ("combine", TextTransformer(TEXT_FEATURES + TEXT_LIST_FEATURES)),
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=20000,
                        ngram_range=(1, 3),
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
        self.preprocess = ColumnTransformer(
            transformers=[
                ("text", self.text_pipeline, TEXT_FEATURES + TEXT_LIST_FEATURES),
                (
                    "category",
                    OneHotEncoder(handle_unknown="ignore", dtype=np.int8),
                    CAT_FEATURES,
                ),
                ("num", "passthrough", NUM_FEATURES),
                ("bool", "passthrough", BOOL_FEATURES),
            ],
            remainder="drop",
            sparse_threshold=1.0,  # keep it sparse; good for XGBoost
        )
        if validation_data is not None:
            self.model = XGBClassifier(
                num_class=self.num_classes,
                **self.clf_config,
            )
        else:
            self.clf_config.pop("early_stopping_rounds", None)
            self.model = XGBClassifier(
                num_class=self.num_classes,
                **self.clf_config,
            )

    def _prepare_data(self) -> None:
        training_data = self._ensure_columns(self.training_data)
        self.train_x = training_data[FEATURES_COLS]
        self.train_y = training_data[self.target]
        if self.validation_data is not None:
            validation_data = self._ensure_columns(self.validation_data)
            self.valid_x = validation_data[FEATURES_COLS]
            self.valid_y = validation_data[self.target]

    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("Input DataFrame must contain at least one record.")
        missing_features = [feature for feature in FEATURES_COLS if feature not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features in data: {missing_features}")
        if self.target not in df.columns:
            raise ValueError(f"Data must include target column `{self.target}`.")
        return df

    def train(self) -> XGBClassifier:
        _train_x = self.preprocess.fit_transform(self.train_x)
        self.label_encoder.fit(self.train_y)
        _num_classes = len(self.label_encoder.classes_)
        if self.num_classes != _num_classes:
            raise ValueError(
                f"Number of classes mismatch for target {self.target}: "
                f"expected {self.num_classes}, got {_num_classes}"
            )
        _train_y = self.label_encoder.transform(self.train_y)
        if self.valid_x is not None and self.valid_y is not None:
            _valid_x = self.preprocess.transform(self.valid_x)
            _valid_y = self.label_encoder.transform(self.valid_y)
            self.model.fit(
                _train_x,
                _train_y,
                eval_set=[(_valid_x, _valid_y)],
                verbose=False,
            )
        else:
            self.model.fit(_train_x, _train_y, verbose=False)
        self._is_trained = True
        return self.model

    def evaluate(self, test_data: pd.DataFrame) -> ModelResult:
        if not self._is_trained:
            raise RuntimeError("Model must be trained before evaluation. Call `train` first.")
        _test_data = self._ensure_columns(test_data)
        features = _test_data[FEATURES_COLS]
        y_true = _test_data[self.target]
        transformed_features = self.preprocess.transform(features)
        encoded_predictions = self.model.predict(transformed_features)
        y_pred = self.label_encoder.inverse_transform(encoded_predictions)

        metrics = classification_report(
            y_true=y_true,
            y_pred=y_pred,
            labels=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        )
        ml_logger.info(
            "Completed XGBoost evaluation for target `{target}`.",
            target=self.target,
            metrics=metrics,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocess", self.preprocess),
                ("model", self.model),
            ]
        )
        self.model_result = ModelResult(
            target=self.target,
            pipeline=pipeline,
            label_encoder=self.label_encoder,
            classes=tuple(self.label_encoder.classes_),
            metrics=metrics,
        )
        return self.model_result

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction. Call `train` first.")
        _X = self._ensure_columns(X)
        features = _X[FEATURES_COLS]
        transformed_features = self.preprocess.transform(features)
        encoded_predictions = self.model.predict(transformed_features)
        predictions = self.label_encoder.inverse_transform(encoded_predictions)

        ml_logger.info(
            "Generated predictions for target `{target}`.",
            target=self.target,
            num_predictions=int(predictions.size),
        )
        return pd.Series(predictions, index=_X.index, name=self.target)
