"""XGBoost trainer implementation for ticket classification."""

from __future__ import annotations

from collections.abc import Sequence

import mlflow
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from tickets.mlmodels.base import ModelTrainer
from tickets.utils.config_util import CONFIG
from tickets.utils.log_util import XGB_LOGGER


class XGBTrainer(ModelTrainer[XGBClassifier]):
    """Trainer orchestrating XGBoost fitting and validation."""

    def __init__(
        self,
        *,
        train_set: tuple[np.ndarray, np.ndarray],
        val_set: tuple[np.ndarray, np.ndarray],
        target_names: Sequence[str] | None = None,
        exp_name: str = "xgb_ticket_classifier",
    ) -> None:
        train_x, train_y = self._normalise_dataset(train_set, split_name="train")
        val_x, val_y = self._normalise_dataset(val_set, split_name="validation")
        if train_x.shape[1] != val_x.shape[1]:
            raise ValueError("Train and validation features must share the same width.")
        if train_x.shape[0] == 0:
            raise ValueError("Training data must not be empty.")
        num_class = np.unique(train_y).size
        model = XGBClassifier(num_class=num_class, **CONFIG.xgboost.gbrt_params.model_dump())
        super().__init__(
            model=model,
            model_name="xgb_ticket_classifier",
            target_names=target_names,
            exp_name=exp_name,
        )
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y

    @staticmethod
    def _normalise_dataset(
        dataset: tuple[np.ndarray, np.ndarray],
        *,
        split_name: str,
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        features, labels = dataset
        feature_arr = np.asarray(features, dtype=np.float64)
        label_arr = np.asarray(labels).reshape(-1)
        if feature_arr.ndim != 2:
            raise ValueError(f"{split_name} features must be a 2D array.")
        if feature_arr.shape[0] == 0:
            raise ValueError(f"{split_name} features must contain at least one sample.")
        if label_arr.size == 0:
            raise ValueError(f"{split_name} labels must not be empty.")
        if feature_arr.shape[0] != label_arr.shape[0]:
            raise ValueError(f"{split_name} features and labels must align by row.")
        return feature_arr, label_arr.astype(np.int64, copy=False)

    def train(self) -> XGBClassifier:
        """Fit the XGBoost classifier using the configured training regime."""

        self._run_grid_search_if_enabled()
        self.model.fit(
            self.train_x,
            self.train_y,
            eval_set=[(self.val_x, self.val_y)],
            verbose=False,
        )
        validation_predictions = self.predict(self.val_x)
        self._record_validation_report(y_true=self.val_y, y_pred=validation_predictions)
        self._log_training_artifacts(
            params=self.model.get_params(),
            model_logger=self._log_model_artifact,
        )
        return self.model

    def predict(
        self,
        x: NDArray[np.float64],
        *,
        return_proba: bool = False,
    ) -> NDArray[np.float64] | NDArray[np.int_]:
        """Run inference on the supplied feature matrix."""

        if not hasattr(self.model, "classes_"):
            raise RuntimeError("Model has not been trained. Call `train()` before `predict()`.")
        if return_proba:
            return self.model.predict_proba(x)
        return self.model.predict(x)

    def _run_grid_search_if_enabled(self) -> None:
        """Execute an optional hyper-parameter sweep backed by scikit-learn."""

        grid_cfg = CONFIG.xgboost.grid_search
        if not grid_cfg.enabled:
            return

        param_grid = dict(grid_cfg.param_grid)
        XGB_LOGGER.info(
            "Starting XGBoost grid search across %d hyperparameters.",
            len(param_grid),
        )
        sample_cap = int(grid_cfg.sample_cap)
        if 0 < sample_cap < self.train_x.shape[0]:
            rng = np.random.default_rng(CONFIG.seed)
            sample_indices = rng.choice(self.train_x.shape[0], size=sample_cap, replace=False)
        else:
            sample_indices = np.arange(self.train_x.shape[0])

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            scoring=grid_cfg.scoring,
            cv=grid_cfg.cv,
            n_jobs=grid_cfg.n_jobs,
            verbose=grid_cfg.verbose,
            refit=grid_cfg.refit,
        )
        grid_search.fit(
            self.train_x[sample_indices],
            self.train_y[sample_indices],
            eval_set=[(self.val_x, self.val_y)],
            verbose=False,
        )
        XGB_LOGGER.info(
            "Grid search complete | best_score=%.4f params=%s",
            float(grid_search.best_score_),
            grid_search.best_params_,
        )
        self.model.set_params(**grid_search.best_params_)

    def _log_model_artifact(self) -> None:
        """Persist the trained XGBoost model to MLflow."""

        mlflow.xgboost.log_model(self.model, name="model")


class XGBTicketClassifer(XGBTrainer):
    """Backward-compatible alias for legacy usage sites."""

    def __init__(
        self,
        train_set: tuple[np.ndarray, np.ndarray],
        val_set: tuple[np.ndarray, np.ndarray],
        *,
        target_names: Sequence[str] | None = None,
        exp_name: str = "xgb_ticket_classifier",
    ) -> None:
        super().__init__(
            train_set=train_set,
            val_set=val_set,
            target_names=target_names,
            exp_name=exp_name,
        )


__all__ = [
    "XGBTicketClassifer",
    "XGBTrainer",
]
