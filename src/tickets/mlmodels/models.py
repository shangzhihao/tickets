"""XGBoost training utilities for ticket classification tasks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from omegaconf import OmegaConf
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from tickets.mlmodels.dataset import TicketDataSet
from tickets.utils.config_util import CONFIG
from tickets.utils.io_util import load_df_from_s3
from tickets.utils.log_util import ML_LOGGER


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
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
    ) -> None:
        self.train_x = train_x.copy()
        self.train_y = train_y.copy()
        self.val_x = val_x.copy()
        self.val_y = val_y.copy()
        num_class = len(np.unique(self.train_y))
        self.model = XGBClassifier(num_class=num_class, **CONFIG.xgboost.gbrt_params)

    def train(self) -> XGBClassifier:
        """Fit the XGBoost classifier using the configured training regime."""

        if self.train_x.size == 0 or self.train_y.size == 0:
            raise ValueError("Training data must not be empty.")

        grid_cfg = CONFIG.xgboost.grid_search
        if grid_cfg.enabled:
            param_grid = OmegaConf.to_container(grid_cfg.param_grid, resolve=True)
            ML_LOGGER.info(
                "Starting XGBoost grid search across %d hyperparameters.",
                len(param_grid),
            )
            sample_cap = int(grid_cfg.sample_cap)
            if sample_cap > 0 and sample_cap < self.train_x.shape[0]:
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
                self.val_y[sample_indices],
                eval_set=[(self.val_x, self.val_y)],
                verbose=False,
            )
            ML_LOGGER.info(
                "Grid search complete | best_score=%.4f params=%s",
                grid_search.best_score_,
                grid_search.best_params_,
            )
            self.model.set_params(**grid_search.best_params_)

        self.model.fit(
            self.train_x,
            self.train_y,
            eval_set=[(self.val_x, self.val_y)],
            verbose=False,
        )

        validation_predictions = self.model.predict(self.val_x)
        report = classification_report(self.val_y, validation_predictions, output_dict=True)
        ML_LOGGER.info("Validation macro F1: %.4f", report["macro avg"]["f1-score"])
        self.validation_report_ = report
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


if __name__ == "__main__":
    tickets = load_df_from_s3(CONFIG.data.online_file, group=__file__)
    dataset = TicketDataSet(df=tickets, target_col="category")
    x, y = dataset.get_xgb_dataset()
    xgb = XGBModel(x, y, x, y)
    xgb.train()
    print(xgb.validation_report_)
