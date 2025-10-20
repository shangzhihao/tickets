"""Shared abstractions for machine-learning model trainers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import mlflow
from numpy.typing import ArrayLike
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from tickets.mlmodels.evaluate import ModelName, ResultReport
from tickets.utils.config_util import CONFIG
from tickets.utils.log_util import ML_LOGGER


@dataclass(frozen=True)
class ModelResult:
    """Container holding an individual target model and its evaluation metrics."""

    target: str
    pipeline: Pipeline
    label_encoder: LabelEncoder
    classes: tuple[str, ...]
    metrics: ResultReport


TModel = TypeVar("TModel")


class ModelTrainer(Generic[TModel], ABC):
    """Abstract base providing shared helpers for model trainers."""

    def __init__(
        self,
        *,
        model: TModel,
        model_name: ModelName,
        target_names: Sequence[str] | None,
        exp_name: str,
    ) -> None:
        self.model = model
        self.model_name: ModelName = model_name
        self.validation_report_: ResultReport | None = None
        self._target_names = tuple(target_names) if target_names is not None else None
        self.exp_name = exp_name

    def _build_tracking_uri(self) -> str:
        """Construct the MLflow tracking URI from configuration."""

        mlflow_cfg = CONFIG.mlflow
        host = getattr(mlflow_cfg, "host", None)
        port = getattr(mlflow_cfg, "port", None)
        if not host or not port:
            raise RuntimeError("MLflow host or port is missing in configuration.")
        return f"http://{host}:{port}"

    def _log_training_artifacts(
        self,
        *,
        params: dict[str, Any],
        model_logger: Callable[[], None],
    ) -> None:
        """Persist the hyper-parameters and trained model into MLflow."""

        tracking_uri = self._build_tracking_uri()
        logger = ML_LOGGER.bind(model=self.model_name, experiment=self.exp_name)
        try:
            mlflow.set_tracking_uri(tracking_uri)
            experiment = mlflow.set_experiment(self.exp_name)
            experiment_id = experiment.experiment_id if experiment is not None else None
            with mlflow.start_run(
                run_name=self.model_name,
                experiment_id=experiment_id,
            ):
                if params:
                    mlflow.log_params(params)
                model_logger()
            logger.info("Logged training artifacts to MLflow | tracking_uri=%s", tracking_uri)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to log training artifacts to MLflow: %s", exc)

    @abstractmethod
    def train(self) -> TModel:
        """Fit the underlying estimator."""

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Run inference with the fitted estimator."""

    def _record_validation_report(self, *, y_true: ArrayLike, y_pred: ArrayLike) -> ResultReport:
        """Persist and emit the validation report for downstream inspection."""

        report = ResultReport.from_predictions(
            model_name=self.model_name,
            y_true=y_true,
            y_pred=y_pred,
            target_names=self._target_names,
        )
        self.validation_report_ = report
        ML_LOGGER.bind(model=self.model_name).info("Validation macro F1: %.4f", report.macro_f1)
        return report


__all__ = [
    "ModelResult",
    "ModelTrainer",
]
