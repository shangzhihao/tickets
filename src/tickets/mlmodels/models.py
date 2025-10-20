"""Model training utilities for ticket classification tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

import mlflow
import mlflow.pytorch
import mlflow.xgboost
import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from omegaconf import OmegaConf
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBClassifier

from tickets.mlmodels.dataset import TicketDataSet, chronological_split
from tickets.mlmodels.evaluate import ResultReport
from tickets.mlmodels.transformer import CategoriesTransformer
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
    metrics: ResultReport


TModel = TypeVar("TModel")


class ModelTrainer(Generic[TModel], ABC):
    """Abstract base providing shared helpers for model trainers."""

    def __init__(
        self,
        *,
        model: TModel,
        model_name: Literal["dnn_ticket_classifier", "xgb_ticket_classifier"],
        target_names: Sequence[str] | None,
        exp_name: str,
    ) -> None:
        self.model = model
        self.model_name = model_name
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
        model = XGBClassifier(num_class=num_class, **CONFIG.xgboost.gbrt_params)
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

        param_grid = OmegaConf.to_container(grid_cfg.param_grid, resolve=True)
        ML_LOGGER.bind(model=self.model_name).info(
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
        ML_LOGGER.bind(model=self.model_name).info(
            "Grid search complete | best_score=%.4f params=%s",
            float(grid_search.best_score_),
            grid_search.best_params_,
        )
        self.model.set_params(**grid_search.best_params_)

    def _log_model_artifact(self) -> None:
        """Persist the trained XGBoost model to MLflow."""

        mlflow.xgboost.log_model(self.model, artifact_path="model")


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


class DNNTicketClassifier(torch.nn.Module):
    """Feed-forward neural network for ticket classification."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_dim, out_features=CONFIG.dnn.hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=CONFIG.dnn.dropout),
            torch.nn.Linear(in_features=CONFIG.dnn.hidden, out_features=out_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute logits for the provided feature batch."""

        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        return self.layers(inputs)


class DNNTrainer(ModelTrainer[DNNTicketClassifier]):
    """Pure-PyTorch trainer handling optimisation and validation loops."""

    def __init__(
        self,
        model: DNNTicketClassifier,
        train_set: Dataset,
        val_set: Dataset,
        *,
        target_names: Sequence[str] | None = None,
        exp_name: str = "dnn_ticket_classifier",
    ) -> None:
        self.device = torch.device(CONFIG.dnn.device)
        model = model.to(self.device)
        super().__init__(
            model=model,
            model_name="dnn_ticket_classifier",
            target_names=target_names,
            exp_name=exp_name,
        )
        self.train_loader = DataLoader(
            train_set,
            batch_size=int(CONFIG.dnn.batch_size),
            shuffle=True,
            num_workers=int(CONFIG.dnn.dl_num_worker),
        )
        self.val_loader = DataLoader(
            val_set,
            batch_size=int(CONFIG.dnn.batch_size),
            shuffle=False,
            num_workers=int(CONFIG.dnn.dl_num_worker),
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=float(CONFIG.dnn.lr))
        self.max_epochs = int(CONFIG.dnn.epoch)
        self.patience = int(CONFIG.dnn.patience)
        self.best_state: dict[str, torch.Tensor] | None = None

    def train(self) -> DNNTicketClassifier:
        """Train the classifier with early stopping on validation loss."""

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        logger = ML_LOGGER.bind(model=self.model_name)

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_epoch()
            val_loss, val_accuracy = self._evaluate()

            logger.info(
                "Epoch %d | train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                epoch,
                train_loss,
                val_loss,
                val_accuracy,
            )

            if val_loss + 1e-6 < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                self.best_state = deepcopy(self.model.state_dict())
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    logger.info(
                        "Early stopping triggered after %d epochs without improvement.",
                        epochs_without_improvement,
                    )
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self._build_validation_report()
        self._log_training_artifacts(
            params=self._collect_hyperparameters(),
            model_logger=self._log_model_artifact,
        )
        return self.model

    def predict(
        self,
        features: torch.Tensor,
        *,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """Run inference on a mini-batch of features."""

        self.model.eval()
        with torch.inference_mode():
            logits = self.model(features.to(self.device))
        if return_logits:
            return logits
        return torch.argmax(logits, dim=1)

    def _train_epoch(self) -> float:
        """Run a single optimisation epoch."""

        self.model.train()
        cumulative_loss = 0.0
        total_samples = 0

        for features, labels in self.train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(features)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            batch_size = features.size(0)
            cumulative_loss += loss.item() * batch_size
            total_samples += batch_size

        if total_samples == 0:
            raise RuntimeError("No samples processed during training epoch.")
        return cumulative_loss / total_samples

    @torch.no_grad()
    def _evaluate(self) -> tuple[float, float]:
        """Evaluate the model on the validation set."""

        self.model.eval()
        cumulative_loss = 0.0
        total_samples = 0
        correct_predictions = 0

        for features, labels in self.val_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(features)
            loss = self.criterion(logits, labels)

            batch_size = features.size(0)
            cumulative_loss += loss.item() * batch_size
            total_samples += batch_size
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()

        if total_samples == 0:
            raise RuntimeError("No samples processed during validation epoch.")

        return cumulative_loss / total_samples, correct_predictions / total_samples

    @torch.no_grad()
    def _build_validation_report(self) -> ResultReport:
        """Generate a structured validation report on the validation dataset."""

        targets: list[int] = []
        predictions: list[int] = []
        self.model.eval()

        for features, labels in self.val_loader:
            if features.numel() == 0:
                continue
            preds = self.predict(features)
            predictions.extend(preds.cpu().tolist())
            targets.extend(labels.cpu().tolist())

        if not targets:
            raise RuntimeError("Validation report cannot be produced without samples.")
        return self._record_validation_report(y_true=targets, y_pred=predictions)

    def _collect_hyperparameters(self) -> dict[str, Any]:
        """Assemble the DNN hyper-parameters for MLflow logging."""

        return {
            "batch_size": int(CONFIG.dnn.batch_size),
            "dropout": float(CONFIG.dnn.dropout),
            "hidden": int(CONFIG.dnn.hidden),
            "learning_rate": float(CONFIG.dnn.lr),
            "max_epochs": int(self.max_epochs),
            "patience": int(self.patience),
            "device": str(CONFIG.dnn.device),
            "dl_num_worker": int(CONFIG.dnn.dl_num_worker),
        }

    def _log_model_artifact(self) -> None:
        """Persist the trained DNN model to MLflow."""

        model_copy = deepcopy(self.model)
        model_copy = model_copy.to("cpu")
        mlflow.pytorch.log_model(model=model_copy, artifact_path="model")


def main(
    *,
    enable_xgb: bool,
    enable_dnn: bool,
) -> tuple[ResultReport | None, ResultReport | None]:
    """Execute the configured training pipelines and return their validation reports."""

    if not enable_xgb and not enable_dnn:
        raise ValueError("At least one trainer must be enabled.")

    tickets = load_df_from_s3(CONFIG.data.online_file, group=__file__)
    splits = chronological_split(tickets)
    train_data = splits.train
    val_data = splits.validation
    target_col = "category"
    cat_transformer = CategoriesTransformer(train_data)
    train_tickets = TicketDataSet(
        df=train_data,
        target_col=target_col,
        cat_transformer=cat_transformer,
    )
    val_tickets = TicketDataSet(
        df=val_data,
        target_col=target_col,
        cat_transformer=cat_transformer,
    )
    resolved_target_values = tuple(map(str, cat_transformer.col_value_map.get(target_col, [])))
    target_names: Sequence[str] | None = resolved_target_values if resolved_target_values else None

    xgb_report: ResultReport | None = None
    dnn_report: ResultReport | None = None

    if enable_xgb:
        xgb_train_set = train_tickets.get_xgb_dataset()
        xgb_val_set = val_tickets.get_xgb_dataset()
        xgb_trainer = XGBTicketClassifer(
            xgb_train_set,
            xgb_val_set,
            target_names=target_names,
        )
        xgb_trainer.train()
        xgb_report = xgb_trainer.validation_report_
        if xgb_report is not None:
            print(xgb_report)

    if enable_dnn:
        dnn_train_ds = train_tickets.get_torch_dataset()
        dnn_val_ds = val_tickets.get_torch_dataset()
        if len(dnn_train_ds) == 0:
            raise RuntimeError("Training dataset is empty.")

        feature_sample = dnn_train_ds[0][0]
        feature_dim = feature_sample.shape[0]
        num_classes = cat_transformer.get_num_class(target_col)
        classifier = DNNTicketClassifier(in_dim=feature_dim, out_dim=num_classes)
        dnn_trainer = DNNTrainer(
            model=classifier,
            train_set=dnn_train_ds,
            val_set=dnn_val_ds,
            target_names=target_names,
        )
        dnn_trainer.train()
        dnn_report = dnn_trainer.validation_report_
        if dnn_report is not None:
            print(dnn_report)

    return xgb_report, dnn_report


if __name__ == "__main__":
    main(enable_xgb=True, enable_dnn=True)
