"""Model training utilities for ticket classification tasks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
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


class XGBTicketClassifer:
    def __init__(
        self, train_set: tuple[np.ndarray, np.ndarray], val_set: tuple[np.ndarray, np.ndarray]
    ) -> None:
        self.train_x = train_set[0].copy()
        self.train_y = train_set[1].copy()
        self.val_x = val_set[0].copy()
        self.val_y = val_set[1].copy()
        self.validation_report_: ResultReport | None = None
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
                self.train_y[sample_indices],
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
        report = ResultReport.from_predictions(
            model_name="xgb_ticket_classifier",
            y_true=self.val_y,
            y_pred=validation_predictions,
        )
        ML_LOGGER.info("Validation macro F1: %.4f", report.macro_f1)
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


class DNNTrainer:
    """Pure-PyTorch trainer handling optimisation and validation loops."""

    def __init__(
        self,
        model: DNNTicketClassifier,
        train_set: Dataset,
        val_set: Dataset,
    ) -> None:
        self.device = torch.device(CONFIG.dnn.device)
        self.model = model.to(self.device)
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
        self.validation_report_: ResultReport | None = None

    def train(self) -> DNNTicketClassifier:
        """Train the classifier with early stopping on validation loss."""

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_epoch()
            val_loss, val_accuracy = self._evaluate()

            ML_LOGGER.info(
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
                    ML_LOGGER.info(
                        "Early stopping triggered after %d epochs without improvement.",
                        epochs_without_improvement,
                    )
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self.validation_report_ = self._build_validation_report()
        ML_LOGGER.info(
            "Validation macro F1 after training: %.4f",
            self.validation_report_.macro_f1,
        )
        return self.model

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
            logits = self.model(features.to(self.device))
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().tolist())
            targets.extend(labels.cpu().tolist())

        if not targets:
            raise RuntimeError("Validation report cannot be produced without samples.")
        return ResultReport.from_predictions(
            model_name="dnn_ticket_classifier",
            y_true=targets,
            y_pred=predictions,
        )


def main_xgb() -> None:
    tickets = load_df_from_s3(CONFIG.data.online_file, group=__file__)
    splits = chronological_split(tickets)

    train_data = splits.train
    val_data = splits.validation
    _ = splits.test
    cat_transformer = CategoriesTransformer(train_data)
    train_tickets = TicketDataSet(
        df=train_data, target_col="category", cat_transformer=cat_transformer
    )
    val_tickets = TicketDataSet(df=val_data, target_col="category", cat_transformer=cat_transformer)
    # train_set[0].shape is (7000, 219), train_set[1].shape is (7000, 0)
    train_set = train_tickets.get_xgb_dataset()
    # val_set[0].shape is (1500, 219), val_set[1].shape is (1500, 0)
    val_set = val_tickets.get_xgb_dataset()
    xgb = XGBTicketClassifer(train_set, val_set)
    xgb.train()
    if xgb.validation_report_ is not None:
        print(xgb.validation_report_.to_dict())


def main_dnn() -> None:
    tickets = load_df_from_s3(CONFIG.data.online_file, group=__file__)
    splits = chronological_split(tickets)
    target_col = "category"

    train_data = splits.train
    val_data = splits.validation
    _ = splits.test
    cat_transformer = CategoriesTransformer(train_data)
    train_tickets = TicketDataSet(
        df=train_data, target_col=target_col, cat_transformer=cat_transformer
    )
    val_tickets = TicketDataSet(df=val_data, target_col=target_col, cat_transformer=cat_transformer)
    train_set = train_tickets.get_torch_dataset()
    val_set = val_tickets.get_torch_dataset()

    feature_dim = train_set[0][0].shape[0]
    num_classes = cat_transformer.get_num_class(target_col)
    classifier = DNNTicketClassifier(in_dim=feature_dim, out_dim=num_classes)
    trainer = DNNTrainer(model=classifier, train_set=train_set, val_set=val_set)
    trainer.train()
    if trainer.validation_report_ is not None:
        print(trainer.validation_report_.to_dict())


if __name__ == "__main__":
    main_xgb()
    main_dnn()
