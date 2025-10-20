"""PyTorch trainer implementation for ticket classification."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import mlflow
import torch
from torch.utils.data import DataLoader, Dataset

from tickets.mlmodels.base import ModelTrainer
from tickets.mlmodels.evaluate import ResultReport
from tickets.utils.config_util import CONFIG
from tickets.utils.log_util import ML_LOGGER


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
        mlflow.pytorch.log_model(pytorch_model=model_copy, artifact_path="model")


__all__ = [
    "DNNTrainer",
    "DNNTicketClassifier",
]
