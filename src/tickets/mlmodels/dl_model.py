"""PyTorch-backed classifier compatible with scikit-learn estimators."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from loguru import logger
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset, random_split

__all__ = ["TorchMLPClassifier"]


ActivationName = str


@dataclass(frozen=True)
class _ActivationFactory:
    """Factory for instantiating activation layers by name."""

    name: ActivationName

    def build(self) -> nn.Module:
        """Return the activation module associated with the given name."""
        activation_map: dict[ActivationName, nn.Module] = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
        }
        if self.name not in activation_map:
            valid = ", ".join(sorted(activation_map))
            raise ValueError(f"Unsupported activation '{self.name}'. Valid options: {valid}")
        return activation_map[self.name]


def _as_float_array(features: ArrayLike) -> NDArray[np.float32]:
    """Convert array-like features to a 2D float32 NumPy array."""
    array = np.asarray(features)
    if array.ndim != 2:
        raise ValueError(
            f"Expected features with shape (n_samples, n_features); got {array.shape}."
        )
    return array.astype(np.float32, copy=False)


def _encode_targets(targets: ArrayLike) -> tuple[NDArray[np.int64], NDArray[Any]]:
    """Encode arbitrary targets to integer indices and return the unique classes."""
    target_array = np.asarray(targets)
    if target_array.ndim != 1:
        raise ValueError(f"Expected targets with shape (n_samples,); got {target_array.shape}.")
    classes, encoded = np.unique(target_array, return_inverse=True)
    return encoded.astype(np.int64, copy=False), classes


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """Fully connected neural network classifier with a scikit-learn interface."""

    def __init__(
        self,
        hidden_dims: Sequence[int] = (128, 64),
        activation: ActivationName = "relu",
        dropout: float = 0.1,
        num_epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int | None = 5,
        validation_fraction: float = 0.1,
        random_state: int | None = None,
        device: str | None = None,
    ) -> None:
        self.hidden_dims = tuple(hidden_dims)
        self.activation = activation
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.device = device

    def fit(self, X: ArrayLike, y: ArrayLike) -> TorchMLPClassifier:
        """Train the neural network classifier on the provided data."""
        self._set_random_seed()
        features = _as_float_array(X)
        encoded_targets, classes = _encode_targets(y)

        self.classes_ = classes
        self.n_features_in_ = features.shape[1]
        self.device_ = self._resolve_device()

        dataset = TensorDataset(
            torch.from_numpy(features),
            torch.from_numpy(encoded_targets),
        )

        train_loader, val_loader = self._train_validation_loaders(dataset)
        self.model_ = self._build_network(self.n_features_in_, len(self.classes_))
        self.model_.to(self.device_)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.loss_history_: list[dict[str, float]] = []
        best_val_loss = float("inf")
        epochs_since_improvement = 0
        best_state: dict[str, torch.Tensor] | None = None

        logger.bind(model="TorchMLPClassifier").info(
            "Starting training for {} epochs on device {}", self.num_epochs, self.device_
        )

        for epoch in range(self.num_epochs):
            train_loss = self._train_one_epoch(train_loader, criterion, optimizer)
            metrics = {"epoch": float(epoch), "train_loss": train_loss}

            if val_loader is not None:
                val_loss = self._evaluate_loss(val_loader, criterion)
                metrics["val_loss"] = val_loss
                if val_loss + 1e-6 < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.detach().clone() for k, v in self.model_.state_dict().items()
                    }
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
            else:
                if train_loss + 1e-6 < best_val_loss:
                    best_val_loss = train_loss
                    best_state = {
                        k: v.detach().clone() for k, v in self.model_.state_dict().items()
                    }
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1

            self.loss_history_.append(metrics)
            logger.bind(model="TorchMLPClassifier", epoch=epoch).debug("Metrics {}", metrics)

            if self.patience is not None and epochs_since_improvement >= self.patience:
                logger.bind(model="TorchMLPClassifier").info(
                    "Early stopping triggered after {} epochs without improvement.",
                    epochs_since_improvement,
                )
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.is_fitted_ = True
        return self

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Predict class labels for the provided data."""
        probabilities = self.predict_proba(X)
        predictions = probabilities.argmax(axis=1)
        return self.classes_[predictions]

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float32]:
        """Return class probabilities for the provided data."""
        check_is_fitted(self, attributes=("is_fitted_", "model_"))
        features = _as_float_array(X)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected input with {self.n_features_in_} features; received {features.shape[1]}."
            )

        self.model_.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(features).to(self.device_)
            logits = self.model_(inputs)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        return probabilities.astype(np.float32, copy=False)

    def _train_one_epoch(
        self,
        loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        criterion: nn.Module,
        optimizer: Optimizer,
    ) -> float:
        """Execute a single training epoch and return the average loss."""
        self.model_.train()
        total_loss = 0.0
        num_batches = 0
        for features, targets in loader:
            optimizer.zero_grad()
            inputs = features.to(self.device_)
            labels = targets.to(self.device_)
            predictions = self.model_(inputs)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        return total_loss / max(1, num_batches)

    def _evaluate_loss(
        self,
        loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        criterion: nn.Module,
    ) -> float:
        """Evaluate loss on a validation loader."""
        self.model_.eval()
        total_loss = 0.0
        batches = 0
        with torch.no_grad():
            for features, targets in loader:
                inputs = features.to(self.device_)
                labels = targets.to(self.device_)
                logits = self.model_(inputs)
                loss = criterion(logits, labels)
                total_loss += loss.item()
                batches += 1
        return total_loss / max(1, batches)

    def _train_validation_loaders(
        self,
        dataset: TensorDataset,
    ) -> tuple[
        DataLoader[tuple[torch.Tensor, torch.Tensor]],
        DataLoader[tuple[torch.Tensor, torch.Tensor]] | None,
    ]:
        """Create DataLoader objects for training and validation splits."""
        if len(dataset) == 0:
            raise ValueError("Cannot train on an empty dataset.")

        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None
        if 0.0 < self.validation_fraction < 1.0 and len(dataset) > 1:
            val_size = max(1, int(len(dataset) * self.validation_fraction))
            train_size = len(dataset) - val_size
            if train_size == 0:
                train_size, val_size = len(dataset), 0
            generator = torch.Generator()
            if self.random_state is not None:
                generator.manual_seed(self.random_state)
            splits = random_split(dataset, lengths=[train_size, val_size], generator=generator)
            train_dataset = splits[0]
            val_dataset = splits[1] if val_size > 0 else None
        else:
            train_dataset = dataset
            val_dataset = None

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def _build_network(self, num_features: int, num_classes: int) -> nn.Module:
        """Construct the multilayer perceptron."""
        layers: list[nn.Module] = []
        input_dim = num_features
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(_ActivationFactory(self.activation).build())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, num_classes))
        return nn.Sequential(*layers)

    def _set_random_seed(self) -> None:
        """Set random seed for reproducibility."""
        if self.random_state is None:
            return
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
