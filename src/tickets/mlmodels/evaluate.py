"""Evaluation helpers shared across ticket classifiers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import classification_report

ModelName = Literal["dnn_ticket_classifier", "xgb_ticket_classifier"]


def _as_1d_array(values: ArrayLike) -> np.ndarray:
    """Convert input into a 1D numpy array."""

    array = np.asarray(values)
    if array.ndim == 0:
        array = np.reshape(array, (1,))
    if array.ndim > 1:
        array = np.squeeze(array)
    return array


@dataclass(frozen=True)
class ResultReport:
    """Structured wrapper around classification metrics."""

    model_name: ModelName
    per_label: dict[str, dict[str, float]]
    overall: dict[str, float]
    target_names: tuple[str, ...] | None = None

    @classmethod
    def from_predictions(
        cls,
        *,
        model_name: ModelName,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        target_names: Sequence[str] | None = None,
    ) -> ResultReport:
        """Build a result report given predictions and ground truth labels."""

        true_arr = _as_1d_array(y_true)
        pred_arr = _as_1d_array(y_pred)
        if true_arr.size == 0:
            raise ValueError("Unable to build report without ground truth labels.")
        if pred_arr.size == 0:
            raise ValueError("Unable to build report without predictions.")
        if true_arr.shape != pred_arr.shape:
            raise ValueError("Ground truth and predictions must share the same shape.")

        unique_labels = np.unique(np.concatenate((true_arr, pred_arr)))
        label_list = unique_labels.tolist()

        resolved_target_names: tuple[str, ...] | None = None
        if target_names is not None:
            if len(target_names) != len(label_list):
                raise ValueError("target_names length must match the number of unique labels.")
            resolved_target_names = tuple(target_names)
        report = classification_report(
            true_arr,
            pred_arr,
            labels=label_list,
            target_names=resolved_target_names,
            output_dict=True,
            zero_division=0,
        )

        per_label: dict[str, dict[str, float]] = {}
        overall: dict[str, float] = {}

        for label, metrics in report.items():
            if isinstance(metrics, dict):
                per_label[label] = {
                    metric_name: float(metric_value)
                    for metric_name, metric_value in metrics.items()
                }
            else:
                overall[label] = float(metrics)

        return cls(
            model_name=model_name,
            per_label=per_label,
            overall=overall,
            target_names=resolved_target_names,
        )

    @property
    def macro_f1(self) -> float:
        """Return the macro-averaged F1 score."""

        macro_metrics = self.per_label.get("macro avg")
        if macro_metrics is None:
            raise KeyError("Macro average metrics are not available in this report.")
        return macro_metrics["f1-score"]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to a JSON-serializable dictionary."""

        payload: dict[str, Any] = {
            "model_name": self.model_name,
            "per_label": self.per_label,
            "overall": self.overall,
        }
        if self.target_names is not None:
            payload["target_names"] = self.target_names
        return payload
