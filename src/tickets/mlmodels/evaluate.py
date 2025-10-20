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

    def __str__(self) -> str:
        """Return a human friendly, tabular representation of the report."""

        lines: list[str] = [f"ResultReport(model={self.model_name})"]

        ordered_rows: list[tuple[str, dict[str, float]]] = []
        base_rows: list[tuple[str, dict[str, float]]] = []
        summary_rows: list[tuple[str, dict[str, float]]] = []
        summary_labels = {"macro avg", "weighted avg"}

        for label, metrics in self.per_label.items():
            if label in summary_labels:
                summary_rows.append((label, metrics))
            else:
                base_rows.append((label, metrics))

        if self.target_names is not None:
            order_map = {label: idx for idx, label in enumerate(self.target_names)}
            base_rows.sort(key=lambda item: order_map.get(item[0], len(order_map)))
        else:
            base_rows.sort(key=lambda item: item[0])

        summary_rows.sort(key=lambda item: item[0])
        ordered_rows.extend(base_rows)
        ordered_rows.extend(summary_rows)

        if not ordered_rows:
            lines.append("No per-label metrics available.")
        else:
            label_width = max(len("label"), max(len(label) for label, _ in ordered_rows))
            header = (
                f"{'label':<{label_width}}  "
                f"{'precision':>9}  "
                f"{'recall':>7}  "
                f"{'f1-score':>9}  "
                f"{'support':>7}"
            )
            lines.append(header)
            lines.append("-" * len(header))

            for label, metrics in ordered_rows:
                precision = metrics.get("precision")
                recall = metrics.get("recall")
                f1 = metrics.get("f1-score")
                support = metrics.get("support")

                precision_str = f"{precision:.3f}" if precision is not None else "n/a"
                recall_str = f"{recall:.3f}" if recall is not None else "n/a"
                f1_str = f"{f1:.3f}" if f1 is not None else "n/a"
                support_str = f"{int(support):d}" if support is not None else "n/a"

                lines.append(
                    f"{label:<{label_width}}  "
                    f"{precision_str:>9}  "
                    f"{recall_str:>7}  "
                    f"{f1_str:>9}  "
                    f"{support_str:>7}"
                )

        accuracy = self.overall.get("accuracy")
        if accuracy is not None:
            lines.append("")
            lines.append(f"accuracy: {accuracy:.3f}")

        return "\n".join(lines)
