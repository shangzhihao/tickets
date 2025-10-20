"""Aggregated exports and orchestration helpers for model training."""

from __future__ import annotations

from collections.abc import Sequence

from tickets.mlmodels.base import ModelResult, ModelTrainer
from tickets.mlmodels.dataset import TicketDataSet, chronological_split
from tickets.mlmodels.dnn_trainer import DNNTicketClassifier, DNNTrainer
from tickets.mlmodels.evaluate import ResultReport
from tickets.mlmodels.transformer import CategoriesTransformer
from tickets.mlmodels.xgb_trainer import XGBTicketClassifer, XGBTrainer
from tickets.utils.config_util import CONFIG
from tickets.utils.io_util import load_df_from_s3


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


__all__ = [
    "DNNTrainer",
    "DNNTicketClassifier",
    "ModelResult",
    "ModelTrainer",
    "XGBTicketClassifer",
    "XGBTrainer",
    "main",
]

if __name__ == "__main__":
    main(enable_dnn=True, enable_xgb=True)
