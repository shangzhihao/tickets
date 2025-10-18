"""Preprocessing helpers for ticket models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from tickets.mlmodels.transformer import (
    bool_transformer,
    cats_num_transformer,
    cats_onehot_transformer,
    num_transformer,
    tfidf_text_transformer,
)
from tickets.schemas.ticket import (
    BOOL_FEATURES,
    CAT_FEATURES,
    NUM_FEATURES,
    TEXT_FEATURES,
    TEXT_LIST_FEATURES,
)
from tickets.utils.config_util import CONFIG
from tickets.utils.io_util import load_df_from_s3
from tickets.utils.log_util import ML_LOGGER

FEATURE_COLUMNS = NUM_FEATURES + BOOL_FEATURES + CAT_FEATURES + TEXT_FEATURES + TEXT_LIST_FEATURES


@dataclass(frozen=True)
class DatasetSplit:
    """Chronological train/validation/test partition."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    frame: pd.DataFrame,
    *,
    time_column: str = "created_at",
    train_ratio: float = CONFIG.split.train,
    validation_ratio: float = CONFIG.split.val,
) -> DatasetSplit:
    """Split frame chronologically into train/validation/test."""

    if frame.empty:
        raise ValueError("Input dataframe must not be empty.")

    ordered = frame.sort_values(time_column, ascending=True, kind="mergesort").reset_index(
        drop=True
    )
    total = len(ordered)
    if total < 3:
        raise ValueError("At least three records are required to produce chronological splits.")

    train_end = min(max(int(total * train_ratio), 1), total - 2)
    validation_end = min(
        max(int(total * (train_ratio + validation_ratio)), train_end + 1), total - 1
    )

    train_df = ordered.iloc[:train_end].reset_index(drop=True)
    validation_df = ordered.iloc[train_end:validation_end].reset_index(drop=True)
    test_df = ordered.iloc[validation_end:].reset_index(drop=True)

    ML_LOGGER.info(
        "Chronological split sizes | train=%d validation=%d test=%d",
        len(train_df),
        len(validation_df),
        len(test_df),
    )

    return DatasetSplit(train=train_df, validation=validation_df, test=test_df)


def validate_feature_frame(
    frame: pd.DataFrame,
    *,
    target_col: str,
    feature_cols: Sequence[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Check feature frame for required columns."""

    if frame.empty:
        raise ValueError("Input dataframe must not be empty.")
    missing_features = [feature for feature in feature_cols if feature not in frame.columns]
    if missing_features:
        raise ValueError(f"Missing required features in data: {missing_features}")
    if target_col not in frame.columns:
        raise ValueError(f"Data must include target column `{target_col}`.")
    return frame


class TorchDataSet(Dataset):
    def __init__(self, tickets: TicketDataSet) -> None:
        self.tickets = tickets

    def __len__(self) -> int:
        return self.tickets.target_num_arr.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(np.hstack(self.tickets.feature_arrs)[idx, :], dtype=torch.float32)
        y = torch.tensor(self.tickets.target_num_arr[idx][0], dtype=torch.long)
        return x, y


class TicketDataSet:
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        *,
        feature_cols: list[str] = FEATURE_COLUMNS,
    ) -> None:
        self.df = validate_feature_frame(df, target_col=target_col, feature_cols=feature_cols)
        self.target_col = target_col
        self.feature_cols = feature_cols
        self._transform_features()

    def _transform_features(self) -> None:
        self.txt_cols = [col for col in self.feature_cols if col in TEXT_FEATURES]
        self.txt_list_cols = [col for col in self.feature_cols if col in TEXT_LIST_FEATURES]
        self.bool_cols = [col for col in self.feature_cols if col in BOOL_FEATURES]
        self.num_cols = [col for col in self.feature_cols if col in NUM_FEATURES]
        self.cat_cols = [col for col in self.feature_cols if col in CAT_FEATURES]

        self.text_arr = tfidf_text_transformer(self.df[self.txt_cols + self.txt_list_cols])
        self.bool_arr = bool_transformer(self.df[self.bool_cols])
        self.num_arr = num_transformer(self.df[self.num_cols])
        self.cat_onehot_arr = cats_onehot_transformer(self.df[self.cat_cols])
        self.cat_num_arr = cats_num_transformer(self.df[self.cat_cols])
        self.feature_arrs = [
            self.text_arr,
            self.bool_arr,
            self.num_arr,
            self.cat_onehot_arr,
        ]

        self.target_onehot_arr = cats_onehot_transformer(self.df[[self.target_col]])
        self.target_num_arr = cats_num_transformer(self.df[[self.target_col]])

    def get_xgb_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        return np.hstack(self.feature_arrs), self.target_num_arr

    def get_torch_dataset(self) -> TorchDataSet:
        return TorchDataSet(self)


if __name__ == "__main__":
    online_df = load_df_from_s3(data_path=CONFIG.data.online_file, group=__file__)
    tickets = TicketDataSet(df=online_df, target_col="category")
    xbg_data = tickets.get_xgb_dataset()
    torch_data = tickets.get_torch_dataset()
    pass
