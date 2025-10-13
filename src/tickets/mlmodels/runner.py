from ..schemas.tasks import Task
from ..schemas.ticket import (
    Category,
)
from ..utils.config_util import cfg
from ..utils.io_util import load_df_from_s3
from .preprocessing import chronological_split
from .xgboost_model import XGBModel

offline_frame = load_df_from_s3(
    data_path=cfg.data.offline_file,
    group=__file__,
)
splits = chronological_split(offline_frame)

training_data = splits.train
validation_data = splits.validation
test_data = splits.test


def runner(task: Task) -> None:
    _training_data = training_data.copy()
    _validation_data = validation_data.copy()
    if task == Task.MODEL_XG_TRAIN_SUB:
        for cat in Category:
            _training_data = training_data[training_data["category"] == cat.value].copy()
            _validation_data = validation_data[validation_data["category"] == cat.value].copy()
            _training_data[f"{cat.value}.subcategory"] = _training_data["subcategory"]
            _validation_data[f"{cat.value}.subcategory"] = _validation_data["subcategory"]
            clf = XGBModel(
                training_data=_training_data,
                target=f"{cat.value}.subcategory",
                validation_data=_validation_data,
            )
            clf.train()
            print(clf.evaluate(_validation_data))
    elif task == Task.MODEL_XG_TRAIN_CAT:
        clf = XGBModel(
            training_data=_training_data,
            target="category",
            validation_data=_validation_data,
        )
        clf.train()
        print(clf.evaluate(_validation_data))
    elif task == Task.MODEL_XG_TRAIN_SENT:
        clf = XGBModel(
            training_data=_training_data,
            target="customer_sentiment",
            validation_data=_validation_data,
        )
        clf.train()
        print(clf.evaluate(_validation_data))
    else:
        raise ValueError(f"Unknown task {task}")
