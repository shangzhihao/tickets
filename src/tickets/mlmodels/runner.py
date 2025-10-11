from ..schemas.tasks import Task
from .xgboost_model import train_xgboost_models


def runner(task: Task) -> None:
    if task == Task.MODEL_XG_TRAIN:
        train_xgboost_models()
