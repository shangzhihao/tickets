from ..schemas.tasks import Task
from ..schemas.ticket import Category
from .xgboost_model import xgboost_cat


def runner(task: Task) -> None:
    if task == Task.MODEL_XG_TRAIN_SUB:
        for cat in Category:
            print("=" * 20)
            print("Training category:", cat)
            print("=" * 20)
            xgboost_cat(target="subcategory", category=cat)
    elif task == Task.MODEL_XG_TRAIN_CAT:
        xgboost_cat(target="category")
    elif task == Task.MODEL_XG_TRAIN_SENT:
        xgboost_cat(target="customer_sentiment")
    else:
        raise ValueError(f"Unknown task {task}")
