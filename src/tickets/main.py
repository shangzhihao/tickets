from .data.runner import runner as data_runner
from .mlmodels.runner import runner as model_runner
from .schemas.tasks import Task


def main() -> None:
    data_tasks = [Task.DATA_INGEST, Task.DATA_CHECK, Task.DATA_ANALYZE]
    data_tasks.clear()
    model_tasks = [Task.MODEL_XG_TRAIN_SUB, Task.MODEL_XG_TRAIN_CAT, Task.MODEL_XG_TRAIN_SENT]
    model_tasks = [Task.MODEL_XG_TRAIN_SUB]
    for task in data_tasks:
        data_runner(task)
    for task in model_tasks:
        model_runner(task)


if __name__ == "__main__":
    main()
