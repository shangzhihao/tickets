from tickets.data.runner import runner as data_runner
from tickets.schemas.tasks import Task


def main() -> None:
    data_tasks = [Task.DATA_INGEST, Task.DATA_CHECK, Task.DATA_ANALYZE]
    for task in data_tasks:
        data_runner(task)


if __name__ == "__main__":
    main()
