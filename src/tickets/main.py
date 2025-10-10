from .data.runner import runner
from .schemas.tasks import Task


def main() -> None:
    tasks = [Task.DATA_INGEST, Task.DATA_CHECK, Task.DATA_ANALYZE]
    for task in tasks:
        runner(task)


if __name__ == "__main__":
    main()
