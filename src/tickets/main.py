from .data.runner import runner
from .schemas.tasks import Task


def main() -> None:
    tasks = [Task.INGEST, Task.CHECK, Task.ANALYZE]
    for task in tasks:
        runner(task)


if __name__ == "__main__":
    main()
