# src/ml_project/train.py

from .data.runner import Task, runner


def main()->None:
    runner(Task.INGEST)

if __name__ == "__main__":
    main()

