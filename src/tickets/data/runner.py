from enum import StrEnum, auto

from .ingest import ingest


class Task(StrEnum):
    INGEST = auto()


def runner(task: Task):
    if task == Task.INGEST:
         ingest()