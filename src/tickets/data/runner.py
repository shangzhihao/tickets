from enum import StrEnum, auto

from omegaconf import DictConfig

from .ingest import ingest


class Task(StrEnum):
    INGEST = auto()


def runner(cfg: DictConfig, task: Task):
    if task == Task.INGEST:
         ingest(cfg)