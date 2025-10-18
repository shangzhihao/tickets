from enum import StrEnum, auto


class Task(StrEnum):
    DATA_INGEST = auto()
    DATA_ANALYZE = auto()
    DATA_CHECK = auto()

    MODEL_XG_TRAIN_CAT = auto()
    MODEL_XG_TRAIN_SUB = auto()
    MODEL_XG_TRAIN_SENT = auto()
