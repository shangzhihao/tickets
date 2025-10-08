from enum import StrEnum, auto


class Task(StrEnum):
    INGEST = auto()
    ANALYZE = auto()
    CHECK = auto()
