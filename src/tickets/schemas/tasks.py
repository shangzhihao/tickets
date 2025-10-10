from enum import StrEnum, auto


class Task(StrEnum):
    DATA_INGEST = auto()
    DATA_INGEST_BRONZE = auto()
    DATA_INGEST_ONLINE = auto()
    DATA_INGEST_OFFINE = auto()

    DATA_ANALYZE = auto()

    DATA_CHECK = auto()
    DATA_CHECK_BUNIESS = auto()
    DATA_CHECK_REPORT = auto()
    DATA_CHECK_CLEAN = auto()
    DATA_CHECK_SCHEMA = auto()
    DATA_CHECK_TIMING = auto()
