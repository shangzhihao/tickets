from prefect import flow

from ..schemas.tasks import Task
from .ingest import ingest


@flow
def runner(task: Task) -> None:
    if task == Task.DATA_INGEST:
        ingest()
    elif task == Task.DATA_ANALYZE:
        pass
    elif task == Task.DATA_CHECK:
        pass
