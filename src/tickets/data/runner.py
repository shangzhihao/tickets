from prefect import flow

from tickets.data.ingest import ingest
from tickets.schemas.tasks import Task


@flow
def runner(task: Task) -> None:
    if task == Task.DATA_INGEST:
        ingest()
    elif task == Task.DATA_ANALYZE:
        pass
    elif task == Task.DATA_CHECK:
        pass
