from prefect import flow

from ..schemas.tasks import Task
from .analyze import offline_analyzer
from .check import DataQuality
from .ingest import ingest


@flow
def runner(task: Task) -> None:
    if task == Task.INGEST:
        ingest()
    elif task == Task.ANALYZE:
        res = offline_analyzer.analyze()
        for key, item in res.to_dict().items():
            print(key, item)
        offline_analyzer.save_metrics_to_s3()
    elif task == Task.CHECK:
        dq = DataQuality()
        print(dq.gen_report())
        dq.clean()
