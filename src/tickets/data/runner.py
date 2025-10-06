
from ..schemas.tasks import Task

from .ingest import ingest
from .analyze import offline_analyzer



def runner(task: Task):
    if task == Task.INGEST:
         ingest()
    if task == Task.ANALYZE:
         res = offline_analyzer.analyze()
         for key, item in res.to_dict().items():
              print(key, item)
         offline_analyzer.save_metrics_to_s3()


