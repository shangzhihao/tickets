from tickets.mlmodels.dataset import chronological_split
from tickets.schemas.tasks import Task
from tickets.utils.config_util import CONFIG
from tickets.utils.io_util import load_df_from_s3

offline_frame = load_df_from_s3(
    data_path=CONFIG.data.offline_file,
    group=__file__,
)
splits = chronological_split(offline_frame)

training_data = splits.train
validation_data = splits.validation
test_data = splits.test


def runner(task: Task) -> None:
    pass
