from tickets.mlmodels.dataset import TicketDataSet, chronological_split
from tickets.mlmodels.models import XGBModel
from tickets.schemas.tasks import Task
from tickets.utils.config_util import CONFIG
from tickets.utils.io_util import load_df_from_s3

offline_frame = load_df_from_s3(
    data_path=CONFIG.data.offline_file,
    group=__file__,
)
splits = chronological_split(offline_frame)

train_data = splits.train
val_data = splits.validation
test_data = splits.test


def runner(task: Task) -> None:
    train_set = TicketDataSet(df=train_data, target_col="category")
    val_set = TicketDataSet(df=val_data, target_col="category")

    tx, ty = train_set.get_xgb_dataset()
    vx, vy = val_set.get_xgb_dataset()
    xgb = XGBModel(tx, ty, vx, vy)
    xgb.train()
    print(xgb.validation_report_)
