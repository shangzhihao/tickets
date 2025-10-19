from tickets.mlmodels.dataset import TicketDataSet, chronological_split
from tickets.mlmodels.models import XGBTicketClassifer
from tickets.mlmodels.transformer import CategoriesTransformer
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
    cat_transformer = CategoriesTransformer(train_data)
    train_set = TicketDataSet(df=train_data, target_col="category", cat_transformer=cat_transformer)
    val_set = TicketDataSet(df=val_data, target_col="category", cat_transformer=cat_transformer)

    xgb = XGBTicketClassifer(train_set.get_xgb_dataset(), val_set.get_xgb_dataset())
    xgb.train()
    print(xgb.validation_report_)
