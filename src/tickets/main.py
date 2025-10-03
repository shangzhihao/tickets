# src/ml_project/train.py
import hydra
from omegaconf import DictConfig
from .utils.logger import logger
@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logger.info(cfg.seed)
    logger.info(cfg.project_name)

if __name__ == "__main__":
    main()
