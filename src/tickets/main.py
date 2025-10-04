# src/ml_project/train.py
import hydra
from omegaconf import DictConfig, OmegaConf
from .utils.logger import logger
from .data.runner import run_ingest

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.set_readonly(cfg, True)
    run_ingest(cfg)

if __name__ == "__main__":
    main()

