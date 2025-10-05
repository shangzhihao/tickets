# src/ml_project/train.py
import hydra
from omegaconf import DictConfig, OmegaConf

from .data.runner import Task, runner


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.set_readonly(cfg, True)
    runner(cfg, Task.INGEST)

if __name__ == "__main__":
    main()

