# src/ml_project/train.py
import hydra
from omegaconf import DictConfig, OmegaConf

from .data.runner import Task, runner
from .utils.config import cfg

def main()->None:
    runner(Task.INGEST)

if __name__ == "__main__":
    main()

