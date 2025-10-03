# src/ml_project/train.py
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(cfg.seed)
    print(cfg.project_name)

if __name__ == "__main__":
    main()
