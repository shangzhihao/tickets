from omegaconf import DictConfig
from .ingest import ingest


def run_ingest(cfg: DictConfig):
    ingest(cfg)