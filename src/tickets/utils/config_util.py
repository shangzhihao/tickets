from __future__ import annotations

import sys
from collections.abc import Sequence

from dotenv import load_dotenv
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

load_dotenv()


def gather_cli_overrides(args: Sequence[str]) -> list[str]:
    """Extract Hydra-compatible overrides from CLI arguments."""
    if not args:
        return []

    override_tokens: list[str] = []
    for token in args:
        if "=" not in token and not token.startswith(("+", "~", "hydra.")):
            continue
        override_tokens.append(token)
    return override_tokens


def load_config(
    config_dir: str,
    config_name: str,
    overrides: Sequence[str] | None = None,
) -> DictConfig:
    """Load a Hydra configuration with optional override parameters."""
    override_list = list(overrides or [])

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize(version_base=None, config_path=config_dir):
        cfg = compose(config_name=config_name, overrides=override_list)
    OmegaConf.set_readonly(cfg, True)
    return cfg


cfg = load_config(
    config_dir="../../../conf",
    config_name="config",
    overrides=gather_cli_overrides(sys.argv[1:]),
)

__all__ = ["cfg"]
