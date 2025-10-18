"""Shared IO utilities configured via Hydra logging defaults."""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path
from typing import Any, Final

from loguru import logger
from omegaconf import DictConfig

from tickets.utils.config_util import CONFIG

MODULES: Final[tuple[str, ...]] = ("api", "data", "ml", "event")


def _module_filter(target: str, record: dict[str, object]) -> bool:
    """Return ``True`` when the log record belongs to the target module."""

    return record["extra"].get("module") == target  # type: ignore


def _configure_module_loggers(config: DictConfig) -> dict[str, Any]:
    """Configure loguru handlers and return binders for known modules."""

    output_dir = Path(config.logging.file_sink.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console_handler = {
        "sink": sys.stdout,
        "level": config.logging.console_sink.level,
        "format": config.logging.console_sink.format,
        "colorize": config.logging.console_sink.colorize,
        "backtrace": config.logging.backtrace,
        "diagnose": config.logging.diagnose,
    }

    file_handler_defaults = {
        "level": config.logging.file_sink.level,
        "serialize": config.logging.file_sink.serialize,
        "enqueue": config.logging.file_sink.enqueue,
        "rotation": config.logging.file_sink.rotation,
        "retention": config.logging.file_sink.retention,
        "compression": config.logging.file_sink.compression,
        "backtrace": config.logging.backtrace,
        "diagnose": config.logging.diagnose,
    }

    handlers = [console_handler]

    for module in MODULES:
        handlers.append(
            {
                "sink": Path(output_dir / f"{module}.log"),
                "filter": partial(_module_filter, module),
                **file_handler_defaults,
            }
        )

    logger.remove()
    logger.configure(handlers=handlers)

    return {module: logger.bind(module=module) for module in MODULES}


module_loggers = _configure_module_loggers(CONFIG)

DATA_LOGGER = module_loggers["data"]
ML_LOGGER = module_loggers["ml"]
API_LOGGER = module_loggers["api"]
EVENT_LOGGER = module_loggers["event"]
