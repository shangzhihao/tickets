from loguru import logger

logger.remove()
logger.add(
    "logs/app.log",
    serialize=True,
    enqueue=True,
    rotation="10 MB",
    retention="7 days",
    compression="zip"
)
logger.add(lambda msg: print(msg, end=""))
# Export the logger for reuse
__all__ = ["logger"]
