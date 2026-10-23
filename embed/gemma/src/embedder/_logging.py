"""Plain one-line-per-record logging, as in embed/virchow.

The log is read out of a slurm `.log` file, so records stay unwrapped and
uncolored, and vLLM's own progress bars are turned off when stderr is not a
terminal -- a bar redrawn with `\\r` into a file is thousands of duplicate lines.
"""

import logging
import sys

LOG_FORMAT = "[%(asctime)s] %(levelname)-7s %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def progress_disabled() -> bool:
    return not sys.stderr.isatty()


def setup_logging() -> logging.Logger:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
    logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True)
    logger = logging.getLogger("embed_reports")
    logger.setLevel(logging.INFO)
    return logger
