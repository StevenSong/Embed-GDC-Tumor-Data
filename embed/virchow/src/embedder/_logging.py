"""Plain-text logging and a JSONL stats sink.

Deliberately no `rich`. A corpus run's log is read out of a slurm `.log` file,
never a terminal, and rich's handler hard-wraps every record to the console
width -- which splits slide names mid-token (`...b9c2-0 3f3a3b4cab6.svs`) and
interleaves ANSI cursor moves with tqdm's bar rewrites. The result parses only
with bespoke regex archaeology, so records here are one line each, unwrapped and
uncolored, however long the slide name is.

Timings and counters do not go in that text at all: `StatsLog` writes one JSON
object per slide to `stats.jsonl`, which is what analysis should read.
"""

import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, TextIO

from tqdm import tqdm

LOG_FORMAT = "[%(asctime)s] %(levelname)-7s %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def progress_disabled() -> bool:
    """True when tqdm would be writing a progress bar into a file.

    A bar redrawn with `\\r` into a slurm log is thousands of duplicate lines --
    38 MB of them on the 48 h corpus run -- and buys nothing a periodic log line
    does not. Interactive runs keep their bars.
    """
    return not sys.stderr.isatty()


class TqdmStream:
    """File-like shim so log records go through `tqdm.write` and never smear a bar.

    The wrapped stream is captured at construction rather than looked up per
    write, so `contextlib.redirect_stderr` (which `compile_model` uses to catch
    inductor's raw autotune output) cannot swallow log records too.
    """

    def __init__(self, stream: TextIO):
        self._stream = stream

    def write(self, text: str):
        if progress_disabled():
            self._stream.write(text)
        else:
            tqdm.write(text, file=self._stream, end="")

    def flush(self):
        self._stream.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

    def fileno(self) -> int:
        return self._stream.fileno()


def setup_logging(logger: logging.Logger | None = None) -> logging.Logger:
    if logger is None:
        logger = logging.getLogger("embed_wsi")
    handler = logging.StreamHandler(TqdmStream(sys.stderr))  # type: ignore[arg-type]
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
    logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True)
    logger.setLevel(logging.INFO)
    return logger


class StatsLog:
    """Append-only JSONL sink, one record per slide.

    Opened in append mode and flushed per record: the corpus job is resumable and
    routinely killed by a time limit, so the stats of every slide that finished
    have to survive the kill. Writes are serialised because `main` and the
    background sampler both use one.
    """

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._fh = open(path, "a", buffering=1)

    def write(self, **record: Any):
        line = json.dumps(record, default=_jsonable, separators=(",", ":"))
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self):
        with self._lock:
            self._fh.close()

    def __enter__(self) -> "StatsLog":
        return self

    def __exit__(self, *exc):
        self.close()


def _jsonable(value: Any):
    """Last-resort encoder: numpy scalars and Paths are the ones that show up."""
    if hasattr(value, "item"):
        return value.item()
    return str(value)
