import logging
import sys

from rich.console import Console
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler
from rich.markup import escape as rich_escape
from tqdm import tqdm


class TqdmStream:
    """File-like shim so rich writes through tqdm.write and never smears a bar.

    isatty/fileno are delegated so rich still sees the real terminal and keeps its
    color and width detection.
    """

    def write(self, text: str):
        tqdm.write(text, file=sys.stdout, end="")

    def flush(self):
        sys.stdout.flush()

    def isatty(self) -> bool:
        return sys.stdout.isatty()

    def fileno(self) -> int:
        return sys.stdout.fileno()


class MarkupFormatter(logging.Formatter):
    """Escape rich markup in every message except slideflow's, which embeds it."""

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if not record.name.startswith("slideflow"):
            msg = rich_escape(msg)
        return msg


def setup_logging(logger=None):
    if logger is None:
        logger = logging.getLogger("embed_wsi")
    handler = RichHandler(
        console=Console(file=TqdmStream()),  # type: ignore
        markup=True,
        highlighter=NullHighlighter(),
        show_path=False,
        log_time_format="[%Y-%m-%d %H:%M:%S]",
        rich_tracebacks=True,
    )
    handler.setFormatter(MarkupFormatter("%(name)s: %(message)s"))
    logging.basicConfig(level=logging.WARNING, handlers=[handler])
    logger.setLevel(logging.INFO)
    return logger
