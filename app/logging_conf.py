"""Application logging for the ProdBuilder website.

Configurable at runtime between INFO (verbose) and ERROR (quiet). Logs go to
both the console and a rotating file under the data directory so they can be
inspected and cleaned up.
"""
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

from .config import DATA_DIR

LOG_FILE = DATA_DIR / "prodbuilder.log"
VALID_LEVELS = ("INFO", "ERROR")

logger = logging.getLogger("prodbuilder")
_configured = False
_current_level = "INFO"


def configure(level: str | None = None) -> None:
    global _configured, _current_level
    level = (level or os.environ.get("PRODBUILDER_LOG_LEVEL", "INFO")).upper()
    if level not in VALID_LEVELS:
        level = "INFO"
    _current_level = level

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-5s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    if not _configured:
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)
        try:
            fileh = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000,
                                        backupCount=3, encoding="utf-8")
            fileh.setFormatter(fmt)
            logger.addHandler(fileh)
        except OSError:
            pass
        logger.propagate = False
        _configured = True

    logger.setLevel(getattr(logging, level))
    # Align uvicorn's own loggers: at ERROR we silence access logs.
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).setLevel(getattr(logging, level))
    logging.getLogger("uvicorn.access").setLevel(
        logging.WARNING if level == "ERROR" else logging.INFO)


def set_level(level: str) -> str:
    configure(level)
    logger.info("Log level set to %s", _current_level)
    return _current_level


def current_level() -> str:
    return _current_level


def clear_log_file() -> bool:
    """Truncate the website log file (and rotated parts). Returns True on success."""
    ok = True
    for path in [LOG_FILE, *(DATA_DIR.glob("prodbuilder.log.*"))]:
        try:
            if path.exists():
                path.write_text("", encoding="utf-8")
        except OSError:
            ok = False
    logger.info("Website log file cleared")
    return ok
