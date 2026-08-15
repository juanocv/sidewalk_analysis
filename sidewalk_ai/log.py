from __future__ import annotations

import json
import logging
import os
import sys

# timezone.utc rather than datetime.UTC: the latter is a 3.11+ alias for the
# same object, and it was the package's only barrier to Python 3.10 -- which
# is what Ubuntu 22.04 LTS ships.
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_LOGGER_NAME = "sidewalk_ai"
TRUTHY = {"1", "true", "yes", "on", "debug"}


class JsonFormatter(logging.Formatter):
    """Small JSON formatter for machine-readable pipeline logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        event = getattr(record, "event", None)
        if event is not None:
            payload["event"] = event
        data = getattr(record, "payload", None)
        if data is not None:
            payload["payload"] = data
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in TRUTHY


def _resolve_level(level: str | int | None, debug: bool | None) -> int:
    if debug is None:
        debug = _env_bool("SWAI_DEBUG", False)
    if debug:
        return logging.DEBUG
    if level is None:
        level = os.getenv("SWAI_LOG_LEVEL", "INFO")
    if isinstance(level, int):
        return level
    return logging.getLevelName(level.upper())


def configure_logging(
    *,
    level: str | int | None = None,
    debug: bool | None = None,
    fmt: str | None = None,
    log_file: str | Path | None = None,
    force: bool = True,
) -> None:
    """Configure package logging from CLI flags or SWAI_* environment variables."""

    resolved_level = _resolve_level(level, debug)
    fmt = (fmt or os.getenv("SWAI_LOG_FORMAT", "text")).lower()
    log_file = log_file or os.getenv("SWAI_LOG_FILE")

    if fmt == "json":
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    root = logging.getLogger(ROOT_LOGGER_NAME)
    if not force and any(not isinstance(handler, logging.NullHandler) for handler in root.handlers):
        root.setLevel(resolved_level)
        return
    if force:
        root.handlers.clear()
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(resolved_level)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def debug_enabled() -> bool:
    return _env_bool("SWAI_DEBUG", False) or logging.getLogger(ROOT_LOGGER_NAME).isEnabledFor(
        logging.DEBUG
    )


def debug_event(logger: logging.Logger, event: str, payload: dict[str, Any]) -> None:
    logger.debug(event, extra={"event": event, "payload": payload})


logging.getLogger(ROOT_LOGGER_NAME).addHandler(logging.NullHandler())
