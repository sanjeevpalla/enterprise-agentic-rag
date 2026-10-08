"""Application logging: console (stdout) output plus optional rotating local log files.

Call ``setup_logging()`` once at process start (entry points only). Modules keep
using ``logging.getLogger(__name__)`` as usual.

Console output always happens, in LOG_FORMAT (``text`` or ``json``). On AWS
(ECS/Fargate, EKS, Lambda) the platform ships stdout to CloudWatch Logs, so set
``LOG_TO_FILE=false`` and ``LOG_FORMAT=json`` there: container disks are
ephemeral and Lambda's code directory is read-only.

With ``LOG_TO_FILE=true`` (local development), files are written to ``LOG_DIR``:
    app.log     every record at LOG_LEVEL and above
    error.log   WARNING and above only, for a quick look at what went wrong
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings

TEXT_FORMAT = "%(asctime)s %(levelname)-7s [%(process)d] %(name)s: %(message)s"

# Third-party loggers that are noisy at INFO.
QUIET_LOGGERS = ("httpx", "httpcore", "google_genai", "urllib3", "langfuse", "opentelemetry")

# Attributes every LogRecord has; anything else came from ``extra=`` and goes into the JSON.
_RECORD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line, including fields passed via ``logger.info(..., extra={...})``."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
        }
        entry.update({k: v for k, v in vars(record).items() if k not in _RECORD_ATTRS})
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def setup_logging(settings: Settings | None = None, level: str | None = None) -> Path | None:
    """Configure root logging. Returns the log directory, or None when file logging is off.

    Safe to call more than once: handlers from a previous call are replaced.
    """
    settings = settings or get_settings()
    log_level = logging.getLevelName((level or settings.log_level).upper())

    def formatter() -> logging.Formatter:
        return JsonFormatter() if settings.log_format == "json" else logging.Formatter(TEXT_FORMAT)

    def rotating(log_dir: Path, name: str, handler_level: int) -> logging.Handler:
        handler = logging.handlers.RotatingFileHandler(
            log_dir / name,
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count,
            encoding="utf-8",
        )
        handler.setLevel(handler_level)
        handler.setFormatter(formatter())
        return handler

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(formatter())

    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    root.setLevel(log_level)
    root.addHandler(console)

    log_dir: Path | None = None
    if settings.log_to_file:
        log_dir = settings.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        root.addHandler(rotating(log_dir, "app.log", log_level))
        root.addHandler(rotating(log_dir, "error.log", logging.WARNING))

    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.captureWarnings(True)  # route warnings.warn(...) into the logs too
    return log_dir
