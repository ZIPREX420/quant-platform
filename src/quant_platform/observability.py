"""Structured (JSON-lines) logging for quant-platform.

Every service and script calls configure_json_logging() once at startup.
One event per line; machine-parseable; UTC timestamps; extras preserved.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

_STDLIB_RECORD_FIELDS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            event["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _STDLIB_RECORD_FIELDS and not key.startswith("_"):
                try:
                    json.dumps(value)
                    event[key] = value
                except (TypeError, ValueError):
                    event[key] = repr(value)
        return json.dumps(event, ensure_ascii=False)


def configure_json_logging(level: int = logging.INFO, stream=None) -> logging.Logger:
    """Configure the root logger for JSON-lines output. Idempotent."""
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        if getattr(handler, "_quant_platform_json", False):
            return root
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    handler._quant_platform_json = True
    root.addHandler(handler)
    return root
