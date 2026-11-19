"""Structured JSON-lines logging with a redaction filter (plan §2.1 "Logging", §11.8).

All CLI commands log JSON lines to stderr and to ``<run dir>/run.log``. Secrets and credential-like
literals are redacted before a record is emitted; record-level data values are never logged.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

LOGGER_NAME = "harness"

REDACTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|client[_-]?secret)\b\s*[:=]\s*[^\s,;\"']+"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bdapi[0-9a-f]{32}\b"),                      # Databricks personal access token shape
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                      # AWS access key id shape
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"),
)

_STANDARD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


def redact(text: str) -> tuple[str, int]:
    count = 0
    for pattern in REDACTION_PATTERNS:
        text, n = pattern.subn("[REDACTED]", text)
        count += n
    return text, count


def _redact_value(value: Any) -> tuple[Any, int]:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        total, out = 0, {}
        for key, item in value.items():
            out[key], n = _redact_value(item)
            total += n
        return out, total
    if isinstance(value, list | tuple):
        total, items = 0, []
        for item in value:
            red, n = _redact_value(item)
            items.append(red)
            total += n
        return items, total
    return value, 0


class JsonFormatter(logging.Formatter):
    def __init__(self, context: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.context = context or {}

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": dt.datetime.fromtimestamp(record.created, tz=dt.UTC).isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            ),
            "level": record.levelname,
            "event": record.getMessage(),
            **self.context,
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        payload, count = _redact_value(payload)
        payload["redactions_applied"] = count
        return json.dumps(payload, default=str, ensure_ascii=False)


def setup_logging(level: str = "INFO", *, run_log: Path | None = None, context: dict[str, Any] | None = None,
                  ) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(level)
    logger.propagate = False
    formatter = JsonFormatter(context)
    stderr = logging.StreamHandler(sys.stderr)
    stderr.setFormatter(formatter)
    logger.addHandler(stderr)
    if run_log is not None:
        run_log.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(run_log, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)
