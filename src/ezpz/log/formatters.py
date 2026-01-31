"""Logging formatters for ezpz."""

from __future__ import annotations

import json
import logging


class JSONFormatter(logging.Formatter):
    """Structured JSON formatter for logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log records as JSON.

        Args:
            record: The log record to format.

        Returns:
            The JSON-encoded log entry.
        """
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "line": record.lineno,
        }
        return json.dumps(log_record)
