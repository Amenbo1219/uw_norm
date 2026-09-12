"""Structured (JSON Lines) plus human-readable logging (3.2, N-06)."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

LOGGER_NAME = "uwnorm"


class JsonLinesHandler(logging.Handler):
    """One JSON object per line, for machine post-processing."""

    def __init__(self, path: str | Path):
        super().__init__()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        payload = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        self._fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            super().close()


def setup_logging(verbose: bool = False, jsonl_path: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    logger.addHandler(stream)
    if jsonl_path:
        logger.addHandler(JsonLinesHandler(jsonl_path))
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def log_event(event: str, **fields) -> None:
    """Emit a structured event that both handlers render sensibly."""
    logger = get_logger()
    human = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.info(f"{event} {human}".strip(), extra={"fields": {"event": event, **fields}})


class Timer:
    """Context manager that logs how long a stage took (N-02 evidence)."""

    def __init__(self, name: str):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._t0
        log_event("stage.done", stage=self.name, seconds=round(self.elapsed, 2))
