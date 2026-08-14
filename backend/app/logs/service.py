from __future__ import annotations

import logging
import re
from collections import deque
from pathlib import Path

from app.core.config import Settings
from app.logs.schemas import RuntimeLogResponse


LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}
LOG_LINE_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} (?P<level>DEBUG|INFO|WARNING|ERROR)\s")


def runtime_log_path(settings: Settings) -> Path:
    return settings.data_dir / "logs" / "backend.log"


def configure_runtime_logging(settings: Settings) -> Path:
    path = runtime_log_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_ywa_runtime_log", False):
            root.removeHandler(handler)
            handler.close()
    handler = logging.FileHandler(path, encoding="utf-8")
    handler._ywa_runtime_log = True  # type: ignore[attr-defined]
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(handler)
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)
    # Alembic's fileConfig may disable loggers that existed before a later
    # application startup. Re-enable the Starter namespace after it resets
    # logging so runtime events continue to reach this handler.
    logging.getLogger("ywa").disabled = False
    return path


def read_runtime_logs(settings: Settings, *, lines: int = 300, level: str | None = None) -> RuntimeLogResponse:
    safe_lines = min(max(lines, 1), 1000)
    normalized = (level or "").strip().upper() or None
    if normalized == "WARN":
        normalized = "WARNING"
    if normalized not in LOG_LEVELS:
        normalized = None
    path = runtime_log_path(settings)
    if not path.is_file():
        return RuntimeLogResponse(path=str(path), level=normalized, lines=[])
    tail: deque[str] = deque(maxlen=safe_lines)
    include_continuation = normalized is None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            clean = line.rstrip("\n")
            if normalized is None:
                tail.append(clean)
                continue
            current = _line_level(clean)
            if current is not None:
                include_continuation = current == normalized
            if include_continuation:
                tail.append(clean)
    return RuntimeLogResponse(path=str(path), level=normalized, lines=list(tail))


def _line_level(line: str) -> str | None:
    match = LOG_LINE_RE.match(line)
    return match.group("level") if match else None
