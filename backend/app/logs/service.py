from __future__ import annotations

import logging
import re
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.config import Settings
from app.logs.schemas import RuntimeLogResponse


LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}
LOG_LINE_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} (?P<level>DEBUG|INFO|WARNING|ERROR)\s")
RUNTIME_LOG_MAX_BYTES = 2 * 1024 * 1024
RUNTIME_LOG_BACKUP_COUNT = 3


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
    handler = RotatingFileHandler(
        path,
        maxBytes=RUNTIME_LOG_MAX_BYTES,
        backupCount=RUNTIME_LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
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
    log_files = _runtime_log_files(path)
    if not log_files:
        return RuntimeLogResponse(path=str(path), level=normalized, lines=[])
    if normalized is None:
        return RuntimeLogResponse(path=str(path), level=None, lines=_tail_log_history(log_files, safe_lines))

    # Each retained file is capped by RotatingFileHandler, so this level-aware
    # pass across the complete rotation history remains bounded while
    # preserving stack-trace continuation lines at file boundaries.
    tail: deque[str] = deque(maxlen=safe_lines)
    include_continuation = normalized is None
    for log_file in log_files:
        with log_file.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                clean = line.rstrip("\n")
                current = _line_level(clean)
                if current is not None:
                    include_continuation = current == normalized
                if include_continuation:
                    tail.append(clean)
    return RuntimeLogResponse(path=str(path), level=normalized, lines=list(tail))


def _runtime_log_files(path: Path) -> list[Path]:
    """Return retained rotation files in chronological order."""

    return [
        candidate
        for candidate in [
            *(path.with_name(f"{path.name}.{index}") for index in range(RUNTIME_LOG_BACKUP_COUNT, 0, -1)),
            path,
        ]
        if candidate.is_file()
    ]


def _tail_log_history(log_files: list[Path], line_count: int) -> list[str]:
    """Tail multiple rotated files without reading older history unnecessarily."""

    remaining = line_count
    chunks: list[list[str]] = []
    for log_file in reversed(log_files):
        lines = _tail_log_lines(log_file, remaining)
        if lines:
            chunks.append(lines)
            remaining -= len(lines)
        if remaining <= 0:
            break
    return [line for chunk in reversed(chunks) for line in chunk][-line_count:]


def _tail_log_lines(path: Path, line_count: int) -> list[str]:
    """Read only the final complete lines without loading the active log."""

    block_size = 8192
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        chunks: list[bytes] = []
        newline_count = 0
        while position > 0 and newline_count <= line_count:
            size = min(block_size, position)
            position -= size
            handle.seek(position)
            block = handle.read(size)
            chunks.append(block)
            newline_count += block.count(b"\n")
    return b"".join(reversed(chunks)).decode("utf-8", errors="replace").splitlines()[-line_count:]


def _line_level(line: str) -> str | None:
    match = LOG_LINE_RE.match(line)
    return match.group("level") if match else None
