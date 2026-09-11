from __future__ import annotations

import json
import re
from typing import Any


MAX_LIST_ITEMS = 20
MAX_ISSUES = 15
MAX_LOG_TAIL_LINES = 40
MAX_LOG_CHARS = 4_000
MAX_TEXT_CHARS = 500
MAX_RESULT_CHARS = 12_000
MAX_ERROR_SAMPLES = 5

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def untrusted_text(value: Any, *, limit: int = MAX_TEXT_CHARS) -> str:
    """Treat dataset names, notes, logs and free text as untrusted model input."""
    text = "" if value is None else str(value)
    text = _CONTROL_CHARS.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()
    if len(text) > limit:
        return text[: max(0, limit - 3)] + "..."
    return text


def bound_log_text(value: Any, *, max_chars: int = MAX_LOG_CHARS, max_lines: int = MAX_LOG_TAIL_LINES) -> str:
    text = untrusted_text(value, limit=max_chars * 2)
    lines = text.split("\n")
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    clipped = "\n".join(lines)
    return untrusted_text(clipped, limit=max_chars)


def omit_paths(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    cleaned = dict(payload)
    for key in keys:
        cleaned.pop(key, None)
    return cleaned


def truncate_list(items: list[Any], *, limit: int = MAX_LIST_ITEMS) -> tuple[list[Any], int]:
    if len(items) <= limit:
        return items, 0
    return items[:limit], len(items) - limit


def bound_json(value: Any, *, limit: int = MAX_RESULT_CHARS) -> Any:
    """Ensure serialized tool output stays within a character budget."""
    try:
        raw = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        raw = json.dumps({"error": "tool_result_not_serializable"}, ensure_ascii=False)
    if len(raw) <= limit:
        return value
    return {
        "truncated": True,
        "original_chars": len(raw),
        "preview": raw[: max(0, limit - 80)] + "...",
    }
