from __future__ import annotations

import re
from typing import Any


_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|authorization|password|secret|token|credential|bearer)",
    re.IGNORECASE,
)
_SECRET_VALUE_RE = re.compile(
    r"(?i)(?:api[_-]?key|authorization|password|secret|token|bearer)\s*[:=]\s*([^\s,;\"']+)"
)

REDACTED = "[REDACTED]"


def looks_like_secret_key(key: str) -> bool:
    return bool(_SECRET_KEY_RE.search(key or ""))


def redact_text(value: Any, *, secrets: list[str] | None = None) -> str:
    """Remove known secret values and key=value secret patterns from log text."""
    text = "" if value is None else str(value)
    for secret in secrets or []:
        if secret and secret in text:
            text = text.replace(secret, REDACTED)
    return _SECRET_VALUE_RE.sub(lambda match: match.group(0).replace(match.group(1), REDACTED), text)


def redact_mapping(value: Any, *, secrets: list[str] | None = None) -> Any:
    """Recursively redact secret-looking keys and known secret values for safe logging."""
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if looks_like_secret_key(str(key)):
                cleaned[str(key)] = REDACTED
            else:
                cleaned[str(key)] = redact_mapping(item, secrets=secrets)
        return cleaned
    if isinstance(value, list):
        return [redact_mapping(item, secrets=secrets) for item in value]
    if isinstance(value, str):
        return redact_text(value, secrets=secrets)
    return value


def redact_tool_arguments(arguments: dict[str, Any] | None, *, secrets: list[str] | None = None) -> dict[str, Any]:
    payload = arguments if isinstance(arguments, dict) else {}
    redacted = redact_mapping(payload, secrets=secrets)
    return redacted if isinstance(redacted, dict) else {"value": redacted}
