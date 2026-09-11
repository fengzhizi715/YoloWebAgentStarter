"""OpenAI-compatible request auth helpers, adapted from upstream YoloWebAgent settings/llm_auth.py."""

from __future__ import annotations

import logging
import re
from typing import Literal

from app.core.errors import ValidationError

logger = logging.getLogger("ywa.agent")

LLMAuthScheme = Literal["bearer", "header", "raw_authorization"]

_VALID_SCHEMES = frozenset({"bearer", "header", "raw_authorization"})
_HEADER_NAME_PATTERN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")


def sanitize_auth_scheme(value: str | None) -> LLMAuthScheme:
    scheme = (value or "bearer").strip().lower()
    if scheme in _VALID_SCHEMES:
        return scheme  # type: ignore[return-type]
    logger.warning("Invalid auth_scheme %r, falling back to bearer", value)
    return "bearer"


def coalesce_auth_scheme(stored_scheme: str | None, auth_header_name: str) -> LLMAuthScheme:
    if stored_scheme and str(stored_scheme).strip():
        return sanitize_auth_scheme(stored_scheme)
    if auth_header_name.strip():
        return "header"
    return "bearer"


def sanitize_auth_header_name(value: str | None) -> str:
    name = (value or "").strip()
    if not name:
        return ""
    if len(name) > 128 or not _HEADER_NAME_PATTERN.match(name):
        logger.warning("Invalid auth_header_name %r, ignoring", name)
        return ""
    return name


def validate_auth_config(auth_scheme: str | None, auth_header_name: str | None) -> tuple[LLMAuthScheme, str]:
    scheme = (auth_scheme or "bearer").strip().lower()
    if scheme not in _VALID_SCHEMES:
        raise ValidationError("invalid_auth_scheme", "auth_scheme must be bearer, header, or raw_authorization.")
    header_name = (auth_header_name or "").strip()
    if header_name:
        if len(header_name) > 128 or not _HEADER_NAME_PATTERN.match(header_name):
            raise ValidationError("invalid_auth_header_name", "auth_header_name is invalid.")
    if scheme == "header" and not header_name:
        raise ValidationError("auth_header_name_required", "auth_header_name is required when auth_scheme is header.")
    if scheme != "header":
        header_name = ""
    return scheme, header_name  # type: ignore[return-value]


def build_llm_request_headers(
    api_key: str | None,
    auth_scheme: str = "bearer",
    auth_header_name: str = "",
) -> dict[str, str]:
    """Build HTTP headers for OpenAI-compatible chat completions (upstream pattern)."""
    headers = {"Content-Type": "application/json"}
    token = (api_key or "").strip()
    if not token:
        return headers

    scheme = sanitize_auth_scheme(auth_scheme)
    header_name = sanitize_auth_header_name(auth_header_name)

    if scheme == "header":
        if not header_name:
            logger.warning("auth_scheme=header but auth_header_name missing, falling back to bearer")
            scheme = "bearer"
        else:
            headers[header_name] = token
            return headers

    if scheme == "raw_authorization":
        headers["Authorization"] = token
        return headers

    headers["Authorization"] = f"Bearer {token}"
    return headers
