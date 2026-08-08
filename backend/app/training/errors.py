from __future__ import annotations

from typing import Any

from app.core.errors import ValidationError


class TrainingConfigError(ValidationError):
    def __init__(self, error_code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(error_code, message, details=details)
