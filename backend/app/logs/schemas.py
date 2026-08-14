from __future__ import annotations

from pydantic import BaseModel


class RuntimeLogResponse(BaseModel):
    path: str
    level: str | None = None
    lines: list[str]
