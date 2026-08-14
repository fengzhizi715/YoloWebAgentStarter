from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_settings
from app.core.config import Settings
from app.logs.schemas import RuntimeLogResponse
from app.logs.service import read_runtime_logs


router = APIRouter(prefix="/logs", tags=["logs"])


@router.get("/runtime", response_model=RuntimeLogResponse)
def runtime_logs(
    lines: int = Query(default=300, ge=1, le=1000),
    level: str | None = Query(default=None, min_length=1, max_length=16),
    settings: Settings = Depends(get_settings),
) -> RuntimeLogResponse:
    return read_runtime_logs(settings, lines=lines, level=level)
