from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_settings
from app.core.config import Settings
from app.settings.schemas import SamSettingsResponse, SamSettingsUpdate
from app.settings.service import SamSettingsService


router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("/sam", response_model=SamSettingsResponse)
def get_sam_settings(settings: Settings = Depends(get_settings)) -> SamSettingsResponse:
    return SamSettingsService(settings).get()


@router.put("/sam", response_model=SamSettingsResponse)
def update_sam_settings(payload: SamSettingsUpdate, settings: Settings = Depends(get_settings)) -> SamSettingsResponse:
    return SamSettingsService(settings).update(payload)
